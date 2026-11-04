"""Embedding stage (implementation.md section 4).

Turns `data/processed/chunks.jsonl` into unit-length float32 vectors and keeps
the model identity in a manifest so query time can refuse to mix models.

Artifacts written:
    data/embeddings/embeddings.npy      (n_chunks, dim) float32
    data/embeddings/id_map.json         {"0": "s01:f_general:0", ...}
    data/embeddings/chunk_hashes.json   {chunk_id: "sha256:..."}
    data/embeddings/manifest.json       model identity + dimensions
    artifacts/runs/embed.json           encode/reuse counts for this run

Caching: a vector is reused only when its chunk_id **and** the sha256 of the
chunk text are unchanged and the manifest model/normalize settings match
(architecture.md §9.4). Changing one page therefore re-encodes only its
chunks, not the whole corpus.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np

from pipeline.lib.io import read_json, read_jsonl, write_json

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_model(cfg):
    """Load the sentence-transformer, with an actionable error if it is not."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:                     # pragma: no cover - env issue
        raise RuntimeError(
            "sentence-transformers is not installed - run "
            "`pip install -r requirements.txt`") from exc
    try:
        return SentenceTransformer(cfg.embedding.model)
    except Exception as exc:                       # pragma: no cover - network/disk
        raise RuntimeError(
            f"could not load embedding model {cfg.embedding.model!r}: {exc}\n"
            "The first run downloads it from Hugging Face; check the network "
            "or point embedding.model at a local model directory.") from exc


def _encode(model, texts: list[str], cfg) -> np.ndarray:
    """Encode with an explicit `normalize_embeddings` - never rely on defaults."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = model.encode(
        texts,
        batch_size=cfg.embedding.batch_size,
        normalize_embeddings=cfg.embedding.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _read_cache(cache_dir: Path):
    """Return (manifest, vectors, id_map, chunk_hashes); missing pieces are None."""
    manifest = read_json(cache_dir / "manifest.json")
    id_map = read_json(cache_dir / "id_map.json")
    chunk_hashes = read_json(cache_dir / "chunk_hashes.json")

    vectors = None
    npy = cache_dir / "embeddings.npy"
    if npy.exists():
        try:
            vectors = np.load(npy)
        except (OSError, ValueError, EOFError):     # corrupt/truncated -> rebuild
            vectors = None
    return manifest, vectors, id_map, chunk_hashes


def _reusable(manifest, vectors, id_map, chunk_hashes, cfg):
    """Map chunk_id -> (text_hash, row) for cache entries we can trust.

    The cache is only trusted when the model and normalization settings match
    and the shapes line up. Anything else is silently dropped and re-encoded.
    """
    if not isinstance(manifest, dict) or vectors is None:
        return {}
    if manifest.get("model") != cfg.embedding.model:
        return {}
    if ("normalize" not in manifest
            or bool(manifest["normalize"]) != bool(cfg.embedding.normalize)):
        return {}
    if not isinstance(id_map, dict) or not isinstance(chunk_hashes, dict):
        return {}
    if vectors.ndim != 2 or vectors.shape[0] != len(id_map):
        return {}
    if manifest.get("dimension") != vectors.shape[1]:
        return {}

    usable: dict[str, tuple[str, np.ndarray]] = {}
    for index, chunk_id in id_map.items():
        try:
            row = int(index)
        except (TypeError, ValueError):
            continue
        if not 0 <= row < vectors.shape[0]:
            continue
        text_hash = chunk_hashes.get(chunk_id)
        if isinstance(text_hash, str) and isinstance(chunk_id, str):
            usable[chunk_id] = (text_hash, vectors[row])
    return usable


# --------------------------------------------------------------------------- #
# stage entrypoint
# --------------------------------------------------------------------------- #

def run(cfg, force: bool = False, rebuild: bool = False) -> dict:
    processed = Path(cfg.sources.processed_dir)
    chunks_path = processed / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} missing - run `python -m pipeline.run --stage chunk` "
            "first")

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(
            f"{chunks_path} has no chunks - refusing to build an empty index")

    chunk_ids = [c["chunk_id"] for c in chunks]
    duplicates = sorted({cid for cid in chunk_ids if chunk_ids.count(cid) > 1})
    if duplicates:
        raise ValueError(f"duplicate chunk_id(s) in chunks.jsonl: {duplicates}")

    texts = [c["text"] for c in chunks]
    text_hashes = [_hash_text(text) for text in texts]
    chunks_hash = _hash_file(chunks_path)

    cache_dir = Path(cfg.embedding.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    old_manifest, old_vectors, old_map, old_hashes = _read_cache(cache_dir)
    reusable = {} if (force or rebuild) else _reusable(
        old_manifest, old_vectors, old_map, old_hashes, cfg)

    missing = [i for i, (chunk_id, text_hash) in enumerate(
        zip(chunk_ids, text_hashes))
        if reusable.get(chunk_id, (None, None))[0] != text_hash]

    if missing:
        model = _load_model(cfg)
        fresh = _encode(model, [texts[i] for i in missing], cfg)
        for row, index in zip(fresh, missing):
            reusable[chunk_ids[index]] = (text_hashes[index], row)
        dimension = int(fresh.shape[1])
    else:
        dimension = int(reusable[chunk_ids[0]][1].shape[0]) if chunk_ids else 0

    for chunk_id in chunk_ids:
        row = reusable[chunk_id][1]
        if row.shape[0] != dimension:
            raise ValueError(
                f"cached dimension {row.shape[0]} != new dimension {dimension} "
                f"for {chunk_id} - run with --rebuild")

    vectors = np.vstack([reusable[chunk_id][1] for chunk_id in chunk_ids]
                        ).astype(np.float32)

    if cfg.embedding.normalize:
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "embedding rows are not unit length after normalization "
                f"(min={norms.min():.3f}, max={norms.max():.3f}) - "
                "check `normalize_embeddings` in the model config")

    np.save(cache_dir / "embeddings.npy", vectors)
    write_json(cache_dir / "id_map.json",
               {str(i): chunk_id for i, chunk_id in enumerate(chunk_ids)})
    write_json(cache_dir / "chunk_hashes.json",
               dict(zip(chunk_ids, text_hashes)))

    previous_built_at = (old_manifest or {}).get("built_at")
    unchanged = (not missing
                 and (old_manifest or {}).get("chunks_hash") == chunks_hash
                 and (old_manifest or {}).get("chunk_count") == len(chunks))
    built_at = previous_built_at if unchanged and previous_built_at else \
        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": cfg.embedding.model,
        "dimension": dimension,
        "normalize": cfg.embedding.normalize,
        "chunk_count": len(chunks),
        "built_at": built_at,
        "chunks_hash": chunks_hash,
        "collection": cfg.vector_db.collection,
    }
    write_json(cache_dir / "manifest.json", manifest)

    report = {
        "stage": "embed",
        "written": len(chunks),
        "dim": dimension,
        "model": cfg.embedding.model,
        "dimension": dimension,
        "chunks": len(chunks),
        "encoded": len(missing),
        "reused": len(chunks) - len(missing),
        "cache_hit": not missing,
        "normalize": cfg.embedding.normalize,
        "chunks_hash": chunks_hash,
        "built_at": built_at,
        "artifacts": {
            "vectors": str(cache_dir / "embeddings.npy"),
            "id_map": str(cache_dir / "id_map.json"),
            "manifest": str(cache_dir / "manifest.json"),
        },
    }
    write_json("artifacts/runs/embed.json", report)
    return report
