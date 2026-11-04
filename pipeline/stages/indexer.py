"""Vector DB stage (implementation.md section 5).

Mirrors the embedding artifacts into a ChromaDB collection:

* `upsert` by `chunk_id`, so re-running the stage never duplicates rows;
* delete ids that exist in Chroma but not in `chunks.jsonl`, so an edited page
  cannot keep serving its old facts;
* validate the manifest against the current chunks and config before writing,
  because vectors from a different model (or from older chunks) are poison.

Artifacts written:
    data/chroma/                 persistent ChromaDB store
    artifacts/runs/index.json    counts, collection settings, rebuild flag

`--rebuild` / `--force` drop the collection first. Without them the stage is
incremental and idempotent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pipeline.lib.io import read_json, read_jsonl, write_json

BATCH_SIZE = 256
METADATA_KEYS = ("doc_id", "url", "title", "scheme_name", "field",
                 "source_type", "fetched_at")


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - {hint}")
    return path


def _load_inputs(cfg) -> tuple[list[dict], np.ndarray, dict]:
    """Read and cross-check chunks, vectors, id_map, and manifest."""
    processed = Path(cfg.sources.processed_dir)
    cache = Path(cfg.embedding.cache_dir)

    chunks_path = _require(processed / "chunks.jsonl",
                           "run `python -m pipeline.run --stage chunk` first")
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(
            f"{chunks_path} has no chunks - refusing to build an empty index")

    ids = [c["chunk_id"] for c in chunks]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        raise ValueError(f"duplicate chunk_id(s) in chunks.jsonl: {duplicates}")

    vectors_path = _require(cache / "embeddings.npy",
                            "run `python -m pipeline.run --stage embed` first")
    manifest_path = _require(cache / "manifest.json",
                             "run `python -m pipeline.run --stage embed` first")
    id_map_path = _require(cache / "id_map.json",
                           "run `python -m pipeline.run --stage embed` first")

    vectors = np.load(vectors_path)
    manifest = read_json(manifest_path) or {}
    id_map = read_json(id_map_path) or {}

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError(
            f"{vectors_path} has shape {vectors.shape} but chunks.jsonl has "
            f"{len(chunks)} rows - re-run `--stage embed`")

    expected_map = {str(i): cid for i, cid in enumerate(ids)}
    if id_map != expected_map:
        raise ValueError(
            f"{id_map_path} does not match chunks.jsonl order - "
            "re-run `--stage embed`")

    if manifest.get("model") != cfg.embedding.model:
        raise ValueError(
            f"embeddings were built with {manifest.get('model')!r} but config "
            f"says {cfg.embedding.model!r} - re-run `--stage embed`")
    if ("normalize" not in manifest
            or bool(manifest["normalize"]) != bool(cfg.embedding.normalize)):
        raise ValueError(
            "embedding normalization changed since the vectors were built - "
            "re-run `--stage embed`")
    if manifest.get("dimension") != vectors.shape[1]:
        raise ValueError(
            f"manifest dimension {manifest.get('dimension')} != vector "
            f"dimension {vectors.shape[1]} - re-run `--stage embed`")
    if manifest.get("chunks_hash") != _hash_file(chunks_path):
        raise ValueError(
            "chunks.jsonl changed after embedding - re-run `--stage embed` "
            "so vectors and chunks line up")

    for chunk in chunks:
        missing = [k for k in METADATA_KEYS if k not in chunk]
        if missing:
            raise ValueError(
                f"{chunk['chunk_id']}: missing metadata field(s) {missing}")
        bad = [k for k in METADATA_KEYS
               if not isinstance(chunk[k], (str, int, float, bool))]
        if bad:
            raise ValueError(
                f"{chunk['chunk_id']}: metadata must be scalar, got {bad}")

    return chunks, np.asarray(vectors, dtype=np.float32), manifest


# --------------------------------------------------------------------------- #
# ChromaDB
# --------------------------------------------------------------------------- #

def _client(cfg):
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:                     # pragma: no cover - env issue
        raise RuntimeError(
            "chromadb is not installed - run `pip install -r requirements.txt`"
        ) from exc
    return chromadb.PersistentClient(
        path=cfg.vector_db.persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_names(client) -> set[str]:
    return {c.name for c in client.list_collections()}


def _index(cfg, client, chunks, vectors) -> dict:
    name = cfg.vector_db.collection
    distance = cfg.vector_db.distance

    if name in _collection_names(client):
        collection = client.get_collection(name)
        actual = (collection.metadata or {}).get("hnsw:space")
        if actual != distance:
            raise ValueError(
                f"collection {name!r} uses {actual!r} distance but config says "
                f"{distance!r} - run `--stage index --rebuild`")
    else:
        collection = client.create_collection(
            name, metadata={"hnsw:space": distance})

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{k: c[k] for k in METADATA_KEYS} for c in chunks]

    upserted = 0
    for start in range(0, len(ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[start:end],
            embeddings=vectors[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        upserted += end - start

    existing = set(collection.get(include=[])["ids"])
    stale = sorted(existing - set(ids))
    for start in range(0, len(stale), BATCH_SIZE):
        collection.delete(ids=stale[start:start + BATCH_SIZE])

    count = collection.count()
    if count != len(ids):
        raise ValueError(
            f"collection {name!r} holds {count} rows but chunks.jsonl has "
            f"{len(ids)} - run `--stage index --rebuild`")

    return {
        "collection": name,
        "distance": distance,
        "upserted": upserted,
        "removed": len(stale),
        "count": count,
    }


# --------------------------------------------------------------------------- #
# stage entrypoint
# --------------------------------------------------------------------------- #

def run(cfg, force: bool = False, rebuild: bool = False) -> dict:
    if cfg.vector_db.provider != "chromadb":
        raise NotImplementedError(
            f"vector_db.provider={cfg.vector_db.provider!r} is not supported; "
            "only 'chromadb' is implemented")

    chunks, vectors, manifest = _load_inputs(cfg)
    client = _client(cfg)

    dropped = False
    if force or rebuild:
        if cfg.vector_db.collection in _collection_names(client):
            client.delete_collection(cfg.vector_db.collection)
            dropped = True

    result = _index(cfg, client, chunks, vectors)
    result["rebuilt"] = dropped
    result["model"] = cfg.embedding.model
    result["provider"] = cfg.vector_db.provider
    result["persist_dir"] = cfg.vector_db.persist_dir
    result["chunks"] = len(chunks)
    result["batch_size"] = BATCH_SIZE

    # The manifest's `collection` is the startup-check contract. Changing the
    # collection name in config does not invalidate vectors, so record it here
    # instead of forcing a pointless re-embed.
    if manifest.get("collection") != cfg.vector_db.collection:
        manifest["collection"] = cfg.vector_db.collection
        write_json(Path(cfg.embedding.cache_dir) / "manifest.json", manifest)
        result["collection_changed"] = True

    write_json("artifacts/runs/index.json", {"stage": "index", **result})
    return result
