"""Tests for the vector DB stage (implementation.md §5).

These run against a real ChromaDB in `tmp_path`: the point of the stage is the
store, so a fake would test nothing. The autouse fixture releases Chroma's
system at teardown, otherwise Windows keeps the HNSW files locked and pytest
cannot delete `tmp_path`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.stages import indexer

DIM = 4
COLLECTION = "test_collection"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def chroma_cleanup(monkeypatch):
    """Record every client and stop its shared system after the test."""
    created = []
    real_client = indexer._client

    def factory(cfg):
        client = real_client(cfg)
        created.append(client)
        return client

    monkeypatch.setattr(indexer, "_client", factory)
    yield
    for client in created:
        try:
            client._system.stop()
        except Exception:                      # shared system already stopped
            pass


@pytest.fixture(autouse=True)
def isolate_run_artifacts(monkeypatch, tmp_path):
    real_write_json = indexer.write_json

    def write(path, obj):
        target = Path(path)
        if target.parts and target.parts[0] == "artifacts":
            target = tmp_path / target
        real_write_json(target, obj)

    monkeypatch.setattr(indexer, "write_json", write)


def _cfg(tmp_path, model="fake/model", **overrides):
    vector_db = dict(provider="chromadb", persist_dir=str(tmp_path / "chroma"),
                     collection=COLLECTION, distance="cosine")
    vector_db.update(overrides)
    return SimpleNamespace(
        sources=SimpleNamespace(processed_dir=str(tmp_path / "processed")),
        embedding=SimpleNamespace(cache_dir=str(tmp_path / "emb"),
                                  model=model, normalize=True),
        vector_db=SimpleNamespace(**vector_db),
    )


def _chunk(chunk_id, text, field="general"):
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split(":")[0],
        "url": "https://example.test/fund",
        "title": "HDFC Test Fund",
        "scheme_name": "HDFC Test Fund",
        "field": field,
        "source_type": "scheme",
        "strategy": "field_aware",
        "text": text,
        "position": 0,
        "fetched_at": "2026-11-02T10:00:00Z",
    }


def _vectors(count, dim=DIM):
    if count == 0:
        return np.empty((0, dim), dtype=np.float32)
    rows = []
    for i in range(count):
        row = np.zeros(dim, dtype=np.float32)
        row[i % dim] = 1.0
        row[(i + 1) % dim] = 0.5
        rows.append(row / np.linalg.norm(row))
    return np.vstack(rows).astype(np.float32)


def _write_artifacts(tmp_path, chunks, vectors=None, model="fake/model",
                     collection=COLLECTION, normalize=True):
    processed = tmp_path / "processed"
    cache = tmp_path / "emb"
    processed.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    chunks_path = processed / "chunks.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks),
        encoding="utf-8")

    if vectors is None:
        vectors = _vectors(len(chunks))
    np.save(cache / "embeddings.npy", vectors)

    ids = [c["chunk_id"] for c in chunks]
    (cache / "id_map.json").write_text(
        json.dumps({str(i): cid for i, cid in enumerate(ids)}), encoding="utf-8")
    (cache / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "model": model,
        "dimension": int(vectors.shape[1]),
        "normalize": normalize,
        "chunk_count": len(chunks),
        "built_at": "2026-11-02T10:00:00Z",
        "chunks_hash": "sha256:" + hashlib.sha256(
            chunks_path.read_bytes()).hexdigest(),
        "collection": collection,
    }), encoding="utf-8")
    return vectors


def _collection(cfg):
    client = indexer._client(cfg)
    return client.get_collection(cfg.vector_db.collection)


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #

def test_index_creates_collection_matching_chunks(tmp_path):
    cfg = _cfg(tmp_path)
    chunks = [
        _chunk("c0", "Expense ratio: 1.02%", "expense_ratio"),
        _chunk("c1", "Exit load: Nil", "exit_load"),
        _chunk("c2", "Riskometer: Very High Risk", "riskometer"),
    ]
    vectors = _write_artifacts(tmp_path, chunks)

    report = indexer.run(cfg)

    assert report["count"] == 3
    assert report["upserted"] == 3
    assert report["removed"] == 0
    assert report["rebuilt"] is False

    collection = _collection(cfg)
    assert collection.count() == 3
    assert collection.metadata["hnsw:space"] == "cosine"

    got = collection.get(ids=[c["chunk_id"] for c in chunks],
                         include=["documents", "metadatas", "embeddings"])
    assert got["documents"] == [c["text"] for c in chunks]
    assert [m["field"] for m in got["metadatas"]] == [
        "expense_ratio", "exit_load", "riskometer"]
    assert got["metadatas"][0]["url"] == "https://example.test/fund"

    result = collection.query(query_embeddings=[vectors[0].tolist()],
                              n_results=1, include=["distances"])
    assert result["ids"][0] == ["c0"]
    assert result["distances"][0][0] == pytest.approx(0.0, abs=1e-5)


def test_rerun_is_idempotent(tmp_path):
    cfg = _cfg(tmp_path)
    chunks = [_chunk("c0", "alpha"), _chunk("c1", "beta")]
    _write_artifacts(tmp_path, chunks)

    indexer.run(cfg)
    report = indexer.run(cfg)

    assert report["count"] == 2
    assert report["upserted"] == 2
    assert report["removed"] == 0
    assert _collection(cfg).count() == 2


def test_changed_chunk_replaces_text_and_deletes_stale_ids(tmp_path):
    cfg = _cfg(tmp_path)
    _write_artifacts(tmp_path, [_chunk("c0", "old"), _chunk("c1", "keep")])
    indexer.run(cfg)

    _write_artifacts(tmp_path, [_chunk("c1", "keep"), _chunk("c2", "new")])
    report = indexer.run(cfg)

    assert report["count"] == 2
    assert report["removed"] == 1

    got = _collection(cfg).get(include=["documents"])
    assert sorted(got["ids"]) == ["c1", "c2"]
    assert got["documents"][got["ids"].index("c2")] == "new"


# --------------------------------------------------------------------------- #
# rebuild / distance
# --------------------------------------------------------------------------- #

def test_rebuild_drops_extra_rows(tmp_path):
    cfg = _cfg(tmp_path)
    chunks = [_chunk("c0", "alpha"), _chunk("c1", "beta")]
    vectors = _write_artifacts(tmp_path, chunks)
    indexer.run(cfg)

    collection = _collection(cfg)
    collection.upsert(ids=["stale"], embeddings=[vectors[0].tolist()],
                      documents=["stale"], metadatas=[{"field": "general"}])
    assert collection.count() == 3

    report = indexer.run(cfg, rebuild=True)

    assert report["rebuilt"] is True
    assert report["count"] == 2
    assert _collection(cfg).count() == 2
    assert "stale" not in _collection(cfg).get(include=[])["ids"]


def test_existing_collection_with_other_distance_requires_rebuild(tmp_path):
    cfg = _cfg(tmp_path)
    _write_artifacts(tmp_path, [_chunk("c0", "alpha")])

    client = indexer._client(cfg)
    client.create_collection(COLLECTION, metadata={"hnsw:space": "l2"})

    with pytest.raises(ValueError, match="--rebuild"):
        indexer.run(cfg)

    report = indexer.run(cfg, rebuild=True)
    assert report["rebuilt"] is True
    assert _collection(cfg).metadata["hnsw:space"] == "cosine"


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def test_missing_chunks_file_names_the_next_stage(tmp_path):
    with pytest.raises(FileNotFoundError, match="--stage chunk"):
        indexer.run(_cfg(tmp_path))


def test_empty_chunks_is_refused(tmp_path):
    _write_artifacts(tmp_path, [])
    with pytest.raises(ValueError, match="no chunks"):
        indexer.run(_cfg(tmp_path))


def test_chunks_changed_after_embedding_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    chunks = [_chunk("c0", "alpha")]
    _write_artifacts(tmp_path, chunks)

    chunks_path = tmp_path / "processed" / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8")
                           + json.dumps(_chunk("c1", "beta")) + "\n",
                           encoding="utf-8")

    with pytest.raises(ValueError, match="--stage embed"):
        indexer.run(cfg)


def test_model_mismatch_is_rejected(tmp_path):
    _write_artifacts(tmp_path, [_chunk("c0", "alpha")], model="fake/model")
    with pytest.raises(ValueError, match="re-run `--stage embed`"):
        indexer.run(_cfg(tmp_path, model="other/model"))


def test_id_map_order_must_match_chunks(tmp_path):
    cfg = _cfg(tmp_path)
    _write_artifacts(tmp_path, [_chunk("c0", "alpha"), _chunk("c1", "beta")])
    (tmp_path / "emb" / "id_map.json").write_text(
        json.dumps({"0": "c1", "1": "c0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        indexer.run(cfg)


def test_vector_row_count_must_match_chunks(tmp_path):
    cfg = _cfg(tmp_path)
    _write_artifacts(tmp_path, [_chunk("c0", "alpha"), _chunk("c1", "beta")],
                     vectors=_vectors(1))
    with pytest.raises(ValueError, match="chunks.jsonl has 2 rows"):
        indexer.run(cfg)


def test_unknown_provider_is_rejected(tmp_path):
    cfg = _cfg(tmp_path, provider="pinecone")
    _write_artifacts(tmp_path, [_chunk("c0", "alpha")])
    with pytest.raises(NotImplementedError, match="pinecone"):
        indexer.run(cfg)


def test_collection_rename_updates_the_manifest(tmp_path):
    cfg = _cfg(tmp_path, collection="new_collection")
    _write_artifacts(tmp_path, [_chunk("c0", "alpha")], collection="old_name")

    report = indexer.run(cfg)

    assert report["collection"] == "new_collection"
    assert report["collection_changed"] is True
    manifest = json.loads(
        (tmp_path / "emb" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["collection"] == "new_collection"
