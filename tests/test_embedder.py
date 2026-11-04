"""Tests for the embedding stage (implementation.md §4).

The sentence-transformer is replaced with a deterministic fake so the whole
stage - caching, manifests, normalization checks - runs in milliseconds.
The real 384-dim model is exercised by the CLI check instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.stages import embedder


class FakeModel:
    """Deterministic unit-length vectors; records calls and kwargs."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls = 0
        self.texts: list[str] = []
        self.kwargs: dict = {}

    def encode(self, texts, **kwargs):
        self.calls += 1
        self.texts.extend(texts)
        self.kwargs = kwargs
        rows = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            for i, byte in enumerate(text.encode("utf-8")):
                vector[i % self.dim] += float(byte)
            if not np.any(vector):
                vector[0] = 1.0
            if kwargs.get("normalize_embeddings"):
                vector = vector / np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows)


@pytest.fixture(autouse=True)
def _isolate_run_artifacts(monkeypatch, tmp_path):
    """Keep `artifacts/runs/embed.json` pointed at the real corpus."""
    real_write_json = embedder.write_json

    def write(path, obj):
        target = Path(path)
        if target.parts and target.parts[0] == "artifacts":
            target = tmp_path / target
        real_write_json(target, obj)

    monkeypatch.setattr(embedder, "write_json", write)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embedder, "_load_model", lambda cfg: model)
    return model


def _cfg(tmp_path, **overrides):
    embedding = dict(model="fake/model", batch_size=4, normalize=True,
                     cache_dir=str(tmp_path / "emb"))
    embedding.update(overrides)
    return SimpleNamespace(
        sources=SimpleNamespace(processed_dir=str(tmp_path / "processed")),
        embedding=SimpleNamespace(**embedding),
        vector_db=SimpleNamespace(collection="test_coll"),
    )


def _write_chunks(tmp_path, texts, ids=None):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    ids = ids or [f"c{i}" for i in range(len(texts))]
    rows = [{"chunk_id": cid, "text": text}
            for cid, text in zip(ids, texts)]
    (processed / "chunks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")


def _read(tmp_path, name):
    return json.loads((tmp_path / "emb" / name).read_text(encoding="utf-8"))


def test_run_writes_vectors_and_manifest(tmp_path, fake_model):
    _write_chunks(tmp_path, ["alpha", "beta", "gamma"])
    report = embedder.run(_cfg(tmp_path))

    assert report["written"] == 3
    assert (report["encoded"], report["reused"]) == (3, 0)
    assert report["dimension"] == 8

    vectors = np.load(tmp_path / "emb" / "embeddings.npy")
    assert vectors.shape == (3, 8)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    manifest = _read(tmp_path, "manifest.json")
    assert manifest["model"] == "fake/model"
    assert manifest["dimension"] == 8
    assert manifest["chunk_count"] == 3
    assert manifest["normalize"] is True
    assert manifest["collection"] == "test_coll"
    assert manifest["chunks_hash"].startswith("sha256:")

    assert _read(tmp_path, "id_map.json") == {
        "0": "c0", "1": "c1", "2": "c2"}
    assert set(_read(tmp_path, "chunk_hashes.json")) == {"c0", "c1", "c2"}

    assert fake_model.kwargs["normalize_embeddings"] is True
    assert fake_model.kwargs["batch_size"] == 4


def test_second_run_is_a_cache_hit(tmp_path, fake_model):
    _write_chunks(tmp_path, ["alpha", "beta", "gamma"])
    cfg = _cfg(tmp_path)
    embedder.run(cfg)
    first_built_at = _read(tmp_path, "manifest.json")["built_at"]
    first_vectors = np.load(tmp_path / "emb" / "embeddings.npy")

    report = embedder.run(cfg)

    assert report["cache_hit"] is True
    assert (report["encoded"], report["reused"]) == (0, 3)
    assert fake_model.calls == 1                  # model not even loaded
    assert _read(tmp_path, "manifest.json")["built_at"] == first_built_at
    assert np.array_equal(np.load(tmp_path / "emb" / "embeddings.npy"),
                          first_vectors)


def test_changed_chunk_reencodes_only_that_row(tmp_path, fake_model):
    _write_chunks(tmp_path, ["alpha", "beta", "gamma"])
    cfg = _cfg(tmp_path)
    embedder.run(cfg)
    before = np.load(tmp_path / "emb" / "embeddings.npy")

    _write_chunks(tmp_path, ["alpha", "BETA changed", "gamma"])
    report = embedder.run(cfg)
    after = np.load(tmp_path / "emb" / "embeddings.npy")

    assert (report["encoded"], report["reused"]) == (1, 2)
    assert fake_model.calls == 2
    assert np.array_equal(before[0], after[0])    # untouched chunk reused
    assert np.array_equal(before[2], after[2])
    assert not np.array_equal(before[1], after[1])


def test_rebuild_forces_a_full_reencode(tmp_path, fake_model):
    _write_chunks(tmp_path, ["alpha", "beta"])
    cfg = _cfg(tmp_path)
    embedder.run(cfg)

    report = embedder.run(cfg, rebuild=True)
    assert (report["encoded"], report["reused"]) == (2, 0)

    report = embedder.run(cfg, force=True)
    assert (report["encoded"], report["reused"]) == (2, 0)


def test_model_change_invalidates_the_whole_cache(tmp_path, fake_model):
    _write_chunks(tmp_path, ["alpha", "beta"])
    embedder.run(_cfg(tmp_path))

    report = embedder.run(_cfg(tmp_path, model="other/model"))
    assert (report["encoded"], report["reused"]) == (2, 0)
    assert _read(tmp_path, "manifest.json")["model"] == "other/model"


def test_normalize_false_is_respected(tmp_path, fake_model):
    _write_chunks(tmp_path, ["alpha", "beta"])
    cfg = _cfg(tmp_path, normalize=False)
    embedder.run(cfg)

    assert fake_model.kwargs["normalize_embeddings"] is False
    assert _read(tmp_path, "manifest.json")["normalize"] is False
    vectors = np.load(tmp_path / "emb" / "embeddings.npy")
    assert not np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_missing_chunks_file_explains_the_next_command(tmp_path, fake_model):
    with pytest.raises(FileNotFoundError, match="--stage chunk"):
        embedder.run(_cfg(tmp_path))


def test_empty_chunks_file_refuses_to_write_an_index(tmp_path, fake_model):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "chunks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no chunks"):
        embedder.run(_cfg(tmp_path))


def test_duplicate_chunk_ids_are_rejected(tmp_path, fake_model):
    _write_chunks(tmp_path, ["alpha", "beta"], ids=["dup", "dup"])
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        embedder.run(_cfg(tmp_path))
