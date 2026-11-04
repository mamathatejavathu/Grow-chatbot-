"""Tests for the retrieval stage (implementation.md §6.1).

The store is a real ChromaDB in `tmp_path`; only the embedding model is faked,
so the tests exercise the actual `where` filter, distance semantics, and
`n_results` handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from query import retriever as retriever_module
from query.retriever import Retriever, infer_scheme, normalize_question

DIM = 4
COLLECTION = "test_collection"
MODEL = "fake/model"

E1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def _unit(*values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def isolate_artifacts(monkeypatch, tmp_path):
    """Keep `artifacts/retrieval/*.json` out of the real repository."""
    real_write_json = retriever_module.write_json

    def write(path, obj):
        target = Path(path)
        if target.parts and target.parts[0] == "artifacts":
            target = tmp_path / target
        real_write_json(target, obj)

    monkeypatch.setattr(retriever_module, "write_json", write)


@pytest.fixture
def make_store(tmp_path):
    """Create a Chroma collection and release its files at teardown."""
    import chromadb
    from chromadb.config import Settings

    created = []

    def build(chunks, vectors, distance="cosine"):
        client = chromadb.PersistentClient(
            path=str(tmp_path / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        created.append(client)
        collection = client.create_collection(
            COLLECTION, metadata={"hnsw:space": distance})
        collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=[np.asarray(v, dtype=np.float32).tolist()
                        for v in vectors],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        return client

    yield build

    for client in created:
        try:
            client._system.stop()
        except Exception:                  # shared system already stopped
            pass


def _cfg(tmp_path, **retrieval):
    settings = dict(top_k=5, min_relevance_score=0.35,
                    filter_by_scheme=True, dedupe_by_field=True)
    settings.update(retrieval)
    return SimpleNamespace(
        embedding=SimpleNamespace(model=MODEL, normalize=True,
                                  cache_dir=str(tmp_path / "emb")),
        vector_db=SimpleNamespace(provider="chromadb",
                                  persist_dir=str(tmp_path / "chroma"),
                                  collection=COLLECTION, distance="cosine"),
        retrieval=SimpleNamespace(**settings),
    )


def _write_manifest(tmp_path, *, model=MODEL, collection=COLLECTION,
                    dimension=DIM, normalize=True):
    cache = tmp_path / "emb"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "model": model,
        "dimension": dimension,
        "normalize": normalize,
        "chunk_count": 1,
        "built_at": "2026-11-02T10:00:00Z",
        "chunks_hash": "sha256:test",
        "collection": collection,
    }), encoding="utf-8")


def _chunk(chunk_id, text, field="general",
           scheme="HDFC Flexi Cap Fund", source_type="scheme"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "doc_id": chunk_id.split(":")[0],
            "url": "https://groww.in/mutual-funds/hdfc-flexi-cap-fund",
            "title": "HDFC Flexi Cap Fund",
            "scheme_name": scheme,
            "field": field,
            "source_type": source_type,
            "fetched_at": "2026-11-02T10:00:00Z",
        },
    }


class FakeModel:
    """Return a pre-set vector for each question, recording call arguments."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        rows = [self.mapping[normalize_question(text)] for text in texts]
        return np.vstack(rows).astype(np.float32)


def _retriever(cfg, client, mapping, **kwargs):
    model = FakeModel(mapping)
    return Retriever(cfg, model=model, client=client, **kwargs), model


# --------------------------------------------------------------------------- #
# alias inference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question,expected", [
    ("expense ratio of HDFC Flexi Cap", "HDFC Flexi Cap Fund"),
    ("expense ratio of HDFC FLEXI-CAP fund", "HDFC Flexi Cap Fund"),
    ("flexi cap exit load", "HDFC Flexi Cap Fund"),
    ("ELSS lock-in", "HDFC ELSS Tax Saver Fund"),
    ("tax saver lock in period", "HDFC ELSS Tax Saver Fund"),
    ("minimum SIP HDFC Mid Cap", "HDFC Mid Cap Fund"),
    ("mid-cap fund exit load", "HDFC Mid Cap Fund"),
    ("HDFC Large Cap expense ratio", "HDFC Large Cap Fund"),
    ("large cap fund riskometer", "HDFC Large Cap Fund"),
    ("What is the expense ratio?", None),
    ("what is the weather", None),
])
def test_infer_scheme(question, expected):
    assert infer_scheme(question) == expected


def test_normalize_question_collapses_punctuation_and_whitespace():
    assert normalize_question("  HDFC   Mid_Cap  ") == "hdfc mid cap"
    assert normalize_question("HDFC\nFlexi-Cap") == "hdfc flexi cap"


# --------------------------------------------------------------------------- #
# score direction and ordering
# --------------------------------------------------------------------------- #

def test_score_is_one_minus_cosine_distance(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    chunks = [_chunk("c0", "Expense ratio: 0.75%", "expense_ratio")]
    client = make_store(chunks, [E1])

    exact, _ = _retriever(cfg, client, {"exact question": E1})
    hits, _, gate = exact.retrieve("exact question")
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert gate == "pass"

    oblique, _ = _retriever(cfg, client, {"oblique": _unit(1.0, 3 ** 0.5, 0.0, 0.0)})
    hits, _, _ = oblique.retrieve("oblique")
    assert hits[0].score == pytest.approx(0.5, abs=1e-5)   # cosine 60 degrees
    assert hits[0].chunk_id == "c0"


def test_hits_are_ordered_best_first(tmp_path, make_store):
    cfg = _cfg(tmp_path, dedupe_by_field=False)
    _write_manifest(tmp_path)
    chunks = [
        _chunk("worst", "gamma", "aum"),
        _chunk("best", "alpha", "expense_ratio"),
        _chunk("middle", "beta", "exit_load"),
    ]
    client = make_store(chunks, [
        _unit(0.2, 1.0, 0.0, 0.0),
        E1,
        _unit(0.8, 0.6, 0.0, 0.0),
    ])

    retriever, _ = _retriever(cfg, client, {"q": E1})
    hits, _, _ = retriever.retrieve("q")

    assert [h.chunk_id for h in hits] == ["best", "middle", "worst"]
    assert hits[0].score > hits[1].score > hits[2].score


def test_hit_carries_metadata_for_the_citation(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    chunk = _chunk("s03:f_exit_load:9", "Exit load: 1% within 1 year", "exit_load")
    client = make_store([chunk], [E1])

    retriever, _ = _retriever(cfg, client, {"q": E1})
    hits, _, _ = retriever.retrieve("q")

    hit = hits[0]
    assert hit.url == chunk["metadata"]["url"]
    assert hit.text == chunk["text"]
    assert hit.field == "exit_load"
    assert hit.metadata["fetched_at"] == "2026-11-02T10:00:00Z"
    assert hit.metadata["scheme_name"] == "HDFC Flexi Cap Fund"


# --------------------------------------------------------------------------- #
# scheme filter
# --------------------------------------------------------------------------- #

def test_scheme_filter_limits_hits_to_the_named_scheme(tmp_path, make_store):
    cfg = _cfg(tmp_path, dedupe_by_field=False)
    _write_manifest(tmp_path)
    chunks = [
        _chunk("c_flexi", "Exit load: 1%", "exit_load", "HDFC Flexi Cap Fund"),
        _chunk("c_mid", "Exit load: 1%", "exit_load", "HDFC Mid Cap Fund"),
        _chunk("c_elss", "Exit load: Nil", "exit_load", "HDFC ELSS Tax Saver Fund"),
    ]
    client = make_store(chunks, [E1, E1, E1])

    retriever, _ = _retriever(cfg, client, {"exit load hdfc flexi cap": E1})
    hits, filters, gate = retriever.retrieve("exit load hdfc flexi cap")

    assert filters == {"scheme_name": "HDFC Flexi Cap Fund"}
    assert [h.chunk_id for h in hits] == ["c_flexi"]
    assert gate == "pass"


def test_generic_question_searches_the_whole_corpus(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    chunks = [
        _chunk("c_flexi", "Exit load: 1%", "exit_load", "HDFC Flexi Cap Fund"),
        _chunk("c_amc", "AMC overview", "general", ""),
    ]
    client = make_store(chunks, [E1, E2])

    retriever, _ = _retriever(cfg, client, {"what is an exit load": E1})
    _, filters, _ = retriever.retrieve("what is an exit load")

    assert filters == {"scheme_name": None}


def test_filter_by_scheme_false_ignores_aliases(tmp_path, make_store):
    cfg = _cfg(tmp_path, filter_by_scheme=False, dedupe_by_field=False)
    _write_manifest(tmp_path)
    chunks = [
        _chunk("c_flexi", "Exit load: 1%", "exit_load", "HDFC Flexi Cap Fund"),
        _chunk("c_mid", "Exit load: 1%", "exit_load", "HDFC Mid Cap Fund"),
    ]
    client = make_store(chunks, [E1, E1])

    retriever, _ = _retriever(cfg, client, {"exit load hdfc flexi cap": E1})
    hits, filters, _ = retriever.retrieve("exit load hdfc flexi cap")

    assert filters == {"scheme_name": None}
    assert {h.chunk_id for h in hits} == {"c_flexi", "c_mid"}


def test_scheme_filter_with_no_matches_returns_no_hits(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    chunks = [_chunk("c_flexi", "Exit load: 1%", "exit_load",
                     "HDFC Flexi Cap Fund")]
    client = make_store(chunks, [E1])

    retriever, _ = _retriever(cfg, client, {"hdfc large cap exit load": E1})
    hits, filters, gate = retriever.retrieve("hdfc large cap exit load")

    assert filters == {"scheme_name": "HDFC Large Cap Fund"}
    assert hits == []
    assert gate == "fail"


# --------------------------------------------------------------------------- #
# dedupe and top_k
# --------------------------------------------------------------------------- #

def test_dedupe_by_field_keeps_the_best_hit_per_field(tmp_path, make_store):
    cfg = _cfg(tmp_path, dedupe_by_field=True)
    _write_manifest(tmp_path)
    chunks = [
        _chunk("general_best", "alpha", "general"),
        _chunk("general_worse", "beta", "general"),
        _chunk("exit_load", "gamma", "exit_load"),
    ]
    client = make_store(chunks, [E1, _unit(0.8, 0.6, 0.0, 0.0), E2])

    retriever, _ = _retriever(cfg, client, {"q": E1})
    hits, _, _ = retriever.retrieve("q")

    assert [h.chunk_id for h in hits] == ["general_best", "exit_load"]


def test_dedupe_disabled_keeps_every_field_repeat(tmp_path, make_store):
    cfg = _cfg(tmp_path, dedupe_by_field=False)
    _write_manifest(tmp_path)
    chunks = [
        _chunk("general_best", "alpha", "general"),
        _chunk("general_worse", "beta", "general"),
        _chunk("exit_load", "gamma", "exit_load"),
    ]
    client = make_store(chunks, [E1, _unit(0.8, 0.6, 0.0, 0.0), E2])

    retriever, _ = _retriever(cfg, client, {"q": E1})
    hits, _, _ = retriever.retrieve("q")

    assert [h.chunk_id for h in hits] == [
        "general_best", "general_worse", "exit_load"]


def test_top_k_is_capped_by_collection_size(tmp_path, make_store):
    cfg = _cfg(tmp_path, top_k=20, dedupe_by_field=False)
    _write_manifest(tmp_path)
    chunks = [_chunk("c0", "alpha"), _chunk("c1", "beta")]
    client = make_store(chunks, [E1, E2])

    retriever, _ = _retriever(cfg, client, {"q": E1})
    hits, _, _ = retriever.retrieve("q", run_id="capped")

    assert len(hits) == 2
    trace = json.loads((tmp_path / retriever.last_trace_path).read_text(
        encoding="utf-8"))
    assert trace["n_results"] == 2
    assert trace["retrieved"] == 2


# --------------------------------------------------------------------------- #
# relevance gate
# --------------------------------------------------------------------------- #

def test_gate_passes_just_above_threshold(tmp_path, make_store):
    cfg = _cfg(tmp_path, min_relevance_score=0.35)
    _write_manifest(tmp_path)
    client = make_store([_chunk("c0", "alpha")], [E1])

    retriever, _ = _retriever(cfg, client, {"q": _unit(0.4, 0.916515, 0, 0)})
    hits, _, gate = retriever.retrieve("q")

    assert hits[0].score == pytest.approx(0.4, abs=1e-4)
    assert gate == "pass"


def test_gate_fails_below_threshold(tmp_path, make_store):
    cfg = _cfg(tmp_path, min_relevance_score=0.35)
    _write_manifest(tmp_path)
    client = make_store([_chunk("c0", "alpha")], [E1])

    retriever, _ = _retriever(cfg, client, {"q": _unit(0.3, 0.953939, 0, 0)})
    hits, _, gate = retriever.retrieve("q")

    assert hits[0].score == pytest.approx(0.3, abs=1e-4)
    assert gate == "fail"


def test_gate_fails_for_unrelated_question(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    client = make_store([_chunk("c0", "Expense ratio: 0.75%")], [E1])

    retriever, _ = _retriever(cfg, client, {"what is the weather": -E1})
    hits, _, gate = retriever.retrieve("what is the weather")

    assert hits[0].score == pytest.approx(-1.0, abs=1e-5)
    assert gate == "fail"


# --------------------------------------------------------------------------- #
# trace artifact
# --------------------------------------------------------------------------- #

def test_trace_records_filters_scores_and_gate(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    chunks = [_chunk("c0", "Exit load: 1%", "exit_load")]
    client = make_store(chunks, [E1])

    retriever, _ = _retriever(cfg, client, {"exit load hdfc flexi cap": E1})
    hits, _, gate = retriever.retrieve(
        "exit load hdfc flexi cap", run_id="trace-test",
        guard={"pii": False, "advice": False, "redactions": []})

    assert retriever.last_trace_path is not None
    trace_path = tmp_path / retriever.last_trace_path
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    assert trace_path.name == "trace-test.json"
    assert trace["stage"] == "retrieve"
    assert trace["question"] == "exit load hdfc flexi cap"
    assert trace["guard"]["pii"] is False
    assert trace["filters"] == {"scheme_name": "HDFC Flexi Cap Fund"}
    assert trace["query_embedding_model"] == MODEL
    assert trace["collection"] == COLLECTION
    assert trace["distance"] == "cosine"
    assert trace["top_k"] == 5
    assert trace["relevance_gate"] == gate == "pass"
    assert trace["best_score"] == pytest.approx(hits[0].score, abs=1e-6)
    assert trace["hits"][0]["chunk_id"] == "c0"
    assert trace["hits"][0]["url"] == chunks[0]["metadata"]["url"]
    assert trace["hits"][0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_trace_can_be_disabled(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    client = make_store([_chunk("c0", "alpha")], [E1])

    retriever, _ = _retriever(cfg, client, {"q": E1}, write_traces=False)
    retriever.retrieve("q")

    assert retriever.last_trace_path is None
    assert not (tmp_path / "artifacts" / "retrieval").exists()


def test_explicit_write_overrides_instance_setting(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    client = make_store([_chunk("c0", "alpha")], [E1])

    retriever, _ = _retriever(cfg, client, {"q": E1}, write_traces=False)
    retriever.retrieve("q", run_id="forced", write_trace=True)

    assert (tmp_path / retriever.last_trace_path).is_file()


# --------------------------------------------------------------------------- #
# startup contract checks
# --------------------------------------------------------------------------- #

def test_query_embedding_uses_the_configured_normalize_flag(
        tmp_path, make_store):
    cfg = _cfg(tmp_path)
    cfg.embedding.normalize = False
    _write_manifest(tmp_path, normalize=False)
    client = make_store([_chunk("c0", "alpha")], [E1])

    retriever, model = _retriever(cfg, client, {"q": E1})
    retriever.retrieve("q")

    assert model.calls[0]["normalize_embeddings"] is False


def test_missing_manifest_names_the_ingest_command(tmp_path):
    with pytest.raises(RuntimeError, match="--stage all"):
        Retriever(_cfg(tmp_path), model=FakeModel({}), client=object())


def test_model_mismatch_requires_reembedding(tmp_path):
    _write_manifest(tmp_path, model="other/model")
    with pytest.raises(RuntimeError, match="--stage embed"):
        Retriever(_cfg(tmp_path), model=FakeModel({}), client=object())


def test_normalize_mismatch_is_rejected(tmp_path):
    _write_manifest(tmp_path, normalize=False)
    with pytest.raises(RuntimeError, match="normalize"):
        Retriever(_cfg(tmp_path), model=FakeModel({}), client=object())


def test_collection_mismatch_requires_reindexing(tmp_path):
    _write_manifest(tmp_path, collection="old_collection")
    with pytest.raises(RuntimeError, match="--stage index"):
        Retriever(_cfg(tmp_path), model=FakeModel({}), client=object())


def test_missing_collection_is_reported(tmp_path):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=Settings(anonymized_telemetry=False))
    try:
        with pytest.raises(RuntimeError, match="not found"):
            Retriever(cfg, model=FakeModel({}), client=client)
    finally:
        client._system.stop()


def test_non_cosine_collection_is_rejected(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    client = make_store([_chunk("c0", "alpha")], [E1], distance="l2")

    with pytest.raises(RuntimeError, match="cosine"):
        Retriever(cfg, model=FakeModel({}), client=client)


def test_query_dimension_must_match_the_manifest(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path, dimension=4)
    client = make_store([_chunk("c0", "alpha")], [E1])

    model = FakeModel({"q": np.array([1.0, 0.0, 0.0], dtype=np.float32)})
    retriever = Retriever(cfg, model=model, client=client)

    with pytest.raises(RuntimeError, match="dimensions"):
        retriever.retrieve("q")


def test_close_is_idempotent(tmp_path, make_store):
    cfg = _cfg(tmp_path)
    _write_manifest(tmp_path)
    client = make_store([_chunk("c0", "alpha")], [E1])

    retriever, _ = _retriever(cfg, client, {"q": E1})
    retriever.close()
    retriever.close()                       # shared system already stopped
