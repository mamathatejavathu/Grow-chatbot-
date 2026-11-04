"""Retrieval stage (implementation.md §6.1).

Question -> scheme alias -> query vector -> Chroma top-k -> `Hit` list + gate.

Design rules:

* **Score direction.** Chroma returns a *distance* (lower is better); the gate
  and everything downstream speak *similarity* (higher is better). The
  conversion `score = 1 - distance` happens exactly once, here. Inverting it
  makes the gate pass everything or nothing, and the symptom looks like a bad
  embedding model.
* **Explicit query embeddings.** Never `query_texts`: Chroma would download
  and use its own default embedder, whose space is not comparable with the
  vectors built in §4.
* **Scheme filter is an alias lookup, one place.** `"HDFC ELSS"` never equals
  the stored `"HDFC ELSS Tax Saver Fund"`, so questions are mapped through
  `SCHEMES` before the `where` clause is built.
* **Contract checks at startup.** The manifest must describe this exact model
  and collection, because vectors from another model (or a renamed collection
  that was never indexed) are unusable.

Artifacts written (unless disabled):
    artifacts/retrieval/<run_id>.json    question, filters, every hit + score, gate

The question written to the trace must already be redacted by `query/guard.py`;
the retriever runs *after* the PII guard (architecture.md §7.1).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from pipeline.lib.io import read_json, write_json
from pipeline.lib.runid import new_run_id
from pipeline.lib.schemas import Hit

# Canonical names exactly as stored in Chroma's `scheme_name` metadata.
# Insertion order is the tie-break when a question mentions two schemes.
SCHEMES = {
    "hdfc large cap": "HDFC Large Cap Fund",
    "large cap": "HDFC Large Cap Fund",
    "hdfc flexi": "HDFC Flexi Cap Fund",
    "flexi cap": "HDFC Flexi Cap Fund",
    "elss": "HDFC ELSS Tax Saver Fund",
    "tax saver": "HDFC ELSS Tax Saver Fund",
    "hdfc mid cap": "HDFC Mid Cap Fund",
    "mid cap": "HDFC Mid Cap Fund",
}


def normalize_question(text: str) -> str:
    """Lowercase, treat hyphens/underscores as spaces, collapse whitespace.

    `"Mid-Cap"`, `"mid cap"` and `"MID_CAP"` must resolve to the same scheme.
    """
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())


def infer_scheme(question: str) -> str | None:
    """Return the canonical `scheme_name` mentioned in the question, if any."""
    normalized = normalize_question(question)
    for alias, name in SCHEMES.items():
        if alias in normalized:
            return name
    return None


def _load_model(cfg):
    """Reuse the embedder's loader so both stages load the model identically."""
    from pipeline.stages.embedder import _load_model as load

    return load(cfg)


def _load_manifest(cfg) -> dict:
    """FR-3.5 startup check: the index must have been built from this config."""
    path = Path(cfg.embedding.cache_dir) / "manifest.json"
    manifest = read_json(path)
    if not manifest:
        raise RuntimeError(
            f"{path} missing - run `python -m pipeline.run --stage all` first")

    if manifest.get("model") != cfg.embedding.model:
        raise RuntimeError(
            f"index was built with {manifest.get('model')!r} but config says "
            f"{cfg.embedding.model!r} - re-run `--stage embed` and `--stage index`")
    if ("normalize" not in manifest
            or bool(manifest["normalize"]) != bool(cfg.embedding.normalize)):
        raise RuntimeError(
            "index normalization differs from config.embedding.normalize - "
            "re-run `--stage embed` and `--stage index`")
    if manifest.get("collection") != cfg.vector_db.collection:
        raise RuntimeError(
            f"index manifest points at collection {manifest.get('collection')!r} "
            f"but config says {cfg.vector_db.collection!r} - "
            "run `python -m pipeline.run --stage index`")
    return manifest


def _open_collection(cfg, client=None):
    """Open Chroma with telemetry disabled and verify the distance metric."""
    if client is None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:             # pragma: no cover - env issue
            raise RuntimeError(
                "chromadb is not installed - run "
                "`pip install -r requirements.txt`") from exc
        client = chromadb.PersistentClient(
            path=cfg.vector_db.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    names = {c.name for c in client.list_collections()}
    if cfg.vector_db.collection not in names:
        raise RuntimeError(
            f"collection {cfg.vector_db.collection!r} not found in "
            f"{cfg.vector_db.persist_dir!r} - run "
            "`python -m pipeline.run --stage index`")

    collection = client.get_collection(cfg.vector_db.collection)
    distance = (collection.metadata or {}).get("hnsw:space")

    # `1 - distance` is a similarity only under cosine. An L2 collection must
    # be rebuilt rather than silently scored with the wrong formula.
    if distance != "cosine":
        raise RuntimeError(
            f"collection {cfg.vector_db.collection!r} uses {distance!r} "
            "distance; retrieval scores require cosine - run "
            "`python -m pipeline.run --stage index --rebuild`")
    if cfg.vector_db.distance != "cosine":
        raise RuntimeError(
            f"vector_db.distance={cfg.vector_db.distance!r} - retrieval "
            "requires 'cosine'")

    return client, collection


class Retriever:
    """Top-k retrieval with scheme filtering, field dedupe, and a relevance gate.

    `model` and `client` are injectable so tests can run without downloading
    MiniLM or touching the real Chroma store.
    """

    def __init__(self, cfg, model=None, client=None, write_traces: bool = True):
        self.cfg = cfg
        self.manifest = _load_manifest(cfg)
        self.client, self.collection = _open_collection(cfg, client)
        self.model = model if model is not None else _load_model(cfg)
        self.write_traces = write_traces
        self.last_trace_path: str | None = None

    # ------------------------------------------------------------------ #
    # embedding
    # ------------------------------------------------------------------ #

    def embed(self, question: str):
        """Encode one question with the *same* normalize flag as ingestion."""
        import numpy as np

        vector = self.model.encode(
            [question],
            normalize_embeddings=self.cfg.embedding.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise RuntimeError(
                f"embedding model returned shape {vector.shape}; expected (1, dim)")

        expected = self.manifest.get("dimension")
        if expected and vector.shape[1] != expected:
            raise RuntimeError(
                f"query vector has {vector.shape[1]} dimensions but the index "
                f"uses {expected} - re-run `--stage embed` and `--stage index`")
        return vector[0]

    # ------------------------------------------------------------------ #
    # retrieval
    # ------------------------------------------------------------------ #

    def retrieve(self, question: str, *, run_id: str | None = None,
                 guard: dict | None = None,
                 write_trace: bool | None = None):
        """Return `(hits, filters, gate)` for one (already redacted) question.

        `hits` is ordered by descending similarity. `filters` is recorded even
        when empty so the trace shows *why* an AMC chunk could or could not be
        returned. `gate` is `"pass"` only when the best hit clears
        `retrieval.min_relevance_score`.
        """
        started = dt.datetime.now(dt.timezone.utc)
        scheme = (infer_scheme(question)
                  if self.cfg.retrieval.filter_by_scheme else None)
        filters = {"scheme_name": scheme}

        vector = self.embed(question)

        count = self.collection.count()
        if count <= 0:
            raise RuntimeError(
                f"collection {self.cfg.vector_db.collection!r} is empty - "
                "run `python -m pipeline.run --stage index`")
        n_results = min(self.cfg.retrieval.top_k, count)

        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=n_results,
            where={"scheme_name": scheme} if scheme else None,
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0] if result["ids"] else []
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        hits = [
            Hit(
                chunk_id=chunk_id,
                score=max(-1.0, min(1.0, 1.0 - float(distance))),
                field=metadata.get("field", ""),
                url=metadata.get("url", ""),
                text=document,
                metadata=dict(metadata),
            )
            for chunk_id, document, metadata, distance
            in zip(ids, documents, metadatas, distances)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        retrieved = len(hits)

        if self.cfg.retrieval.dedupe_by_field:
            seen: set[str] = set()
            deduped = []
            for hit in hits:                   # already best-first
                if hit.field in seen:
                    continue
                seen.add(hit.field)
                deduped.append(hit)
            hits = deduped

        threshold = self.cfg.retrieval.min_relevance_score
        gate = "pass" if hits and hits[0].score >= threshold else "fail"

        should_write = self.write_traces if write_trace is None else write_trace
        if should_write:
            self.last_trace_path = self._write_trace(
                question=question,
                run_id=run_id or new_run_id(),
                guard=guard,
                filters=filters,
                hits=hits,
                retrieved=retrieved,
                gate=gate,
                n_results=n_results,
                started=started,
            )

        return hits, filters, gate

    # ------------------------------------------------------------------ #
    # trace artifact
    # ------------------------------------------------------------------ #

    def _write_trace(self, *, question, run_id, guard, filters, hits, retrieved,
                     gate, n_results, started) -> str:
        finished = dt.datetime.now(dt.timezone.utc)
        path = Path("artifacts") / "retrieval" / f"{run_id}.json"
        write_json(path, {
            "stage": "retrieve",
            "run_id": run_id,
            "question": question,
            "guard": guard,
            "query_embedding_model": self.cfg.embedding.model,
            "collection": self.cfg.vector_db.collection,
            "distance": "cosine",
            "filters": filters,
            "top_k": self.cfg.retrieval.top_k,
            "n_results": n_results,
            "retrieved": retrieved,
            "dedupe_by_field": self.cfg.retrieval.dedupe_by_field,
            "min_relevance_score": self.cfg.retrieval.min_relevance_score,
            "best_score": round(hits[0].score, 6) if hits else None,
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.score, 6),
                    "field": hit.field,
                    "url": hit.url,
                    "text": hit.text,
                    "metadata": hit.metadata,
                }
                for hit in hits
            ],
            "relevance_gate": gate,
            "created_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "latency_ms": int(
                (finished - started).total_seconds() * 1000),
        })
        return path.as_posix()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Release Chroma's files so Windows can delete `persist_dir`."""
        try:
            self.client._system.stop()
        except AttributeError:                  # shared system already stopped
            pass
