"""Data contracts shared by the ingest pipeline and the query service.

Every stage boundary is a typed object (see implementation.md §1.6). When the
chunker forgets to carry `scheme_name`, this fails at write time instead of
leaving a silent None in Chroma metadata that breaks filtering at retrieval.
"""

from dataclasses import asdict, dataclass


@dataclass
class Document:
    doc_id: str
    url: str
    title: str
    source_type: str        # scheme | amc | education | regulator | factsheet
    scheme_name: str
    fetched_at: str
    content_hash: str
    text: str

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    scheme_name: str
    field: str              # closed vocabulary, see implementation.md §3.2
    source_type: str
    strategy: str
    text: str
    position: int
    fetched_at: str

    def to_json(self) -> dict:
        return asdict(self)

    def metadata(self) -> dict:
        """Flat scalar metadata for ChromaDB — nested values are rejected."""
        return {k: getattr(self, k) for k in
                ("doc_id", "url", "title", "scheme_name", "field",
                 "source_type", "fetched_at")}


@dataclass
class Hit:
    chunk_id: str
    score: float
    field: str
    url: str
    text: str
    metadata: dict
