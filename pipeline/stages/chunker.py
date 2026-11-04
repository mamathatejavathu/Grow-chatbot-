"""Chunking stage (implementation.md §3).

Two strategies, selected by `chunking.strategy` (or automatically):

* ``field_aware`` - scheme pages are key/value fact tables. One chunk per
  fact (expense ratio, exit load, AUM, ...), then the leftover prose, return
  tables, and holdings are grouped into ``general`` chunks.
* ``recursive`` - prose pages (the AMC overview) are grouped by whole lines.

`chunk_size` is measured in **characters**. A line is never split, so one
long line can exceed the budget; a fact is never truncated to fit.

Artifacts written:
    data/processed/chunks.jsonl
    data/processed/chunks_preview.md
    artifacts/runs/load_documents.json
    artifacts/runs/chunk.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pipeline.lib.documents import load_documents
from pipeline.lib.fields import (
    GLOSSARY_TERMS,
    LABELS,
    VOCABULARY,
    Fact,
    header_facts,
    looks_like_definition,
    normalize_label,
    priority_for,
    valid_value,
)
from pipeline.lib.io import write_json, write_jsonl
from pipeline.lib.schemas import Chunk


# --------------------------------------------------------------------------- #
# strategy
# --------------------------------------------------------------------------- #

def choose(doc: dict, cfg) -> tuple[str, str]:
    """Pick a chunker and record why, so `chunk.json` is self-explaining."""
    if cfg.chunking.strategy != "auto":
        return cfg.chunking.strategy, "forced by config"
    if doc["source_type"] == "scheme":
        return "field_aware", "scheme page is a key-value fact table"
    return "recursive", f"{doc['source_type']} is prose"


# --------------------------------------------------------------------------- #
# field-aware
# --------------------------------------------------------------------------- #

def parse_pairs(lines: list[str]) -> list[Fact]:
    """Find `label` / next-line-value pairs, including glossary definitions.

    The value must be the immediately next line. Sliding past a blank line is
    how a chunker accidentally attaches the fund manager to the expense ratio.
    """
    facts: list[Fact] = []
    for i, raw in enumerate(lines):
        label = normalize_label(raw)
        if label not in LABELS and label not in GLOSSARY_TERMS:
            continue
        if i + 1 >= len(lines):
            continue
        value = lines[i + 1].strip()
        if not valid_value(value):
            continue

        display = raw.strip().rstrip(":").strip()
        if label in GLOSSARY_TERMS and looks_like_definition(value):
            facts.append(Fact("definition", display, value, i, 1))
            continue
        field = LABELS.get(label)
        if field is None:
            continue
        facts.append(Fact(field, display, value, i, priority_for(label)))
    return facts


def best_facts(facts: list[Fact]) -> list[Fact]:
    """One fact per field; definitions are kept one per term.

    Ties go to the last occurrence. That is the rule that makes the real
    ``Exit load: Nil`` beat the glossary definition earlier on the page.
    """
    chosen: dict[tuple[str, str], Fact] = {}
    for fact in facts:
        key = ((fact.field, fact.label.lower())
               if fact.field == "definition" else (fact.field, ""))
        current = chosen.get(key)
        if current is None or (fact.priority, fact.line) >= (current.priority,
                                                             current.line):
            chosen[key] = fact
    return sorted(chosen.values(), key=lambda f: (f.line, f.field))


def _skip_block(lines: list[str], i: int) -> int:
    """Return the index after the block starting at `i`.

    Two Groww blocks are deliberately not chunked:

    * ``Fund management`` - the manager names are a field chunk; the bios and
      the 20-40 scheme "Also manages" list are retrieval noise.
    * ``About`` - it repeats the **AMC-level** AUM on every scheme page, which
      contradicts the scheme-level ``Fund size (AUM)`` fact.
    """
    label = normalize_label(lines[i])
    if label == "fund management":
        while i < len(lines) and normalize_label(lines[i]) not in {
                "about", "investment objective"}:
            i += 1
        return i
    if label == "about":
        j = i + 1
        while j < len(lines) and lines[j].strip() not in {
                ";", "Investment Objective"}:
            j += 1
        return j + 1 if j < len(lines) and lines[j].strip() == ";" else j
    return i + 1


def residual_lines(lines: list[str], consumed: set[int]) -> list[str]:
    """Text left after the field facts have been taken out."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        if normalize_label(lines[i]) in {"fund management", "about"}:
            i = _skip_block(lines, i)
            continue
        if i not in consumed:
            out.append(lines[i])
        i += 1
    return out


def line_chunks(lines: list[str], size: int, overlap: int,
                min_chars: int) -> list[str]:
    """Group whole lines up to `size` characters, with line-boundary overlap.

    `clean_text()` removed blank lines, so splitting on ``\\n\\n`` would make
    the whole document one paragraph. Grouping lines is the honest unit here.
    """
    lines = [line for line in lines if line.strip()]
    chunks: list[str] = []
    i = 0
    while i < len(lines):
        buf: list[str] = []
        used = 0
        j = i
        while j < len(lines):
            extra = len(lines[j]) + (1 if buf else 0)
            if buf and used + extra > size:
                break
            buf.append(lines[j])
            used += extra
            j += 1

        chunk = "\n".join(buf)
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if j >= len(lines):
            break

        # Step back by whole lines until the overlap budget is reached.
        back = 0
        k = j
        while k > i and back < overlap:
            k -= 1
            back += len(lines[k]) + 1
        i = max(i + 1, k)
    return chunks


def field_pieces(doc: dict, lines: list[str], cfg) -> tuple[list[tuple[str, str, int]], list[Fact]]:
    """Return (field chunks + residual general chunks, selected facts)."""
    pairs = parse_pairs(lines)
    facts = best_facts(pairs + header_facts(lines))

    # Remove every recognised pair, not just the winners: the unselected
    # `Total AUM` would otherwise sit in a general chunk next to the
    # scheme-level `Fund size (AUM)` and invite the wrong answer.
    consumed: set[int] = set()
    for fact in pairs:
        consumed.add(fact.line)
        consumed.add(fact.line + 1)

    pieces: list[tuple[str, str, int]] = []
    for fact in facts:
        if fact.field == "definition":
            text = f"{fact.label}: {fact.value}"
        else:
            prefix = f"{doc['scheme_name']} — " if doc.get("scheme_name") else ""
            text = f"{prefix}{fact.label}: {fact.value}"
        pieces.append((fact.field, text, fact.line))

    residual = residual_lines(lines, consumed)
    for offset, text in enumerate(line_chunks(
            residual, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap,
            cfg.chunking.min_chunk_chars)):
        pieces.append(("general", text, len(lines) + offset))

    pieces.sort(key=lambda p: p[2])
    return pieces, facts


def recursive_pieces(lines: list[str], cfg) -> list[tuple[str, str, int]]:
    return [("general", text, i) for i, text in enumerate(line_chunks(
        lines, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap,
        cfg.chunking.min_chunk_chars))]


# --------------------------------------------------------------------------- #
# stage entrypoint
# --------------------------------------------------------------------------- #

def run(cfg, force: bool = False, rebuild: bool = False) -> dict:
    docs, load_report = load_documents(cfg)          # explicit Data Loading hand-off
    write_json("artifacts/runs/load_documents.json", load_report)

    chunks: list[dict] = []
    details: list[dict] = []
    warnings: list[str] = list(load_report.get("warnings", []))

    for doc in docs:
        lines = [line.strip() for line in doc["text"].splitlines()
                 if line.strip()]
        strategy, rationale = choose(doc, cfg)

        if strategy == "field_aware":
            pieces, facts = field_pieces(doc, lines, cfg)
        else:
            pieces, facts = recursive_pieces(lines, cfg), []

        doc_chunks: list[dict] = []
        for position, (field, text, _line) in enumerate(pieces):
            if field == "general" and len(text) < cfg.chunking.min_chunk_chars:
                continue
            if field not in VOCABULARY:
                raise ValueError(
                    f"{doc['doc_id']}: field {field!r} is outside the vocabulary")
            if len(text.splitlines()) == 1 and text.rstrip().endswith(":"):
                warnings.append(f"{doc['doc_id']}: bare label chunk ({field})")
            doc_chunks.append(Chunk(
                chunk_id=f"{doc['doc_id']}:f_{field}:{position}",
                doc_id=doc["doc_id"],
                url=doc["url"],
                title=doc["title"],
                scheme_name=doc["scheme_name"],
                field=field,
                source_type=doc["source_type"],
                strategy=strategy,
                text=text,
                position=position,
                fetched_at=doc["fetched_at"],
            ).to_json())

        field_count = sum(1 for c in doc_chunks if c["field"] != "general")
        if strategy == "field_aware" and field_count == 0:
            raise ValueError(
                f"{doc['doc_id']}: field_aware produced no facts - the label map "
                "no longer matches this page")
        if strategy == "field_aware" and field_count < 5:
            warnings.append(
                f"{doc['doc_id']}: only {field_count} field chunks (expected 5+)")

        for chunk in doc_chunks:
            if (chunk["field"] == "exit_load"
                    and looks_like_definition(chunk["text"].split(": ", 1)[-1])):
                warnings.append(
                    f"{doc['doc_id']}: exit_load looks like a glossary definition")

        chunks.extend(doc_chunks)
        details.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "scheme_name": doc["scheme_name"],
            "source_type": doc["source_type"],
            "strategy": strategy,
            "rationale": rationale,
            "chars": len(doc["text"]),
            "field_chunks": field_count,
            "general_chunks": len(doc_chunks) - field_count,
            "chunks": len(doc_chunks),
            "fields": sorted({c["field"] for c in doc_chunks}),
            "facts": [{"field": f.field, "label": f.label, "value": f.value}
                      for f in facts],
        })

    processed = Path(cfg.sources.processed_dir)
    write_jsonl(processed / "chunks.jsonl", chunks)
    write_preview(processed / "chunks_preview.md", chunks)

    report = {
        "stage": "chunk",
        "documents": len(docs),
        "chunks": len(chunks),
        "by_field": dict(Counter(c["field"] for c in chunks).most_common()),
        "by_strategy": dict(Counter(c["strategy"] for c in chunks)),
        "documents_detail": details,
        "warnings": warnings,
    }
    write_json("artifacts/runs/chunk.json", report)
    return report


def write_preview(path: Path, chunks: list[dict]) -> None:
    """One bullet per chunk so a human can eyeball the whole corpus."""
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    lines = ["# Chunk preview", "",
             f"{len(chunks)} chunks from {len(by_doc)} documents.", ""]
    for doc_id, doc_chunks in by_doc.items():
        first = doc_chunks[0]
        lines += [f"## {doc_id} - {first['title']}", "",
                  f"strategy: `{first['strategy']}` | "
                  f"chunks: {len(doc_chunks)}", ""]
        for chunk in doc_chunks:
            text = " ".join(chunk["text"].split())
            if len(text) > 180:
                text = text[:177] + "..."
            lines.append(
                f"- `{chunk['chunk_id']}` **{chunk['field']}** "
                f"({len(chunk['text'])} chars) - {text}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
