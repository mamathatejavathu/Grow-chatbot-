"""Scrape + extract stage (implementation.md section 2).

Scope is deliberately narrow: fetch exactly the URLs listed in `sources.csv`,
one request each, and store the cleaned text. This module never follows links
found inside a page - no internal pages, no related-scheme pages, no sitemaps.

Artifacts written:
    data/raw/<doc_id>.html     verbatim response, for debugging extraction
    data/raw/<doc_id>.txt      cleaned text
    data/processed/documents.jsonl
    artifacts/runs/load.json
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import re
from itertools import zip_longest
from pathlib import Path

from bs4 import BeautifulSoup

from pipeline.lib.http import FetchError, RobotsDisallowed, fetch, robots_verdict
from pipeline.lib.io import read_jsonl, write_json, write_jsonl
from pipeline.lib.schemas import Document

MIN_TEXT_CHARS = 300
MAX_TEXT_CHARS = 200_000
REQUIRED_COLUMNS = ("id", "url", "title", "source_type", "scheme_name")

# Groww wraps the useful part of every page in `div.layout-main`. Readability
# picks the "Understand terms" glossary instead, which is why it is not used.
CONTENT_SELECTORS = ("div.layout-main", "main", "[role=main]")
DROP_TAGS = ("script", "style", "noscript", "svg", "iframe")
CHROME_TAGS = ("nav", "footer", "aside")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def table_to_lines(table) -> list[str]:
    """Flatten a table to `header: value` lines, one line per row.

    Groww's fund list has a header row (`Fund Name`, `Category`, `Expense
    Ratio`, ...), so repeating the header on every row keeps the value attached
    to its label once the table is gone. Two-column label/value tables without
    a header row are emitted as a single `label: value` pair.
    """
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return []

    header: list[str] = []
    if table.find("th") is not None and len(rows) > 1 and len(rows[0]) >= 2:
        header, rows = rows[0], rows[1:]

    lines: list[str] = []
    if header:
        for row in rows:
            pairs = [f"{h}: {v}" for h, v in zip_longest(header, row) if h and v]
            if pairs:
                lines.append(", ".join(pairs))
    else:
        for row in rows:
            if len(row) >= 2:
                lines.append(f"{row[0]}: {' '.join(c for c in row[1:] if c)}")
    return [line for line in lines if line]


def select_content(soup) -> tuple[object, str]:
    """Return (node, selector_name). Falls back to body minus site chrome."""
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node is not None and len(node.get_text(strip=True)) >= MIN_TEXT_CHARS:
            return node, selector
    body = soup.body or soup
    for tag in body(CHROME_TAGS):
        tag.decompose()
    return body, "body"


def extract(html: str) -> tuple[str, list[str]]:
    """Main content only, tables flattened to `key: value` lines.

    Anchors are dropped by `get_text()`, so link text survives but hrefs are
    never collected - nothing here can start a crawl.
    """
    warnings: list[str] = []
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(DROP_TAGS):
        tag.decompose()

    node, selector = select_content(soup)
    if selector != CONTENT_SELECTORS[0]:
        warnings.append(f"content_fallback:{selector}")

    for table in node.find_all("table"):
        lines = table_to_lines(table)
        table.replace_with("\n" + "\n".join(lines) + "\n" if lines else "\n")

    text = clean_text(node.get_text("\n"))[:MAX_TEXT_CHARS]
    if len(text) < MIN_TEXT_CHARS:
        warnings.append(f"short_extraction:{len(text)}")
    return text, warnings


def read_sources(cfg) -> list[dict]:
    """The only URL list in the system. No discovery happens anywhere else."""
    path = Path(cfg.sources.file)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - sources.csv is the URL list")

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return rows


# --------------------------------------------------------------------------- #
# stage entrypoint
# --------------------------------------------------------------------------- #

def run(cfg, force: bool = False, rebuild: bool = False) -> dict:
    sources = read_sources(cfg)
    processed_path = Path(cfg.sources.processed_dir) / "documents.jsonl"
    raw_dir = Path(cfg.sources.raw_dir)
    prior = {d["doc_id"]: d for d in read_jsonl(processed_path)}

    documents: list[dict] = []
    report: dict = {
        "stage": "load",
        "sources": len(sources),
        "written": 0,
        "skipped": 0,
        "failed": 0,
        "warnings": [],
        "documents": [],
        "crawl": "disabled (sources.csv only)",
    }

    for src in sources:
        doc_id = (src.get("id") or "").strip()
        url = (src.get("url") or "").strip()
        if not doc_id or not url:
            report["failed"] += 1
            report["warnings"].append(f"row skipped, missing id/url: {src}")
            continue

        try:
            allowed, note = robots_verdict(url, cfg.sources.user_agent)
            if note:
                report["warnings"].append(f"{doc_id}: {note}")
            if not allowed:
                raise RobotsDisallowed(f"robots.txt disallows {url}")

            response = fetch(url, cfg.sources.user_agent,
                             cfg.sources.request_delay_seconds)
            text, warnings = extract(response.text)
            digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

            if (not force and doc_id in prior
                    and prior[doc_id].get("content_hash") == digest):
                documents.append(prior[doc_id])
                report["skipped"] += 1
                report["documents"].append({
                    "doc_id": doc_id,
                    "url": url,
                    "chars": len(prior[doc_id].get("text", "")),
                    "warnings": [],
                    "status": "skipped",
                })
                continue

            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{doc_id}.html").write_text(response.text, encoding="utf-8")
            (raw_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

            document = Document(
                doc_id=doc_id,
                url=url,
                title=(src.get("title") or "").strip(),
                source_type=(src.get("source_type") or "").strip(),
                scheme_name=(src.get("scheme_name") or "").strip(),
                fetched_at=utc_now(),
                content_hash=digest,
                text=text,
            ).to_json()

            documents.append(document)
            report["written"] += 1
            report["documents"].append({
                "doc_id": doc_id,
                "url": url,
                "chars": len(text),
                "warnings": warnings,
                "status": "written",
            })
            report["warnings"].extend(f"{doc_id}: {w}" for w in warnings)

        except (FetchError, RobotsDisallowed) as exc:
            report["failed"] += 1
            report["warnings"].append(f"{doc_id}: {exc}")
            report["documents"].append({
                "doc_id": doc_id, "url": url, "chars": 0,
                "warnings": [str(exc)], "status": "failed",
            })
            if doc_id in prior:            # keep the last good copy
                documents.append(prior[doc_id])
        except Exception as exc:           # never let one page kill the stage
            report["failed"] += 1
            report["warnings"].append(f"{doc_id}: {type(exc).__name__}: {exc}")
            report["documents"].append({
                "doc_id": doc_id, "url": url, "chars": 0,
                "warnings": [f"{type(exc).__name__}: {exc}"], "status": "failed",
            })
            if doc_id in prior:
                documents.append(prior[doc_id])

    write_jsonl(processed_path, documents)
    report["read"] = len(documents)
    report["fetched"] = report["written"] + report["skipped"]
    write_json("artifacts/runs/load.json", report)
    return report
