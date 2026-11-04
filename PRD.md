# PRD — Groww Mutual Fund FAQ Assistant (RAG Prototype)

**Document owner:** Product / Engineering
**Version:** 1.0
**Status:** Draft for build
**Last updated:** 2026-11-01
**Source input:** `Docs/Problemstatement.txtt`

---

## 1. Overview

A small, facts-only FAQ assistant for HDFC Mutual Fund schemes, built on top of official public Groww scheme pages. A user asks a factual question (expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, statement download) and receives a **short answer (≤3 sentences) with exactly one official source link**.

The assistant **refuses** opinion, advice, and portfolio questions, and **never** accepts or stores PII.

This is a **RAG demo**, not a production product. The emphasis is on making every stage of the pipeline **explicitly inspectable** — each stage writes its own artifact (raw documents, chunks, embeddings, vector store, retrieved context, prompt, final answer) so the pipeline can be shown end to end.

---

## 2. Problem statement

Retail investors comparing mutual fund schemes, and support/content teams answering repetitive mutual fund questions, repeatedly look up the same small set of facts scattered across scheme pages. That lookup is slow, inconsistent, and easy to get wrong — and generic chatbots blur the line between **facts** and **advice**.

We need a narrow, trustworthy assistant that:

1. Answers only the exact fact asked, using only official public pages.
2. Always shows one clear citation.
3. Politely declines anything that looks like investment advice.
4. Refuses to handle personal or sensitive data.

---

## 3. Goals and non-goals

### 3.1 Goals

| # | Goal | Measure |
|---|------|---------|
| G1 | Answer the 6 core factual question types correctly | ≥90% correct on the 5–10 query sample set |
| G2 | Cite exactly one official URL in every answer | 100% of answers |
| G3 | Refuse advice/opinion questions with a facts-only message + one official corpus link | 100% of refusal test cases |
| G4 | Never ingest, echo, or store PII | 0 PII in logs and artifacts |
| G5 | Make all RAG stages inspectable | Each stage persists its own artifact |
| G6 | Keep every answer ≤3 sentences | 100% of answers |

### 3.2 Non-goals

- No performance comparison, return computation, or CAGR/NAV analysis.
- No portfolio construction, buy/sell/hold recommendations, or risk profiling.
- No account-specific data (holdings, capital gains numbers, statements) — only *how-to* instructions.
- No coverage outside HDFC Mutual Fund + the four named schemes.
- No mobile app, no authentication, no multi-tenant deployment.

---

## 4. Users and use cases

| User | Need | Success looks like |
|------|------|--------------------|
| Retail investor comparing schemes | Quick, sourced facts | "Expense ratio is 0.75%. Source: …" |
| Support / content team | Repeatable, citable answers | Copy answer + link into a ticket |
| Reviewer / evaluator (demo) | See the RAG pipeline work | Can open each stage's artifact |

### Core question types (in scope)

1. Expense ratio of a scheme
2. Exit load
3. Minimum SIP amount / minimum investment
4. ELSS lock-in period
5. Riskometer level
6. Benchmark index
7. AMC-level facts (fund house overview, total schemes, AUM as stated on page)

### Out of scope questions (must refuse)

- "Should I buy HDFC Flexi Cap?" / "Is this a good fund?"
- "Which fund will give me the best returns?"
- "Compare these two funds for me"
- "What returns did this fund give last year?" → redirect to official factsheet link, do not compute
- "How do I download my capital-gains statement?" → not covered by the 5 source pages; refuse and link the AMC page
- Anything containing PAN, Aadhaar, account number, OTP, email, phone

---

## 5. Scope of data

### 5.1 Primary sources (must scrape)

| # | Page | URL |
|---|------|-----|
| 1 | AMC overview — HDFC Mutual Fund | https://groww.in/mutual-funds/amc/hdfc-mutual-funds |
| 2 | Large Cap — HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 3 | Flexi Cap — HDFC Flexi Cap Direct Plan Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 4 | ELSS — HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 5 | Mid Cap — HDFC Mid Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |

### 5.2 Scope decision: the 5 URLs only

The corpus is **exactly the 5 pages in §5.1**. There is no supplementary crawl: the loader does not follow links found inside a page, does not read sitemaps, and does not fetch Groww help/education pages, HDFC MF factsheets, SEBI or AMFI pages.

Consequence to accept deliberately: definitional and how-to questions (what is an exit load, how do I download a statement) are answered only from what these 5 pages already contain — which includes Groww's own glossary block on each scheme page. Anything not covered is refused as "not in my sources" rather than answered from an extra page.

> **Deliverable:** `sources.csv` with the 5 URLs, each with: URL, page title, source type (AMC / scheme), and which question types it supports.

### 5.3 Source rules

- Public pages only. No app back-end screenshots, no login-gated content.
- Official sources only: Groww scheme/AMC pages (HDFC MF, SEBI, AMFI only if a row is added by hand).
- **No crawling.** One HTTP request per `sources.csv` row. A page's outbound links are never enqueued.
- Every stored chunk must carry its source URL and fetch timestamp so citations are always reconstructable.
- Respect `robots.txt`, rate-limit politely, cache locally, and re-fetch only on a manual refresh command.

---

## 6. Functional requirements

### 6.1 Ingestion — Phase 1: Data Loading

| ID | Requirement |
|------|-------------|
| FR-1.1 | Fetch each URL in the source list and store the raw response (HTML + extracted text) under `data/raw/` |
| FR-1.2 | Extract main content only (strip nav, footer, ads, related-links) — prefer the page's main content container, then `<main>` / `role=main` / body-minus-chrome |
| FR-1.3 | Normalize text: collapse whitespace, fix encoding, keep table rows as `key: value` pairs |
| FR-1.4 | Preserve metadata per document: `url`, `title`, `source_type`, `scheme_name`, `fetched_at`, `content_hash` |
| FR-1.5 | Skip re-fetch when `content_hash` is unchanged; log added/updated/skipped counts |
| FR-1.6 | Write a `data/processed/documents.jsonl` — one record per page |

### 6.2 Ingestion — Phase 2: Chunking

Chunking strategy is **data-aware** and configurable; a small "chunking agent" may choose the strategy per document type, but the chosen strategy and rationale must be logged.

| ID | Requirement |
|------|-------------|
| FR-2.1 | Detect document type: `scheme_page` (key-value fact table heavy), `amc_page`, `education_page` (prose) |
| FR-2.2 | For `scheme_page`: **field-aware chunking** — one chunk per factual field (e.g. "Expense ratio", "Exit load", "Min SIP") so retrieval maps 1 question → 1 fact |
| FR-2.3 | For `education_page` / `amc_page`: **recursive character chunking** with configurable `chunk_size` (default 600 tokens) and `chunk_overlap` (default 80 tokens), split on headings → paragraphs → sentences |
| FR-2.4 | Never split a `key: value` fact across chunks; always attach the scheme name and field label to the chunk text |
| FR-2.5 | Store `chunk_id`, `doc_id`, `url`, `scheme_name`, `field`, `text`, `token_count`, `chunk_strategy`, `position` |
| FR-2.6 | Write `data/processed/chunks.jsonl` and a human-readable `data/processed/chunks_preview.md` |
| FR-2.7 | Log a chunking report: strategy used per doc, chunk counts, avg/median token length |

### 6.3 Ingestion — Phase 3: Embedding

| ID | Requirement |
|------|-------------|
| FR-3.1 | Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (configurable) |
| FR-3.2 | Batch-embed all chunks; persist vectors to `data/embeddings/embeddings.npy` plus an id map |
| FR-3.3 | Cache embeddings keyed by `content_hash` + model name to avoid recompute |
| FR-3.4 | Log model name, vector dimension, chunk count, and wall-clock time |
| FR-3.5 | Same embedding model must be used at query time; refuse to start if the configured model differs from the one used to build the index |

### 6.4 Ingestion — Phase 4: Vector DB (ChromaDB)

| ID | Requirement |
|------|-------------|
| FR-4.1 | Vector store: **ChromaDB** (persistent local client), configurable |
| FR-4.2 | One collection (default `groww_mf_faq`), configurable name |
| FR-4.3 | Upsert with metadata: `url`, `title`, `scheme_name`, `field`, `source_type`, `fetched_at` |
| FR-4.4 | Support a full rebuild (`--rebuild`) and an incremental upsert |
| FR-4.5 | Log collection name, count, and persistence path |

### 6.5 Retrieval — Phase 5

| ID | Requirement |
|------|-------------|
| FR-5.1 | Embed the user query with the same embedding model as ingestion |
| FR-5.2 | Query ChromaDB for top-`k` (default 5, configurable) chunks |
| FR-5.3 | Apply metadata filtering when the query names a scheme (e.g. `scheme_name = "HDFC ELSS Tax Saver"`) |
| FR-5.4 | Re-rank/dedupe so chunks from the same field/scheme do not crowd out others |
| FR-5.5 | Persist the retrieval trace: query, filters, top-k chunks with scores and URLs → `artifacts/retrieval/<run_id>.json` |
| FR-5.6 | If the best score is below `min_relevance_score` (configurable), return the "not enough information" response with the AMC overview link — never guess |

### 6.6 Generation — Phase 6

| ID | Requirement |
|------|-------------|
| FR-6.1 | LLM provider configurable: **Groq** (default), Gemini, OpenAI, Anthropic, Mistral |
| FR-6.2 | Model ID read from `.env` (`LLM_MODEL`), never hard-coded; default Groq model `qwen/qwen3.8-27b` per the brief, changeable without code edits |
| FR-6.3 | Prompt is grounded strictly in retrieved context; the model must not use outside knowledge |
| FR-6.4 | Answer format: **≤3 sentences**, plain language, no advice, ends with a citation line containing exactly **one** URL taken from the retrieved metadata |
| FR-6.5 | Append `Last updated from sources: <YYYY-MM-DD>` using the chunk's `fetched_at` |
| FR-6.6 | If context is insufficient → say so and cite the AMC overview page |
| FR-6.7 | Persist the full prompt + raw model response → `artifacts/generation/<run_id>.json` |
| FR-6.8 | Support a `--no-llm` extractive fallback (return the matching chunk text verbatim) so the demo works without an API key |

### 6.7 Safety — PII guard and refusal policy

| ID | Requirement |
|------|-------------|
| FR-7.1 | Pre-LLM PII detector: PAN (5 letters + 4 digits + 1 letter), Aadhaar (12 digits), OTP (4–6 digits near "OTP"), email, phone (10-digit Indian), account numbers (9–18 digits) |
| FR-7.2 | On PII detection: do **not** log or echo the value; respond "Please don't share personal identifiers…" and offer the AMC overview page |
| FR-7.3 | Advice/opinion classifier (rules + optional LLM check): buy/sell/hold, "should I", "best fund", "worth it", "compare", "will it give returns" |
| FR-7.4 | On refusal: polite, facts-only message + one relevant official link from the corpus (scheme page for a scheme question, AMC overview otherwise) — no link outside `sources.csv` |
| FR-7.5 | Refusals are never persisted with raw user text if the text contained PII — store a redacted form |
| FR-7.6 | Answers never contain performance figures or computed returns; return-link to the official factsheet instead |

### 6.8 UI

| ID | Requirement |
|------|-------------|
| FR-8.1 | Minimal chat UI (Streamlit or Gradio): welcome line, 3 example question chips, input box, answer area |
| FR-8.2 | Persistent disclaimer: **"Facts-only. No investment advice."** |
| FR-8.3 | Every answer shows the citation as a single clickable link |
| FR-8.4 | "Show pipeline" expander per answer: retrieved chunks, scores, prompt, model, latency |
| FR-8.5 | Scheme selector (optional convenience) that pre-fills the scheme name into the query |

---

## 7. Configuration

All tunables live in one `config.yaml` (overridable by `.env` for secrets). No magic values in code.

```yaml
# config.yaml
app:
  name: "Groww MF FAQ Assistant"
  disclaimer: "Facts-only. No investment advice."
  max_answer_sentences: 3
  example_questions:
    - "What is the expense ratio of HDFC Flexi Cap Direct Plan?"
    - "What is the lock-in period for HDFC ELSS Tax Saver?"
    - "What is the riskometer level of HDFC Large Cap Fund?"

sources:
  file: "sources.csv"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  user_agent: "groww-mf-faq-demo/1.0"
  request_delay_seconds: 2

chunking:
  strategy: "auto"          # auto | field_aware | recursive
  chunk_size: 600
  chunk_overlap: 80
  min_chunk_tokens: 20
  log_strategy_rationale: true

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  normalize: true
  cache_dir: "data/embeddings"

vector_db:
  provider: "chromadb"
  persist_dir: "data/chroma"
  collection: "groww_mf_faq"
  distance: "cosine"

retrieval:
  top_k: 5
  min_relevance_score: 0.35
  metadata_filter_by_scheme: true
  dedupe_by_field: true

llm:
  provider: "groq"          # groq | gemini | openai | anthropic | mistral
  model_env: "LLM_MODEL"    # model id always read from .env
  temperature: 0.0
  max_tokens: 300
  timeout_seconds: 30

safety:
  enable_pii_guard: true
  enable_advice_guard: true
  refuse_message: "I can only share factual information from official sources — I can't give investment advice."
  pii_message: "Please don't share personal identifiers. I can only answer general factual questions."
```

`.env` (never committed):

```
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
GROQ_API_KEY=...
# GEMINI_API_KEY=...
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
# MISTRAL_API_KEY=...
```

---

## 8. Architecture and artifacts

```
sources.csv
    │
    ▼
[1] Load      → data/raw/*.html, data/processed/documents.jsonl
    │
    ▼
[2] Chunk     → data/processed/chunks.jsonl, chunks_preview.md, chunking_report.json
    │
    ▼
[3] Embed     → data/embeddings/embeddings.npy, id_map.json
    │
    ▼
[4] Index     → data/chroma/ (collection: groww_mf_faq)
    │
    ▼
[5] Retrieve  → artifacts/retrieval/<run_id>.json
    │
    ▼
[6] Generate  → artifacts/generation/<run_id>.json  →  UI answer + citation
```

**Inspectability requirement:** running `python -m pipeline.run --stage all` must leave every artifact above on disk, and `python -m pipeline.inspect <run_id>` must print the full trace for one query (chunks, scores, prompt, answer, latency, model).

---

## 9. Prompting and output contract

### 9.1 System prompt (draft)

```
You are a facts-only assistant for HDFC Mutual Fund scheme pages.
Use ONLY the provided context. If the context does not contain the answer, say you
don't have that information and cite the AMC overview page.

Rules:
- Answer in at most 3 sentences.
- Do not give advice, opinions, or recommendations.
- Do not compute or compare returns.
- Do not ask for or repeat personal data.
- End with exactly one line: "Source: <URL>" using a URL from the context.
- End with: "Last updated from sources: <date>".
```

### 9.2 Output shape

```
<answer, ≤3 sentences>
Source: <single official URL>
Last updated from sources: 2026-11-01
```

### 9.3 Refusal shape

```
I can only share factual information from official sources — I can't give investment advice.
You can learn the basics here: <one official URL from sources.csv>
```

---

## 10. Phased delivery plan

| Phase | Deliverable | Exit criteria |
|-------|-------------|---------------|
| **P0 — Setup** | Repo skeleton, `config.yaml`, `.env.example`, `sources.csv` (5 URLs) | Config loads; source list validated |
| **P1 — Data loading** | Scraper + extractor + `documents.jsonl` | All source URLs fetched; metadata complete; re-run is idempotent |
| **P2 — Chunking** | Strategy selector + `chunks.jsonl` + report | Every scheme field retrievable as its own chunk; no split key-values |
| **P3 — Embedding** | MiniLM embeddings + cache | Dimension/count logged; rebuild is deterministic |
| **P4 — Vector DB** | ChromaDB collection populated | Query returns expected chunk for all 6 question types |
| **P5 — Retrieval** | Top-k + filters + trace artifact | Retrieval trace shows correct source URL per question type |
| **P6 — Generation** | Grounded answer + citation + fallback | ≤3 sentences, one URL, correct `Last updated` date |
| **P7 — Safety & UI** | PII guard, advice refusal, Streamlit UI | 100% refusal on advice set; 0 PII in logs |
| **P8 — Deliverables** | README, sample Q&A, demo video/link | All submission items present |

---

## 11. Acceptance criteria

- [ ] All 5 `sources.csv` pages are scraped and stored with URL + timestamp, and nothing outside that list is fetched.
- [ ] Each RAG stage writes a separate, human-readable artifact.
- [ ] Expense ratio, exit load, min SIP, ELSS lock-in, riskometer, benchmark are answered correctly for all four schemes.
- [ ] "How to download capital gains statement" is refused as out-of-scope with the AMC page link, not answered from an external page.
- [ ] Every answer has exactly one official citation and `Last updated from sources:`.
- [ ] Every answer is ≤3 sentences.
- [ ] 10/10 advice/opinion test questions are refused politely with one official corpus link.
- [ ] PII test inputs are never echoed and never written to logs or artifacts.
- [ ] No answer contains a computed return or a performance comparison.
- [ ] Embedding model and vector DB are swapped by editing `config.yaml` only.
- [ ] LLM provider/model is swapped by editing `.env` only.
- [ ] App runs locally with `pip install -r requirements.txt` + `python -m pipeline.run --stage all` + `streamlit run app.py`.

---

## 12. Deliverables checklist

1. **Working prototype** — local app link or ≤3-min demo video.
2. **Source list** — `sources.csv` with the 5 official URLs.
3. **README** — setup steps, scope (AMC + schemes), known limits.
4. **Sample Q&A** — 5–10 queries with answers + links (`samples/sample_qa.md`).
5. **Disclaimer snippet** — "Facts-only. No investment advice." used in the UI.
6. **Pipeline artifacts** — raw docs, chunks, embeddings, Chroma store, retrieval + generation traces.

---

## 13. Known limits and risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Groww page markup changes | Extraction breaks | Store raw HTML; extractor tests on fixtures; alert on empty extraction |
| Facts change (expense ratio, exit load) | Stale answers | `fetched_at` shown in every answer; manual refresh command |
| Ambiguous query ("expense ratio?") | Wrong scheme retrieved | Ask a clarifying question naming the four in-scope schemes |
| MiniLM weak on numeric/table facts | Poor recall | Field-aware chunks + metadata filter + rerank |
| LLM hallucinates a fact or URL | Trust loss | Grounded prompt, URL must come from retrieved metadata, post-check that the cited URL exists in context |
| LLM adds advice despite prompt | Policy breach | Pre- and post-guard; reject and regenerate once, else fall back to extractive answer |
| Groq rate limits / key missing | Demo fails | `--no-llm` extractive fallback; provider switch via `.env` |
| PII leaking into logs | Compliance breach | Redact before logging; never persist raw input on PII hit |
| Scraping etiquette | Blocked IP | 2s delay, cached local copies, robots.txt respected |

---

## 14. Success metrics (demo scale)

- Answer accuracy on sample set: **≥90%**
- Citation validity (URL appears in retrieved context): **100%**
- Advice refusal rate: **100%**
- PII leakage incidents: **0**
- Median end-to-end latency: **<5s** (LLM) / **<1s** (extractive fallback)

---

## 15. Open questions

1. Confirm the exact Groq model ID — the brief lists `qwen/qwen3.8-27b`; verify it against Groq's current model list and pin the valid ID in `.env`.
2. Should scheme selection be mandatory (forces a scheme) or inferred from the query?
3. ~~Do we need a "download statement" walkthrough?~~ Resolved: out of scope — no help pages are scraped, so the question is refused with the AMC link.
4. Is a public deployment required, or is the demo video acceptable (per the brief)?
5. Refresh cadence for scraped pages — on demand only, or scheduled?
