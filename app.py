"""Streamlit UI (implementation.md §6.4, architecture.md §5).

One page, one question at a time:

* a persistent facts-only disclaimer,
* three example chips from `config.yaml`,
* an optional scheme selector that pre-fills the question,
* the answer body plus exactly one clickable official source and its date,
* a "Show pipeline" expander with the guard decision, retrieved chunks and
  scores, validation checks, prompt, provider/model, and latency.

The config and the retriever are cached with `@st.cache_resource`. Re-creating
the retriever on every keystroke would reload MiniLM (~20 s) and reopen Chroma.

The UI never sees the raw question after a PII refusal: the input box is cleared
on the next rerun, and `service.ask()` only persists the redacted text.
"""

from __future__ import annotations

import streamlit as st

from pipeline.lib.config import load_config
from query import service
from query.retriever import SCHEMES

SCHEME_ANY = "Any scheme / AMC overview"
SCHEME_OPTIONS = [SCHEME_ANY, *dict.fromkeys(SCHEMES.values())]

st.set_page_config(page_title="Groww MF FAQ", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner=False)
def _config():
    """Load `config.yaml` + `.env` once per Streamlit process."""
    return load_config()


def _init_state() -> None:
    st.session_state.setdefault("q_input", "")
    st.session_state.setdefault("scheme_choice", SCHEME_ANY)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("pending", None)


def _prefill_scheme() -> None:
    """Selector callback: put the chosen scheme at the front of the question."""
    scheme = st.session_state.get("scheme_choice", SCHEME_ANY)
    if scheme == SCHEME_ANY:
        return
    current = (st.session_state.get("q_input") or "").strip()
    if scheme.lower() not in current.lower():
        st.session_state.q_input = f"{scheme} {current}".strip()


def _split_answer(answer: str) -> tuple[str, str, str]:
    """Separate the body from the two contract lines the validator appends."""
    body_lines: list[str] = []
    source = ""
    updated = ""
    for line in (answer or "").splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("source:"):
            source = stripped.split(":", 1)[1].strip()
        elif lowered.startswith("last updated from sources:"):
            updated = stripped.split(":", 1)[1].strip()
        elif stripped:
            body_lines.append(stripped)
    return " ".join(body_lines), source, updated


def _friendly_error(exc: Exception) -> str:
    """Turn index/manifest startup failures into a build instruction."""
    message = str(exc)
    if any(marker in message for marker in
           ("manifest.json", "collection", "--stage", "index")):
        return ("The index is not ready. Run "
                "`python -m pipeline.run --stage all` and reload this page.\n\n"
                f"Details: {message}")
    return f"Something went wrong while answering: {message}"


def _run(question: str, cfg) -> None:
    question = (question or "").strip()
    if not question:
        st.session_state.result = None
        st.session_state.error = service.EMPTY_MESSAGE
        return

    try:
        with st.spinner("Checking official sources…"):
            result = service.ask(question, cfg)
    except Exception as exc:                    # index missing, Chroma lock, ...
        st.session_state.result = None
        st.session_state.error = _friendly_error(exc)
        return

    st.session_state.result = result
    st.session_state.error = None

    # Do not echo the identifier back in the input box. Clearing is deferred to
    # the next run because the widget has already been instantiated in this one.
    if result.get("mode") == "pii_refusal":
        st.session_state["_clear_input"] = True
        st.rerun()


def _render_answer(result: dict) -> None:
    body, source, updated = _split_answer(result.get("answer", ""))
    source = source or result.get("citation", "")
    updated = updated or result.get("last_updated", "")

    st.markdown(body or "I don't have that information in my sources.")
    if source:
        st.markdown(f"**Source:** [{source}]({source})")
    if updated:
        st.caption(f"Last updated from sources: {updated}")


def _render_pipeline(result: dict) -> None:
    checks = result.get("checks") or {}
    hits = result.get("hits") or []
    mode = result.get("mode", "")
    extractive = mode.startswith("extractive")
    provider = result.get("provider") or ("none" if extractive else "—")
    model = result.get("model") or ("extractive" if extractive else "—")

    with st.expander("Show pipeline"):
        st.markdown(
            f"**Run:** `{result.get('run_id', '')}`  \n"
            f"**Mode:** `{mode}` · "
            f"**Provider:** `{provider}` · "
            f"**Model:** `{model}` · "
            f"**Latency:** {result.get('latency_ms', 0)} ms  \n"
            f"**Gate:** `{result.get('gate', 'skipped')}` · "
            f"**Citation valid:** `{checks.get('citation_valid')}` · "
            f"**Sentences:** `{checks.get('sentences')}`"
        )

        st.markdown("**Guard** (redacted text only)")
        st.json(result.get("guard") or {})

        st.markdown("**Validation checks**")
        st.json(checks)

        st.markdown("**Retrieved chunks**")
        if hits:
            st.dataframe(
                [
                    {
                        "rank": index,
                        "score": round(hit.get("score", 0.0), 3),
                        "field": hit.get("field", ""),
                        "scheme": (hit.get("metadata") or {}).get(
                            "scheme_name") or "AMC overview",
                        "chunk_id": hit.get("chunk_id", ""),
                        "url": hit.get("url", ""),
                        "text": (hit.get("text") or "")[:180],
                    }
                    for index, hit in enumerate(hits, start=1)
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No retrieval was performed for this response.")

        st.markdown("**Prompt sent to the model**")
        prompt = result.get("prompt") or ""
        if prompt:
            st.code(prompt, language="text")
        elif extractive:
            st.caption(
                "Extractive mode — no LLM prompt was sent; the top retrieved "
                "chunk was used verbatim.")
        else:
            st.caption(
                "No generation was run for this response — the guard or the "
                "relevance gate stopped the pipeline before the LLM.")
        st.caption(f"Full trace: artifacts/runs/{result.get('run_id', '')}.json")


def main() -> None:
    cfg = _config()
    _init_state()

    # Applied before the text-input widget exists in this run, which is the only
    # point where Streamlit allows a widget's session value to be reset.
    if st.session_state.pop("_clear_input", False):
        st.session_state.q_input = ""

    st.title(cfg.app.name)
    st.info(cfg.app.disclaimer)                 # persistent, top of page

    with st.sidebar:
        st.subheader("Optional filter")
        st.selectbox(
            "Scheme",
            SCHEME_OPTIONS,
            key="scheme_choice",
            on_change=_prefill_scheme,
            help="Pre-fills the question with the scheme name. "
                 "Filtering happens only when a scheme is named.",
        )

    columns = st.columns(3)
    for column, example in zip(columns, cfg.app.example_questions):
        if column.button(example, use_container_width=True):
            st.session_state.q_input = example
            st.session_state.pending = example

    with st.form("question_form", clear_on_submit=False):
        st.text_input(
            "Your question",
            key="q_input",
            placeholder="e.g. What is the exit load of HDFC Mid Cap Fund?",
        )
        submitted = st.form_submit_button("Ask", type="primary")
        if submitted:
            st.session_state.pending = st.session_state.q_input

    pending = st.session_state.pending
    if pending is not None:
        st.session_state.pending = None
        _run(pending, cfg)

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.result:
        _render_answer(st.session_state.result)
        _render_pipeline(st.session_state.result)


main()
