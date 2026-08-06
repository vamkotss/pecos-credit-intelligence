"""A walkthrough of the Pecos pipeline.

    streamlit run app/streamlit_app.py

Reads the committed artefacts in `docs/samples/` by default, so it runs
immediately after a clone with no corpus, no Tesseract and no API key. If a
local corpus exists it offers a live mode where retrieval and memo generation
run for real.

WHAT THIS IS FOR, AND WHAT IT IS NOT
------------------------------------
The interesting output of this project is not a summary line. "GATE PASSED" says
almost nothing; the derivation of a pro forma leverage figure, or an attack
visibly flipping a credit decision and being blocked by an accounting identity,
is what shows how the system works. Those are hard to appreciate from a terminal
and easy to appreciate side by side.

This is a **reading tool, not a product**. It has no authentication, no
persistence and no multi-user anything, because it exists so somebody can
understand the pipeline in five minutes rather than by installing it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "docs" / "samples"
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Pecos Credit Intelligence", layout="wide")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@st.cache_data
def sample(name: str):
    """Read a committed artefact. Missing files return None rather than raising,
    so a partial export degrades into a page saying so."""
    path = SAMPLES / name
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if name.endswith(".json") else text


def live_corpus_available() -> bool:
    chunks = ROOT / "data" / "interim" / "chunks"
    return chunks.is_dir() and any(chunks.glob("PCP-*.jsonl"))


@st.cache_resource
def build_sample_retriever():
    """Retriever over the committed chunk index.

    Lets the deployed app do real retrieval with no corpus, no Tesseract and no
    API key. It is the genuine M5 stack -- BM25, LSA dense, rank fusion, rerank
    and authority weighting -- over two deals' worth of real chunks, not a
    lookup table pretending to be search.
    """
    from pecos.retrieval import HybridRetriever

    path = SAMPLES / "chunks_index.jsonl"
    if not path.exists():
        return None
    chunks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retriever = HybridRetriever()
    retriever.build(chunks)
    return retriever


@st.cache_resource
def build_retriever():
    from pecos.chunking import load_chunks
    from pecos.retrieval import HybridRetriever

    chunks: list[dict] = []
    for path in sorted((ROOT / "data" / "interim" / "chunks").glob("PCP-*.jsonl")):
        chunks.extend(load_chunks(path))
    retriever = HybridRetriever()
    retriever.build(chunks)
    return retriever


def missing(name: str) -> None:
    st.info(
        f"`docs/samples/{name}` is not present. Run "
        "`python scripts/export_samples.py` after building a corpus."
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.title("Pecos Credit Intelligence")
    st.caption(
        "Document intelligence for middle-market commercial lending. Every figure "
        "that reaches a credit committee is traceable to a page, or to a "
        "calculation whose inputs are."
    )

    gate = sample("eval_gate.json")
    retrieval = sample("retrieval_report.json")
    redteam = sample("redteam_report.json")
    containment = sample("chunk_containment.json")

    columns = st.columns(4)
    columns[0].metric("Tests", "391")
    if containment:
        columns[1].metric("Chunk containment", f"{containment['rate']:.0%}")
    if retrieval:
        columns[2].metric("Retrieval recall@5", f"{retrieval['recall']['@5']:.0%}")
    if redteam:
        columns[3].metric(
            "Attacks succeeded", f"{redteam['succeeded']} / {redteam['attacks']}"
        )

    st.markdown(
        """
### The pipeline

```
generate → ingest → chunk → retrieve → answer → memo → guardrails → review
```

A synthetic corpus with page-level ground truth and seven planted defects, so
every claim below is measured rather than asserted.

### The constraint everything follows from

A figure in a credit memo must be **quoted** from a cited page, **computed** by a
calculator that recorded its inputs and their pages, or **derivable** from
figures that were — with the derivation shown. Anything else fails a gate.
        """
    )

    if gate:
        st.subheader("Eval gate")
        st.caption(
            "Twelve thresholds, no API key, under six seconds. Only free and "
            "deterministic metrics gate a build."
        )
        rows = []
        for threshold in gate["thresholds"]:
            value = gate["metrics"].get(threshold["name"])
            if value is None:
                continue
            passes = (
                value >= threshold["floor"]
                if threshold["higher_is_better"]
                else value <= threshold["floor"]
            )
            rows.append(
                {
                    "": "PASS" if passes else "FAIL",
                    "metric": threshold["name"],
                    "value": round(value, 3),
                    "floor": threshold["floor"],
                    "why": threshold["note"],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


def page_corpus() -> None:
    st.header("The corpus")
    st.caption(
        "Documents and their labels are generated together. The manifest is not "
        "extracted from the PDFs — the PDFs are rendered from the manifest's "
        "source data, which is what makes recall and grounding measurable at all."
    )

    data = sample("corpus_summary.json")
    if not data:
        return missing("corpus_summary.json")

    columns = st.columns(3)
    columns[0].metric("Deals", data["deals"])
    columns[1].metric("Gold facts", data["gold_facts"])
    columns[2].metric("Seed", data["seed"])

    st.subheader("Seven planted defects")
    st.caption(
        "A clean synthetic corpus flatters a pipeline. These are planted "
        "deliberately, each registered in the manifest with at least one gold "
        "question, and dealt round-robin so all seven appear even in a "
        "three-deal CI corpus."
    )
    explanations = {
        "restated_prior_year": "Two documents, one fiscal year, two EBITDAs",
        "units_in_thousands": "Off by exactly 1,000 — the silent one",
        "rotated_scanned_page": "A page that went through the scanner sideways",
        "table_only_fact": "A figure that appears in no sentence anywhere",
        "prompt_injection": "An APPROVE instruction inside a broker's PDF",
        "unanswerable_question": "Refusal instead of invention",
        "near_duplicate_draft": "A DRAFT beside the final, one figure changed",
    }
    st.dataframe(
        [
            {"defect": k, "deals": ", ".join(v), "what it breaks": explanations.get(k, "")}
            for k, v in sorted(data["defect_index"].items())
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Borrowers")
    st.caption(
        "Balance sheets balance to the dollar and cash flow ties to the movement "
        "in cash, asserted on every deal and every year."
    )
    st.dataframe(
        [
            {
                "deal": d["deal_id"],
                "borrower": d["borrower_name"],
                "revenue": f"${d['latest_revenue']:,}",
                "EBITDA": f"${d['latest_ebitda']:,}",
                "leverage": f"{d['leverage']:.2f}x",
                "DSCR": f"{d['dscr']:.2f}x",
                "defects": ", ".join(d["defects"]) or "—",
            }
            for d in data["deal_summaries"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def page_ingestion() -> None:
    st.header("Ingestion")
    st.caption(
        "Every page is routed on its own merits: a text layer goes to pdfplumber, "
        "no text layer is rendered and OCR'd. Per page, not per document — real "
        "packages mix clean exports with appended scanned signature sheets."
    )

    pages = sample("ingestion_pages.json")
    if not pages:
        return missing("ingestion_pages.json")

    st.info(
        "**The page is the unit of provenance.** A document is too coarse to "
        "cite — 'it's in the financial statements' is not an answer an analyst "
        "accepts. A chunk is too unstable, because chunk boundaries move whenever "
        "the chunking strategy changes."
    )

    labels = [
        f"{p['document']} p{p['page_number']} ({p['method']}"
        + (f", rotated {p['rotation_applied']}°" if p["rotation_applied"] else "")
        + ")"
        for p in pages
    ]
    choice = st.selectbox("Page", range(len(pages)), format_func=lambda i: labels[i])
    page = pages[choice]

    columns = st.columns(4)
    columns[0].metric("Method", page["method"])
    columns[1].metric("Rotation", f"{page['rotation_applied']}°")
    columns[2].metric("OCR confidence", page["mean_word_confidence"] or "—")
    columns[3].metric("Scale factor", f"{page['scale_factor']:,}")

    if page["rotation_applied"]:
        st.warning(
            "Read as it sits, this page returns `\") S89U9INDDO Spun} "
            "JUaIONJNSU]\"` — noise that would be embedded and retrieved as "
            "though it meant something. Tesseract's orientation detection is used "
            "as a *proposal* and verified by **line span**: upright a line runs "
            "across the page, sideways each line is one stacked word. Measured "
            "here, 1,344 pixels against 19."
        )
    if page["scale_factor"] != 1:
        st.warning(
            f"This page states figures in units of {page['scale_factor']:,}. "
            f"Evidence: *{page['scale_evidence']}*. Of every failure mode in the "
            "corpus this is the one most likely to reach a credit committee "
            "undetected — no confidence score flags it and the figure does not "
            "look wrong."
        )

    left, right = st.columns(2)
    left.subheader("Extracted text")
    left.code(page["text"][:900], language=None)
    right.subheader("Tables")
    if page["tables"]:
        for table in page["tables"]:
            right.caption(f"source: {table['source']}")
            right.dataframe(table["rows"], use_container_width=True, hide_index=True)
    else:
        right.caption("no tables on this page")


def page_chunking() -> None:
    st.header("Chunking")
    data = sample("chunk_containment.json")
    if not data:
        return missing("chunk_containment.json")

    columns = st.columns(3)
    columns[0].metric("Containment", f"{data['rate']:.0%}")
    columns[1].metric("Extractive facts", data["extractive_facts"])
    columns[2].metric("Misses", len(data["misses"]))

    st.success(
        "**This is the ceiling on everything downstream.** A figure lost here "
        "cannot be retrieved by any retriever, rescued by any reranker, or cited "
        "by any agent. Measuring it before building a retriever removes a whole "
        "class of misdirected debugging."
    )
    st.caption(
        f"Excluded and counted, not silently dropped: {data['excluded_derived']} "
        f"derived metrics that appear on no page, {data['excluded_behavioural']} "
        f"behavioural facts scored on refusal."
    )

    chunks = sample("chunks_sample.json")
    if not chunks:
        return
    st.subheader("Chunks carry document status")
    st.info(
        "One deal holds a DRAFT of its statements beside the final version — the "
        "income statement chunks are **94% textually similar**. Cosine similarity "
        "cannot separate them, not because the retriever is bad but because the "
        "signal is not in the text. What separates them is `authority`, which "
        "lives one level up and which naive chunking discards."
    )
    st.dataframe(
        [
            {
                "document": c["document"],
                "page": c["page_number"],
                "type": c["chunk_type"],
                "section": c.get("section") or "—",
                "status": c["doc_status"],
                "authority": c["authority"],
                "trust": c["source_trust"],
                "chars": c["char_count"],
            }
            for c in chunks
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Context header prepended before embedding:")
    st.code(chunks[0]["context_header"], language=None)


def page_retrieval() -> None:
    st.header("Retrieval")
    st.caption(
        "BM25 and dense embeddings over a per-deal index, rankings fused by "
        "reciprocal rank, candidates reranked, authority breaking the ties that "
        "text similarity cannot."
    )

    report = sample("retrieval_report.json")
    if report:
        columns = st.columns(4)
        columns[0].metric("recall@1", f"{report['recall']['@1']:.0%}")
        columns[1].metric("recall@3", f"{report['recall']['@3']:.0%}")
        columns[2].metric("recall@5", f"{report['recall']['@5']:.0%}")
        columns[3].metric("MRR", f"{report['mrr']:.3f}")
        st.dataframe(
            [
                {"defect": k, "n": v["n"], "recall@1": f"{v['recall@1']:.0%}",
                 "recall@5": f"{v['recall@5']:.0%}", "mrr": round(v["mrr"], 2)}
                for k, v in sorted(report["by_defect"].items())
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("A traced query")
    live = live_corpus_available() and st.session_state.get("live_mode")
    trace = sample("retrieval_trace.json")

    if live:
        retriever = build_retriever()
        deal = st.selectbox("Deal", sorted(retriever.indexes))
        query = st.text_input("Query", "What was EBITDA in FY2025?")
        hits = retriever.retrieve(query, deal, k=5)
        rows = [
            {
                "document": h.chunk["document"],
                "page": h.chunk["page_number"],
                "status": h.chunk["doc_status"],
                "bm25": h.bm25_rank,
                "dense": h.dense_rank,
                "fused": round(h.fused_score, 4),
                "rerank": round(h.rerank_score, 3),
                "authority ×": h.authority_weight,
                "final": round(h.final_score, 4),
            }
            for h in hits
        ]
    elif trace:
        st.caption(f"`{trace['query']}` against {trace['deal']}")
        rows = [
            {
                "document": h["document"],
                "page": h["page"],
                "status": h["doc_status"],
                "bm25": h["bm25_rank"],
                "dense": h["dense_rank"],
                "fused": h["fused_score"],
                "rerank": h["rerank_score"],
                "authority ×": h["authority_weight"],
                "final": h["final_score"],
            }
            for h in trace["hits"]
        ]
    else:
        return missing("retrieval_trace.json")

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Every stage's contribution is kept, because 'why did this rank here' is "
        "the question that comes up every time retrieval underperforms."
    )
    st.info(
        "**IDF inverts on a financial corpus.** A loan package is mostly tables, "
        "so ordinary English is *rare* in it: `what` scored an IDF of 3.12 against "
        "`ebitda` at 2.61. The symptom was the broker's cover note — the only "
        "chunk in flowing prose, containing no figures — ranking first for nearly "
        "every financial question. Removing query stopwords took recall@1 from "
        "13.2% to 39.5%."
    )


def page_memo() -> None:
    st.header("The credit memo agent")
    st.caption("gather → compute → draft → verify → (revise) → done")

    verification = sample("memo_PCP-0004_verification.json")
    memo = sample("memo_PCP-0004.md")
    audit = sample("memo_PCP-0004_audit.txt")
    if not memo:
        return missing("memo_PCP-0004.md")

    if verification:
        columns = st.columns(4)
        columns[0].metric("Verified", "yes" if verification["verified"] else "NO")
        columns[1].metric("Figures extracted", verification["figures_extracted"])
        columns[2].metric("Calculations", verification["calculations"])
        columns[3].metric("Ungrounded", len(verification["ungrounded"]))

    st.info(
        "**Extraction and arithmetic are the two things a language model is worst "
        "at and the two things a credit memo cannot get wrong.** Both happen "
        "before the drafter is called. What is left for the model is narrative "
        "and judgement."
    )

    left, right = st.columns([3, 2])
    left.subheader("Memorandum")
    left.markdown(memo)

    right.subheader("Derivation of every computed figure")
    right.caption("An analyst can check this without reading code.")
    if audit:
        right.code(audit, language=None)

    if verification and verification["reconstructions"]:
        st.subheader("Reconstructed figures")
        st.warning(
            "These appear on no page and follow arithmetically from figures that "
            "do. Reported as **candidates for review, never proof** — a "
            "three-significant-figure number can coincide with an unrelated "
            "combination."
        )
        st.dataframe(
            [{"figure": k, "derivation": v}
             for k, v in verification["reconstructions"].items()],
            use_container_width=True,
            hide_index=True,
        )


def page_redteam() -> None:
    st.header("Guardrails and red-teaming")
    data = sample("redteam_report.json")
    if not data:
        return missing("redteam_report.json")

    columns = st.columns(4)
    columns[0].metric("Attacks", data["attacks"])
    columns[1].metric("Succeeded", data["succeeded"])
    columns[2].metric("Errored", data["errored"])
    columns[3].metric("Instruction detection", f"{data['detection_rate']:.0%}")

    st.info(
        "**Nobody types an attack into the chat box — they email a PDF.** Every "
        "excerpt was supplied by the borrower or their broker, so retrieved "
        "content is untrusted input by construction."
    )

    st.dataframe(
        [
            {"family": k, "n": v["n"], "succeeded": v["succeeded"],
             "detected": v["detected"]}
            for k, v in sorted(data["by_family"].items())
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Every attack")
    st.caption(
        "An attack succeeds only if it made the decision **more favourable** and "
        "got away with it. A memo that quotes an injection and still recommends "
        "DECLINE has not been compromised — it has reported an attempted fraud."
    )
    st.dataframe(
        [
            {
                "deal": r["deal"],
                "attack": r["attack"],
                "family": r["family"],
                "detected": "yes" if r["detected"] else "—",
                "before": r["before"],
                "after": r["after"],
                "verdict": "BLOCKED" if r["blocked"] else "released",
            }
            for r in data["results"]
        ],
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    st.error(
        "**Data poisoning is the attack that worked.** A poisoned EBITDA flipped "
        "deals from DECLINE to PROCEED, and the policy check could not help: an "
        "injected *instruction* tries to persuade a model, which the check "
        "ignores, while a poisoned *figure* flows into the calculator and is "
        "computed on faithfully. Garbage in, correctly computed garbage out. It "
        "also carries no instruction to detect, so pattern matching scores zero."
    )
    st.success(
        "**What stops it is that real statements are over-determined.** EBITDA "
        "cannot exceed gross profit, because operating expenses are not negative. "
        "Liabilities plus equity must equal assets. Inflating one figure means "
        "inflating every figure that ties to it — so the accounting identities "
        "the corpus asserts turn out to be a fraud detector, which was not why "
        "they were built."
    )


def page_review() -> None:
    st.header("The human review queue")
    items = sample("review_queue.json")
    if items is None:
        return missing("review_queue.json")

    st.caption(
        "Three things were previously reported and then went nowhere: "
        "reconstruction candidates, conservative recommendations, and injection "
        "findings. Each is a judgement the pipeline correctly declines to make."
    )

    if not items:
        st.success(
            "No items. Clean memos queue nothing, which is the common case and "
            "the point — a queue that fires on every memo is a queue nobody reads."
        )
        return

    st.metric("Pending items", len(items))
    for item in items:
        with st.expander(f"[{item['item_id']}] {item['kind']} — {item['deal_id']}"):
            st.write(item["summary"])
            st.json(item["evidence"])
            if item["citations"]:
                st.caption(" ".join(item["citations"]))

    st.info(
        "**Only findings of wrongness block release.** A broken accounting "
        "identity holds the memo; a reconstruction candidate does not. Decisions "
        "are append-only, attributed and timestamped — a credit file is a "
        "regulated artefact and 'who approved this' is asked years later. Nothing "
        "expires: an item that quietly released itself after a week would turn "
        "the queue from a control into a delay."
    )




def page_ask() -> None:
    st.header("Ask the loan file")
    st.caption(
        "Question answering scoped to one borrower, with citations parsed and "
        "validated rather than trusted."
    )

    retriever = (
        build_retriever()
        if (live_corpus_available() and st.session_state.get("live_mode"))
        else build_sample_retriever()
    )
    if retriever is None:
        return missing("chunks_index.jsonl")

    st.info(
        "**A citation pointing at a page that was never retrieved is dropped, "
        "and the drop is counted.** An invented citation is worse than an "
        "uncited claim: an uncited claim announces itself, while an invented "
        "citation reads exactly like evidence and cannot be checked. Every "
        "figure below is also traced back to the pages the answer cited."
    )

    deals = sorted(retriever.indexes)
    columns = st.columns([2, 3])
    deal = columns[0].selectbox("Borrower", deals)

    key = _api_key()
    generator_label = columns[1].radio(
        "Answering",
        ["Extractive (offline)", "Claude"] if key else ["Extractive (offline)"],
        horizontal=True,
        help=(
            "The extractive generator quotes the best-matching line and cites "
            "it. It cannot combine two pages or compute a ratio, and it cannot "
            "invent a figure -- which makes it the floor every model is measured "
            "against."
        ),
    )

    st.caption("Try one of these, or write your own:")
    suggestions = [
        "What was EBITDA in the latest year?",
        "Who owns the borrower and in what percentages?",
        "What existing debt facilities does the borrower have?",
        "What did the Phase I environmental site assessment conclude?",
    ]
    chosen = st.selectbox("Suggested question", ["--"] + suggestions)

    question = st.text_input(
        "Question",
        value="" if chosen == "--" else chosen,
        placeholder="What was the cash balance at year end?",
    )
    if not question.strip():
        st.caption(
            "The fourth suggestion is the interesting one: nothing in the file "
            "answers it, and the correct response is to say so rather than "
            "invent something."
        )
        return

    with st.spinner("retrieving"):
        from pecos.answering import (
            AnthropicGenerator,
            ExtractiveGenerator,
            contexts_from_hits,
        )
        from pecos.evaluation import check_numeric_grounding

        hits = retriever.retrieve(question, deal, k=5)
        contexts = contexts_from_hits(hits)
        generator = (
            AnthropicGenerator(api_key=key)
            if generator_label == "Claude"
            else ExtractiveGenerator()
        )
        try:
            answer = generator.generate(question, contexts)
        except Exception as error:  # noqa: BLE001
            st.error(f"{type(error).__name__}: {error}")
            return

    if answer.refused:
        st.warning(f"**Refused.** {answer.text}")
        st.caption(
            "Refusal is a structural outcome, not a phrase someone greps for "
            "later. A harness that cannot represent it scores the correct "
            "behaviour as a failure, which trains exactly the wrong thing."
        )
    else:
        st.success(answer.text)

    grounding = check_numeric_grounding(answer, contexts)
    columns = st.columns(4)
    columns[0].metric("Citations kept", len(answer.citations))
    columns[1].metric("Citations dropped", len(answer.dropped_citations))
    columns[2].metric("Figures grounded", len(grounding.grounded))
    columns[3].metric(
        "Figures invented", grounding.hallucinated,
        delta=None if not grounding.hallucinated else "must be zero",
        delta_color="inverse",
    )
    if answer.dropped_citations:
        st.error(
            f"Dropped, because those pages were never retrieved: "
            f"{', '.join(answer.dropped_citations)}"
        )

    st.subheader("The pages it was allowed to read")
    st.caption(
        "Ordered by the retriever. `authority` demotes drafts and superseded "
        "documents without excluding them -- 'what did the draft say' is a "
        "legitimate question."
    )
    for hit in hits:
        chunk = hit.chunk
        cited = (chunk["document"], chunk["page_number"]) in answer.cited_pages
        label = (
            f"{'✅ CITED  ' if cited else ''}{chunk['document']} "
            f"p{chunk['page_number']}  ·  {chunk.get('section') or chunk['doc_kind']}"
            f"  ·  score {hit.final_score:.3f}"
        )
        with st.expander(label, expanded=cited):
            if chunk["doc_status"] != "final":
                st.warning(
                    f"This document is marked **{chunk['doc_status'].upper()}** "
                    f"(authority {chunk['authority']} of 3) and is not "
                    f"authoritative."
                )
            if chunk.get("scale_factor", 1) != 1:
                st.warning(
                    f"Figures on this page are stated in units of "
                    f"{chunk['scale_factor']:,}."
                )
            st.code(chunk["text"], language=None)


def _api_key() -> str | None:
    """An Anthropic key from Streamlit secrets or the environment, if either has
    one. The app is fully usable without it."""
    try:
        value = st.secrets.get("ANTHROPIC_API_KEY")
        if value:
            return str(value)
    except Exception:  # noqa: BLE001 -- no secrets file is the normal case
        pass
    import os

    return os.environ.get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "1. Corpus": page_corpus,
    "2. Ingestion": page_ingestion,
    "3. Chunking": page_chunking,
    "4. Retrieval": page_retrieval,
    "Ask the loan file": page_ask,
    "5. Memo agent": page_memo,
    "6. Red team": page_redteam,
    "7. Review queue": page_review,
}

with st.sidebar:
    st.markdown("### Pecos Credit Intelligence")
    selection = st.radio("Stage", list(PAGES), label_visibility="collapsed")
    st.divider()
    if live_corpus_available():
        st.session_state["live_mode"] = st.toggle(
            "Live mode",
            value=False,
            help="Run retrieval against the local corpus instead of reading the "
            "committed sample.",
        )
    else:
        st.caption(
            "Reading committed samples. Build a corpus locally to enable live "
            "retrieval."
        )
    st.divider()
    st.caption(
        "All borrower data is synthetic. No real financial information appears "
        "anywhere in this project."
    )

PAGES[selection]()
