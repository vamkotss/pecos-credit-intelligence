"""Export sample artefacts so the pipeline can be understood without running it.

    python scripts/export_samples.py

Writes `docs/samples/`: real output from a real run, small enough to commit and
readable on GitHub. The Streamlit app falls back to these when no local corpus
exists, so someone who clones the repository can see what the system does before
installing Tesseract.

Two reasons this exists beyond convenience.

**Every number in the README becomes checkable.** Claims about recall, grounding
and attack resistance are worth more when the artefact they came from is sitting
in the repository next to them.

**The interesting output is not the summary line.** "GATE PASSED" says little;
the derivation of a pro forma leverage figure, or an attack visibly flipping a
decision and being blocked, is what shows how the system works.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.answering import ExtractiveGenerator, contexts_from_hits  # noqa: E402
from pecos.chunk_audit import audit_containment  # noqa: E402
from pecos.chunking import load_chunks  # noqa: E402
from pecos.config import REPO_ROOT  # noqa: E402
from pecos.drafting import TemplateDrafter  # noqa: E402
from pecos.evaluation import evaluate_answers  # noqa: E402
from pecos.gate import as_dict, evaluate_gate  # noqa: E402
from pecos.guardrails import check_memo  # noqa: E402
from pecos.memo import MEMO_QUESTIONS, MemoWriter, extract_figures  # noqa: E402
from pecos.redteam import format_redteam, run_redteam  # noqa: E402
from pecos.retrieval import HybridRetriever  # noqa: E402
from pecos.retrieval_eval import evaluate_retrieval, format_report  # noqa: E402
from pecos.review import build_queue_for_memo  # noqa: E402

OUT = REPO_ROOT / "docs" / "samples"

# Deals chosen for what they show, not at random. Ideally the first is the
# richest memo, and the others carry defects worth looking at.
SHOWCASE = "PCP-0004"


def _write(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"  {name:<34} {path.stat().st_size / 1024:.1f} KB")


def main() -> int:
    chunks_dir = REPO_ROOT / "data" / "interim" / "chunks"
    manifest_path = REPO_ROOT / "data" / "raw" / "corpus_manifest.json"
    if not chunks_dir.is_dir() or not any(chunks_dir.glob("PCP-*.jsonl")):
        print("No chunks. Run generate_corpus, ingest_corpus and chunk_corpus first.")
        return 1

    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("PCP-*.jsonl")):
        chunks.extend(load_chunks(path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    facts = manifest["facts"]

    retriever = HybridRetriever()
    retriever.build(chunks)
    deals = sorted(retriever.indexes)
    showcase = SHOWCASE if SHOWCASE in retriever.indexes else deals[0]

    print("exporting samples")

    # --- corpus ------------------------------------------------------------
    _write(
        "corpus_summary.json",
        {
            "seed": manifest["seed"],
            "deals": manifest["n_deals"],
            "gold_facts": len(facts),
            "defect_index": manifest["defect_index"],
            "deal_summaries": manifest["deals"][:4],
        },
    )

    # --- ingestion: one digital page and one OCR page ----------------------
    extractions = REPO_ROOT / "data" / "interim" / "extractions"
    pages = []
    if extractions.is_dir():
        for path in sorted(extractions.glob("PCP-*.jsonl"))[:2]:
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                # One of each kind, plus the rotated page if it is here: those
                # three show the whole routing decision.
                interesting = (
                    record["method"] == "ocr" or record["rotation_applied"] or len(pages) < 1
                )
                if interesting and len(pages) < 3:
                    record["text"] = record["text"][:1200]
                    record["tables"] = record["tables"][:2]
                    pages.append(record)
    _write("ingestion_pages.json", pages)

    # --- chunking ----------------------------------------------------------
    containment = audit_containment(chunks, facts)
    _write(
        "chunk_containment.json",
        {
            "extractive_facts": containment.extractive,
            "found": containment.found,
            "rate": containment.rate,
            "excluded_derived": containment.excluded_derived,
            "excluded_behavioural": containment.excluded_behavioural,
            "by_defect": dict(containment.by_defect),
            "misses": [m.__dict__ for m in containment.misses],
        },
    )
    _write(
        "chunks_sample.json",
        [c for c in chunks if c["deal_id"] == showcase][:6],
    )

    # --- retrieval ---------------------------------------------------------
    report = evaluate_retrieval(retriever, facts)
    _write(
        "retrieval_report.json",
        {
            "queries": report.n,
            "recall": {f"@{k}": report.recall_at(k) for k in report.ks},
            "mrr": report.mrr(),
            "by_defect": {
                defect: {
                    "n": len(rows),
                    "recall@1": report.recall_at(1, rows),
                    "recall@5": report.recall_at(5, rows),
                    "mrr": report.mrr(rows),
                }
                for defect, rows in report.by_defect().items()
            },
        },
    )
    _write("retrieval_report.txt", format_report(report))

    # A traced query, so the per-stage scores are visible.
    query = "What was EBITDA in FY2025?"
    traced = [
        {
            "document": hit.chunk["document"],
            "page": hit.chunk["page_number"],
            "section": hit.chunk.get("section"),
            "doc_status": hit.chunk["doc_status"],
            "authority": hit.chunk["authority"],
            "bm25_rank": hit.bm25_rank,
            "dense_rank": hit.dense_rank,
            "fused_score": round(hit.fused_score, 4),
            "rerank_score": round(hit.rerank_score, 3),
            "authority_weight": hit.authority_weight,
            "final_score": round(hit.final_score, 4),
            "text": hit.chunk["text"][:220],
        }
        for hit in retriever.retrieve(query, showcase, k=5)
    ]
    _write("retrieval_trace.json", {"deal": showcase, "query": query, "hits": traced})

    # --- answering ---------------------------------------------------------
    answers = evaluate_answers(ExtractiveGenerator(), retriever, facts)
    _write(
        "answer_baseline.json",
        {
            "generator": answers.generator,
            "questions": answers.n,
            "citation_accuracy": answers.citation_accuracy,
            "answer_accuracy": answers.answer_accuracy,
            "grounding_rate": answers.grounding_rate,
            "hallucinated_figures": answers.hallucinated_figures,
            "invented_citations": answers.invented_citations,
            "refusal_accuracy": answers.refusal_accuracy,
            "over_refusal_rate": answers.over_refusal_rate,
        },
    )

    # --- memo --------------------------------------------------------------
    writer = MemoWriter(retriever=retriever, drafter=TemplateDrafter(), k=6)
    result = writer.write(showcase)
    _write(f"memo_{showcase}.md", result.text)
    _write(f"memo_{showcase}_audit.txt", result.audit_trail())
    _write(
        f"memo_{showcase}_verification.json",
        {
            "deal": showcase,
            "verified": result.verified,
            "figures_extracted": result.figures_extracted,
            "calculations": len(result.computations.entries),
            "citations": list(result.citations),
            "reconstructions": result.reconstructions,
            "ungrounded": list(result.ungrounded),
            "revisions": result.revisions,
        },
    )

    # --- guardrails and red team -------------------------------------------
    redteam = run_redteam(retriever, TemplateDrafter(), deals[:3])
    _write(
        "redteam_report.json",
        {
            "attacks": redteam.n,
            "succeeded": len(redteam.successes),
            "errored": len(redteam.errors),
            "detection_rate": redteam.detection_rate,
            "by_family": {
                family: {
                    "n": len(rows),
                    "succeeded": sum(1 for r in rows if r.succeeded),
                    "detected": sum(1 for r in rows if r.detected),
                }
                for family, rows in redteam.by_family().items()
            },
            "results": [
                {
                    "deal": r.deal_id,
                    "attack": r.attack,
                    "family": r.family,
                    "detected": r.detected,
                    "before": r.recommendation_before,
                    "after": r.recommendation_after,
                    "blocked": r.blocked,
                    "notes": r.notes[:160],
                }
                for r in redteam.results
            ],
        },
    )
    _write("redteam_report.txt", format_redteam(redteam))

    # --- gate ---------------------------------------------------------------
    metrics = {
        "chunk_containment": containment.rate,
        "retrieval_recall_at_1": report.recall_at(1),
        "retrieval_recall_at_5": report.recall_at(5),
        "retrieval_mrr": report.mrr(),
        "baseline_grounding_rate": answers.grounding_rate,
        "baseline_hallucinated_figures": answers.hallucinated_figures,
        "baseline_invented_citations": answers.invented_citations,
        "over_refusal_rate": answers.over_refusal_rate,
        "refusal_accuracy": answers.refusal_accuracy or 1.0,
        "memos_verified_rate": sum(
            1 for d in deals if writer.write(d).verified
        ) / len(deals),
        "redteam_successes": len(redteam.successes),
        "redteam_instruction_detection": redteam.detection_rate,
    }
    _write("eval_gate.json", as_dict(evaluate_gate(metrics)))

    # --- a searchable index for the deployed app ---------------------------
    # The chat page needs real chunks, and the deployed app has no corpus. Two
    # deals is enough to demonstrate retrieval and small enough to commit; the
    # whole corpus would be several megabytes of text nobody reads.
    indexed = [c for c in chunks if c["deal_id"] in deals[:2]]
    for chunk in indexed:
        # Drop the fields the app never reads. Halves the file and keeps the
        # committed artefact honest about what it is for.
        chunk.pop("mean_word_confidence", None)
    (OUT / "chunks_index.jsonl").write_text(
        "\n".join(json.dumps(c, sort_keys=True) for c in indexed) + "\n",
        encoding="utf-8",
    )
    print(f"  {'chunks_index.jsonl':<34} "
          f"{(OUT / 'chunks_index.jsonl').stat().st_size / 1024:.1f} KB "
          f"({len(indexed)} chunks, {len(deals[:2])} deals)")

    # --- review queue -------------------------------------------------------
    items = []
    for deal_id in deals[:4]:
        memo = writer.write(deal_id)
        contexts = []
        seen: set[str] = set()
        for _, question in MEMO_QUESTIONS:
            for context in contexts_from_hits(retriever.retrieve(question, deal_id, 6)):
                if context["chunk_id"] not in seen:
                    seen.add(context["chunk_id"])
                    contexts.append(context)
        guard = check_memo(
            deal_id, memo.text, memo.computations, contexts, extract_figures(contexts)
        )
        items += [i.to_record() for i in build_queue_for_memo(deal_id, memo, guard)]
    _write("review_queue.json", items)

    print(f"\nwrote {len(list(OUT.glob('*')))} file(s) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
