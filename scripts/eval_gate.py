"""Run every free, deterministic check and gate the build on the results.

    python scripts/eval_gate.py
    python scripts/eval_gate.py --out reports/gate.json

Exit code 6 if any threshold is breached or any metric is missing.

Everything here runs offline in seconds and gives the same answer every time,
which is what makes it safe to block a merge on. Metrics needing an API key --
Claude's answer accuracy, judged faithfulness -- are deliberately excluded: a
build that fails because a provider had a bad afternoon teaches people to ignore
red builds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.answering import ExtractiveGenerator  # noqa: E402
from pecos.chunk_audit import audit_containment  # noqa: E402
from pecos.chunking import load_chunks  # noqa: E402
from pecos.config import REPO_ROOT  # noqa: E402
from pecos.drafting import TemplateDrafter  # noqa: E402
from pecos.evaluation import evaluate_answers  # noqa: E402
from pecos.gate import as_dict, evaluate_gate, format_gate  # noqa: E402
from pecos.memo import MemoWriter  # noqa: E402
from pecos.redteam import run_redteam  # noqa: E402
from pecos.retrieval import HybridRetriever  # noqa: E402
from pecos.retrieval_eval import evaluate_retrieval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Pecos eval gate.")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--redteam-deals", type=int, default=3)
    args = parser.parse_args()

    chunks_dir = args.chunks or (REPO_ROOT / "data" / "interim" / "chunks")
    manifest_path = args.manifest or (
        REPO_ROOT / "data" / "raw" / "corpus_manifest.json"
    )
    if not chunks_dir.is_dir() or not any(chunks_dir.glob("PCP-*.jsonl")):
        print(f"No chunks at {chunks_dir}. Run ingest_corpus.py then chunk_corpus.py.")
        return 1

    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("PCP-*.jsonl")):
        chunks.extend(load_chunks(path))
    facts = json.loads(manifest_path.read_text(encoding="utf-8"))["facts"]

    started = time.time()
    metrics: dict[str, float] = {}

    # --- M4 containment ----------------------------------------------------
    containment = audit_containment(chunks, facts)
    metrics["chunk_containment"] = containment.rate

    # --- M5 retrieval ------------------------------------------------------
    retriever = HybridRetriever()
    retriever.build(chunks)
    retrieval = evaluate_retrieval(retriever, facts)
    metrics["retrieval_recall_at_1"] = retrieval.recall_at(1)
    metrics["retrieval_recall_at_5"] = retrieval.recall_at(5)
    metrics["retrieval_mrr"] = retrieval.mrr()

    # --- M6 answering, extractive baseline ---------------------------------
    answers = evaluate_answers(ExtractiveGenerator(), retriever, facts)
    metrics["baseline_grounding_rate"] = answers.grounding_rate
    metrics["baseline_hallucinated_figures"] = answers.hallucinated_figures
    metrics["baseline_invented_citations"] = answers.invented_citations
    metrics["over_refusal_rate"] = answers.over_refusal_rate
    metrics["refusal_accuracy"] = (
        answers.refusal_accuracy if answers.refusal_accuracy is not None else 1.0
    )

    # --- M7 memos ----------------------------------------------------------
    writer = MemoWriter(retriever=retriever, drafter=TemplateDrafter())
    deals = sorted(retriever.indexes)
    verified = sum(1 for deal in deals if writer.write(deal).verified)
    metrics["memos_verified_rate"] = verified / len(deals) if deals else 0.0

    # --- M8 red team -------------------------------------------------------
    redteam = run_redteam(retriever, TemplateDrafter(), deals[: args.redteam_deals])
    metrics["redteam_successes"] = len(redteam.successes)
    metrics["redteam_instruction_detection"] = redteam.detection_rate

    result = evaluate_gate(metrics)
    print(format_gate(result))
    print(f"\nelapsed  {time.time() - started:.1f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(as_dict(result), indent=2, sort_keys=True))
        print(f"wrote {args.out}")

    return 0 if result.passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
