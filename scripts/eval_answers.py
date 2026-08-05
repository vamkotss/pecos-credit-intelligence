"""Answer every gold question and score the result.

    python scripts/eval_answers.py                          # offline baseline
    python scripts/eval_answers.py --generator anthropic    # Claude answers
    python scripts/eval_answers.py --generator anthropic --judge anthropic
    python scripts/eval_answers.py --limit 10 --out reports/answers.json

The defaults need no API key and no network: an extractive generator that quotes
the best-matching line, and a token-overlap proxy in place of a judge. That keeps
CI hermetic and free.

The extractive baseline is a floor, not a competitor. Because it only quotes, it
is grounded by construction -- so any generator scoring below it on grounding is
measurably making things worse. What it cannot do is combine two pages, compute a
ratio, or interpret a paraphrase, and the gap on *answer accuracy* is exactly
what the language model is being paid for.

Costs: --generator anthropic issues one call per question, --judge anthropic
another. On Haiku, a full run over the corpus is cents. Use --limit while
iterating.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.answering import AnthropicGenerator, ExtractiveGenerator  # noqa: E402
from pecos.chunking import load_chunks  # noqa: E402
from pecos.config import REPO_ROOT, settings  # noqa: E402
from pecos.evaluation import (  # noqa: E402
    AnthropicJudge,
    OverlapJudge,
    evaluate_answers,
    format_evaluation,
)
from pecos.retrieval import HybridRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Pecos answers.")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--generator", default="extractive", choices=["extractive", "anthropic"]
    )
    parser.add_argument("--judge", default="overlap", choices=["overlap", "anthropic"])
    parser.add_argument("--k", type=int, default=5, help="pages given to the generator")
    parser.add_argument("--limit", type=int, default=None, help="first N questions")
    parser.add_argument("--deals", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    chunks_dir = args.chunks or (REPO_ROOT / "data" / "interim" / "chunks")
    manifest_path = args.manifest or (
        REPO_ROOT / "data" / "raw" / "corpus_manifest.json"
    )
    if not chunks_dir.is_dir() or not any(chunks_dir.glob("PCP-*.jsonl")):
        print(f"No chunks at {chunks_dir}. Run ingest_corpus.py then chunk_corpus.py.")
        return 1

    if "anthropic" in (args.generator, args.judge) and not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set.")
        print("Set it, or use the defaults which need no key.")
        return 1

    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("PCP-*.jsonl")):
        chunks.extend(load_chunks(path))
    facts = json.loads(manifest_path.read_text(encoding="utf-8"))["facts"]
    if args.limit:
        facts = facts[: args.limit]

    generator = (
        ExtractiveGenerator()
        if args.generator == "extractive"
        else AnthropicGenerator()
    )
    judge = OverlapJudge() if args.judge == "overlap" else AnthropicJudge()

    retriever = HybridRetriever()
    retriever.build(chunks)

    print(f"chunks       {len(chunks)}")
    print(f"generator    {args.generator}")
    print(f"judge        {args.judge}")
    print(f"pages per q  {args.k}")
    print("-" * 62)

    started = time.time()
    report = evaluate_answers(
        generator,
        retriever,
        facts,
        judge=judge,
        k=args.k,
        deal_ids=set(args.deals) if args.deals else None,
    )
    elapsed = time.time() - started

    print(format_evaluation(report))
    print(f"\nelapsed               {elapsed:.1f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "generator": report.generator,
                    "judge": report.judge,
                    "questions": report.n,
                    "citation_accuracy": report.citation_accuracy,
                    "answer_accuracy": report.answer_accuracy,
                    "grounding_rate": report.grounding_rate,
                    "hallucinated_figures": report.hallucinated_figures,
                    "invented_citations": report.invented_citations,
                    "over_refusal_rate": report.over_refusal_rate,
                    "refusal_accuracy": report.refusal_accuracy,
                    "faithfulness": report.faithfulness,
                    "relevance": report.relevance,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"wrote {args.out}")

    # A hallucinated figure is the one failure that must never pass silently.
    return 3 if report.hallucinated_figures else 0


if __name__ == "__main__":
    raise SystemExit(main())
