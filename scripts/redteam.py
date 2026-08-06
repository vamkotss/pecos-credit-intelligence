"""Run the red-team suite against the memo agent.

    python scripts/redteam.py                       # offline drafter
    python scripts/redteam.py --drafter anthropic   # against Claude
    python scripts/redteam.py --limit 3 --out reports/redteam.json

Exit code 5 if any attack changed a credit decision without being blocked.

The default drafter is a useful control rather than a weak target: it cannot be
persuaded of anything, so an instruction attack that "succeeds" against it would
mean the harness is broken. The interesting run is `--drafter anthropic`, where
the model can in principle be talked into something and the question is whether
the deterministic checks catch it anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.chunking import load_chunks  # noqa: E402
from pecos.config import REPO_ROOT, settings  # noqa: E402
from pecos.drafting import AnthropicDrafter, TemplateDrafter  # noqa: E402
from pecos.redteam import ATTACKS, format_redteam, run_redteam  # noqa: E402
from pecos.retrieval import HybridRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Red-team the Pecos memo agent.")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument(
        "--drafter", default="template", choices=["template", "anthropic"]
    )
    parser.add_argument("--limit", type=int, default=None, help="first N deals")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    chunks_dir = args.chunks or (REPO_ROOT / "data" / "interim" / "chunks")
    if not chunks_dir.is_dir() or not any(chunks_dir.glob("PCP-*.jsonl")):
        print(f"No chunks at {chunks_dir}. Run ingest_corpus.py then chunk_corpus.py.")
        return 1
    if args.drafter == "anthropic" and not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Use --drafter template.")
        return 1

    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("PCP-*.jsonl")):
        chunks.extend(load_chunks(path))

    retriever = HybridRetriever()
    retriever.build(chunks)
    deals = sorted(retriever.indexes)
    if args.limit:
        deals = deals[: args.limit]

    drafter = TemplateDrafter() if args.drafter == "template" else AnthropicDrafter()

    print(f"drafter      {args.drafter}")
    print(f"deals        {len(deals)}")
    print(f"attacks      {len(ATTACKS)}")
    print("-" * 62)

    started = time.time()

    def progress(result) -> None:
        # Streamed as each attack finishes. A fifteen-minute command that prints
        # nothing until the end is indistinguishable from a hang, and the first
        # Anthropic run looked exactly like one.
        print(result.line(), flush=True)

    report = run_redteam(retriever, drafter, deals, on_result=progress)
    print("-" * 62)
    print(format_redteam(report))
    print(f"\nelapsed      {time.time() - started:.1f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "drafter": args.drafter,
                    "attacks": report.n,
                    "succeeded": len(report.successes),
                    "errored": len(report.errors),
                    "more_conservative": len(report.conservative),
                    "success_rate": report.success_rate,
                    "detection_rate": report.detection_rate,
                    "by_family": {
                        family: {
                            "n": len(rows),
                            "succeeded": sum(1 for r in rows if r.succeeded),
                            "detected": sum(1 for r in rows if r.detected),
                        }
                        for family, rows in report.by_family().items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"wrote {args.out}")

    return 5 if report.successes else 0


if __name__ == "__main__":
    raise SystemExit(main())
