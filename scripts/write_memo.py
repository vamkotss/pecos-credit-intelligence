"""Write a credit memorandum for one deal, or all of them.

    python scripts/write_memo.py --deal PCP-0001
    python scripts/write_memo.py --all
    python scripts/write_memo.py --deal PCP-0001 --drafter anthropic
    python scripts/write_memo.py --all --out reports/memos

The default drafter needs no API key. Extraction and arithmetic happen before
either drafter is called, so the two things a credit memo cannot get wrong are
not the model's job in either mode.

Exit code 4 if any memo contains a figure that is neither quoted from a cited
page nor produced by a recorded calculation. An unverifiable figure in a credit
memo is the failure this whole project is built around, so it fails the command
rather than printing a warning.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.chunking import load_chunks  # noqa: E402
from pecos.config import REPO_ROOT, settings  # noqa: E402
from pecos.drafting import AnthropicDrafter, TemplateDrafter  # noqa: E402
from pecos.memo import MemoWriter  # noqa: E402
from pecos.retrieval import HybridRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Pecos credit memos.")
    parser.add_argument("--deal", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument(
        "--drafter", default="template", choices=["template", "anthropic"]
    )
    parser.add_argument("--k", type=int, default=6, help="pages retrieved per section")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--audit", action="store_true", help="print the derivations")
    args = parser.parse_args()

    if not args.deal and not args.all:
        print("Pass --deal PCP-0001 or --all.")
        return 1

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

    deals = sorted(retriever.indexes) if args.all else [args.deal]
    missing = [d for d in deals if d not in retriever.indexes]
    if missing:
        print(f"No chunks for: {', '.join(missing)}")
        return 1

    drafter = TemplateDrafter() if args.drafter == "template" else AnthropicDrafter()
    writer = MemoWriter(retriever=retriever, drafter=drafter, k=args.k)

    print(f"drafter      {args.drafter}")
    print(f"deals        {len(deals)}")
    print("-" * 62)

    failures = 0
    started = time.time()
    for deal_id in deals:
        result = writer.write(deal_id)
        status = "VERIFIED" if result.verified else "UNVERIFIED"
        print(
            f"{deal_id}  {status:<11} figures={result.figures_extracted:<3} "
            f"calcs={len(result.computations.entries):<2} "
            f"citations={len(result.citations):<3} revisions={result.revisions}"
        )
        if not result.verified:
            failures += 1
            print(f"  ungrounded figures: {', '.join(result.ungrounded)}")
        # Reconstructions are shown, never silently accepted. A three-figure
        # number can match a combination by coincidence, so the arithmetic is
        # put in front of a reviewer rather than treated as proof.
        for figure, derivation in result.reconstructions.items():
            print(f"  reconstructed {figure}: {derivation}")

        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{deal_id}.md").write_text(result.text, encoding="utf-8")
        elif len(deals) == 1:
            print()
            print(result.text)

        if args.audit and result.computations.entries:
            print()
            print(result.audit_trail())

    print("-" * 62)
    print(f"elapsed      {time.time() - started:.1f}s")
    if args.out:
        print(f"wrote        {args.out}")
    if failures:
        print(
            f"UNVERIFIED   {failures} of {len(deals)} memos contain ungrounded figures"
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
