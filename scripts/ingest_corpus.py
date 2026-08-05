"""Ingest the loan-package corpus into page-level extractions.

    python scripts/ingest_corpus.py                    # everything
    python scripts/ingest_corpus.py --deals PCP-0002   # one deal
    python scripts/ingest_corpus.py --limit 3          # first three deals

Reads   data/raw/packages/<deal_id>/*.pdf
Writes  data/interim/extractions/<deal_id>.jsonl
        data/interim/extractions/ingest_summary.json

OCR is the slow part -- roughly two seconds a scanned page -- so a full pass
over twelve deals takes a few minutes. Nothing here is incremental yet; re-runs
redo the work. That is deliberate for now, because a stale extraction cache is a
far nastier bug than a slow rebuild, and the corpus is small enough that the
rebuild is affordable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.config import REPO_ROOT  # noqa: E402
from pecos.ingest import EXTRACTOR_VERSION, ingest_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the Pecos corpus.")
    parser.add_argument(
        "--deals", nargs="*", default=None, help="specific deal ids, e.g. PCP-0002"
    )
    parser.add_argument("--limit", type=int, default=None, help="first N deals only")
    parser.add_argument("--packages", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    packages_root = args.packages or (REPO_ROOT / "data" / "raw" / "packages")
    out_dir = args.out or (REPO_ROOT / "data" / "interim" / "extractions")

    if not packages_root.is_dir():
        print(f"No corpus at {packages_root}.")
        print("Run: python scripts/generate_corpus.py")
        return 1

    deal_ids = args.deals
    if deal_ids is None and args.limit is not None:
        folders = sorted(p.name for p in packages_root.iterdir() if p.is_dir())
        deal_ids = folders[: args.limit]

    print(f"extractor   {EXTRACTOR_VERSION}")
    print(f"packages    {packages_root}")
    print(f"output      {out_dir}")
    print(f"deals       {', '.join(deal_ids) if deal_ids else 'all'}")
    print("-" * 60)

    started = time.time()
    summary = ingest_corpus(packages_root, out_dir, deal_ids=deal_ids)
    elapsed = time.time() - started

    print(f"deals            {summary.deals}")
    print(f"documents        {summary.documents}")
    print(f"pages            {summary.pages}")
    print(f"  digital        {summary.digital_pages}")
    print(f"  ocr            {summary.ocr_pages}")
    print(f"rotated pages    {summary.rotated_pages}")
    print(f"scaled pages     {summary.scaled_pages}")
    print(f"tables           {summary.tables}")
    print(f"mean ocr conf    {summary.mean_ocr_confidence}")
    print(f"elapsed          {elapsed:.1f}s")

    if summary.empty_pages:
        # Never silent. A page that produced no text is a page the pipeline
        # cannot answer from, and it should be visible here rather than
        # discovered as a retrieval miss three milestones later.
        print("-" * 60)
        print(f"WARNING: {len(summary.empty_pages)} page(s) produced no text:")
        for page in summary.empty_pages[:20]:
            print(f"  {page}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
