"""Chunk the page extractions into retrievable units.

    python scripts/chunk_corpus.py               # chunk everything
    python scripts/chunk_corpus.py --audit       # and score against ground truth

Reads   data/interim/extractions/*.jsonl
Writes  data/interim/chunks/<deal_id>.jsonl
        data/interim/chunks/chunk_summary.json

The `--audit` flag is the one worth running. It reports the **retrieval
ceiling**: the fraction of extractive gold facts that survive into a chunk
anchored to the page the manifest cites. Nothing downstream can score above that
number, so a drop here invalidates every retrieval result measured afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.chunk_audit import audit_containment, audit_near_duplicates  # noqa: E402
from pecos.chunking import CHUNKER_VERSION, chunk_corpus, load_chunks  # noqa: E402
from pecos.config import REPO_ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk the Pecos extractions.")
    parser.add_argument("--extractions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="score chunk containment against the ground-truth manifest",
    )
    args = parser.parse_args()

    extractions = args.extractions or (REPO_ROOT / "data" / "interim" / "extractions")
    out_dir = args.out or (REPO_ROOT / "data" / "interim" / "chunks")
    manifest_path = args.manifest or (
        REPO_ROOT / "data" / "raw" / "corpus_manifest.json"
    )

    if not extractions.is_dir() or not any(extractions.glob("*.jsonl")):
        print(f"No extractions at {extractions}.")
        print("Run: python scripts/ingest_corpus.py")
        return 1

    print(f"chunker     {CHUNKER_VERSION}")
    print(f"input       {extractions}")
    print(f"output      {out_dir}")
    print("-" * 62)

    started = time.time()
    summary = chunk_corpus(extractions, out_dir)
    elapsed = time.time() - started

    print(f"deals             {summary.deals}")
    print(f"pages             {summary.pages}")
    print(f"chunks            {summary.chunks}")
    print(f"  prose           {summary.prose_chunks}")
    print(f"  table           {summary.table_chunks}")
    print(f"draft chunks      {summary.draft_chunks}")
    print(f"superseded chunks {summary.superseded_chunks}")
    print(f"rescaled chunks   {summary.scaled_chunks}")
    print(f"mean chars        {summary.mean_chunk_chars}")
    print(f"max chars         {summary.max_chunk_chars}")
    print(f"elapsed           {elapsed:.1f}s")

    if summary.pages_with_no_chunks:
        # A page that produced nothing is a page nothing can be retrieved from.
        print("-" * 62)
        print(
            f"WARNING: {len(summary.pages_with_no_chunks)} page(s) produced no chunks:"
        )
        for page in summary.pages_with_no_chunks[:20]:
            print(f"  {page}")
        return 2

    if not args.audit:
        return 0

    if not manifest_path.exists():
        print(f"\nNo manifest at {manifest_path}; skipping audit.")
        return 0

    chunks: list[dict] = []
    for path in sorted(out_dir.glob("PCP-*.jsonl")):
        chunks.extend(load_chunks(path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = audit_containment(chunks, manifest["facts"])
    print("-" * 62)
    print("CONTAINMENT AUDIT -- the ceiling on everything downstream")
    print(f"  extractive facts   {result.extractive}")
    print(f"  found in a chunk   {result.found}")
    print(f"  containment        {result.rate:.1%}")
    print(f"  excluded, derived  {result.excluded_derived}  (never printed on a page)")
    print(f"  excluded, refusal  {result.excluded_behavioural}")
    print(f"  excluded, no deal  {result.excluded_out_of_scope}")

    if result.by_defect:
        print("  defect coverage")
        for defect, count in sorted(result.by_defect.items()):
            print(f"    {defect:<24} {count}")

    if result.misses:
        print("  MISSES")
        for miss in result.misses:
            print(
                f"    {miss.fact_id}  {miss.needle!r} not in any of "
                f"{miss.chunks_on_page} chunk(s) on "
                f"{miss.document}#{miss.page_number}"
            )

    near = audit_near_duplicates(chunks)
    if near:
        print("-" * 62)
        print("NON-FINAL DOCUMENTS -- separable by metadata, not by text")
        for deal_id, entry in sorted(near.items()):
            print(f"  {deal_id}: {entry['statuses']} authority={entry['authorities']}")
            for document in entry["documents"]:
                print(f"    {document}")

    return 0 if not result.misses else 3


if __name__ == "__main__":
    raise SystemExit(main())
