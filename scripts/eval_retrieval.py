"""Build the hybrid index and score retrieval against ground truth.

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --no-authority       # ablation
    python scripts/eval_retrieval.py --embedder st        # sentence-transformers
    python scripts/eval_retrieval.py --reranker cross     # cross-encoder

Reads   data/interim/chunks/*.jsonl
        data/raw/corpus_manifest.json

The two flags at the bottom of that list are the point of the interfaces. The
defaults need no model download and no network, so the suite stays hermetic; the
alternatives are stronger and measurable on the same gold set, which turns
"a trained encoder would be better" from an assumption into a number.

`--no-authority` is the ablation that matters most. It disables the authority
weighting and re-scores, which is the only way to show what that weighting is
actually buying on the near-duplicate and restatement defects.
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
from pecos.retrieval import (  # noqa: E402
    CrossEncoderReranker,
    HybridRetriever,
    LexicalOverlapReranker,
    LsaEmbedder,
    SentenceTransformerEmbedder,
)
from pecos.retrieval_eval import evaluate_retrieval, format_report  # noqa: E402


def build_embedder(name: str):
    if name == "lsa":
        return LsaEmbedder(seed=settings.seed)
    if name == "st":
        return SentenceTransformerEmbedder()
    raise SystemExit(f"unknown embedder: {name}")


def build_reranker(name: str):
    if name == "lexical":
        return LexicalOverlapReranker()
    if name == "cross":
        return CrossEncoderReranker()
    raise SystemExit(f"unknown reranker: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Pecos retrieval.")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--embedder", default="lsa", choices=["lsa", "st"])
    parser.add_argument("--reranker", default="lexical", choices=["lexical", "cross"])
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument(
        "--no-authority",
        action="store_true",
        help="ablation: disable the authority weighting",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write the report as JSON"
    )
    args = parser.parse_args()

    chunks_dir = args.chunks or (REPO_ROOT / "data" / "interim" / "chunks")
    manifest_path = args.manifest or (
        REPO_ROOT / "data" / "raw" / "corpus_manifest.json"
    )

    if not chunks_dir.is_dir() or not any(chunks_dir.glob("PCP-*.jsonl")):
        print(f"No chunks at {chunks_dir}.")
        print("Run: python scripts/ingest_corpus.py && python scripts/chunk_corpus.py")
        return 1

    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("PCP-*.jsonl")):
        chunks.extend(load_chunks(path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"chunks       {len(chunks)}")
    print(f"embedder     {args.embedder}")
    print(f"reranker     {args.reranker}")
    print(f"authority    {'off (ablation)' if args.no_authority else 'on'}")
    print("-" * 62)

    started = time.time()
    retriever = HybridRetriever(
        embedder=build_embedder(args.embedder),
        reranker=build_reranker(args.reranker),
        candidate_k=args.candidates,
        use_authority=not args.no_authority,
    )
    retriever.build(chunks)
    build_seconds = time.time() - started

    started = time.time()
    report = evaluate_retrieval(retriever, manifest["facts"])
    eval_seconds = time.time() - started

    print(f"deals indexed         {len(retriever.indexes)}")
    print(f"index build           {build_seconds:.1f}s")
    print(f"queries               {eval_seconds:.1f}s")
    print("-" * 62)
    print(format_report(report))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": args.embedder,
            "reranker": args.reranker,
            "authority": not args.no_authority,
            "chunks": len(chunks),
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
        }
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
