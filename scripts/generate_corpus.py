"""Generate the synthetic loan-package corpus.

    python scripts/generate_corpus.py                # full corpus
    python scripts/generate_corpus.py --deals 2      # quick check
    python scripts/generate_corpus.py --no-pdfs      # manifest only, seconds

This script is the ONLY place in the corpus code path that touches `settings`.
Everything under `src/pecos/corpus*.py` takes an explicit `CorpusSpec`, which is
what lets the generator be tested with no environment set up at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `src/` importable when the script is run directly, so the project does
# not have to be pip-installed just to generate data.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.config import REPO_ROOT, settings  # noqa: E402
from pecos.corpus import CorpusSpec  # noqa: E402
from pecos.corpus_build import generate_corpus  # noqa: E402

# Corpus size. CI generates a small corpus because rasterising scans is slow,
# but round-robin defect assignment guarantees all seven failure modes are still
# present -- so the fast run is a real test, not a token one.
FULL_DEALS = 12
CI_DEALS = 3


def _require(name: str):
    """Read an attribute off `settings`, failing with a useful message.

    M2 depends on exactly two settings from M1. Naming them here means that if
    `config.py` is ever refactored, the failure is a one-line explanation rather
    than an AttributeError from three frames deep.
    """
    if not hasattr(settings, name):
        raise AttributeError(
            f"pecos.config.settings has no attribute '{name}'. "
            f"The corpus generator needs 'seed' and 'ci_mode'. "
            f"Either add it to Settings in src/pecos/config.py or update this script."
        )
    return getattr(settings, name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Pecos synthetic corpus.")
    parser.add_argument(
        "--deals", type=int, default=None, help="number of loan packages"
    )
    parser.add_argument(
        "--years", type=int, default=3, help="fiscal years per borrower"
    )
    parser.add_argument("--seed", type=int, default=None, help="override PC_SEED")
    parser.add_argument(
        "--out", type=Path, default=None, help="output root (default data/raw)"
    )
    parser.add_argument(
        "--no-pdfs",
        action="store_true",
        help="build the manifest and gold set without rendering any PDF",
    )
    args = parser.parse_args()

    ci_mode = bool(_require("ci_mode"))
    seed = args.seed if args.seed is not None else int(_require("seed"))
    deals = (
        args.deals if args.deals is not None else (CI_DEALS if ci_mode else FULL_DEALS)
    )
    out_dir = args.out if args.out is not None else REPO_ROOT / "data" / "raw"

    spec = CorpusSpec(seed=seed, n_deals=deals, out_dir=out_dir, years=args.years)

    print(f"seed        {spec.seed}")
    print(f"deals       {spec.n_deals}")
    print(f"years       {spec.years}")
    print(f"output      {spec.out_dir}")
    print(f"render pdfs {not args.no_pdfs}")
    print("-" * 60)

    started = time.time()
    manifest = generate_corpus(
        spec,
        gold_dir=REPO_ROOT / "evals" / "datasets",
        write_pdfs=not args.no_pdfs,
    )
    elapsed = time.time() - started

    print(f"deals written     {len(manifest.deals)}")
    print(f"gold facts        {len(manifest.facts)}")
    print(f"manifest sha256   {manifest.sha256()[:16]}...")
    print(f"elapsed           {elapsed:.1f}s")
    print("-" * 60)
    print("defect coverage")
    for defect, deal_ids in sorted(manifest.defect_index.items()):
        print(f"  {defect:<24} {', '.join(deal_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
