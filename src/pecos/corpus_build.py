"""Corpus orchestration: write packages, manifest, and gold eval set (M2).

Kept in its own module so `corpus.py` stays free of any PDF dependency. That
separation is not cosmetic -- it means the financial engine and the ground-truth
logic can be tested without ReportLab, PyMuPDF or Pillow installed, so the
fastest and most important tests in the suite have almost no dependency surface.

Three artefacts come out of a run:

  data/raw/packages/<deal_id>/*.pdf   the loan packages themselves
  data/raw/corpus_manifest.json       every deal, every fact, the defect index
  evals/datasets/qa_gold.jsonl        the same facts as a scorable eval set

The PDFs are gitignored -- they are regenerable from the seed, and committing
hundreds of megabytes of synthetic scans to a portfolio repository would be a
mistake a reviewer would notice. The manifest and the gold set ARE committed,
because they are small, they are the contract every later milestone codes
against, and a reviewer can read them without running anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from pecos.corpus import (
    ALL_DEFECTS,
    CorpusManifest,
    CorpusSpec,
    Deal,
    build_deal,
    build_gold_facts,
    summarise_deal,
)
from pecos.rendering import render_package

PACKAGES_SUBDIR = "packages"
MANIFEST_NAME = "corpus_manifest.json"
GOLD_NAME = "qa_gold.jsonl"


def generate_corpus(
    spec: CorpusSpec,
    gold_dir: Path | None = None,
    write_pdfs: bool = True,
) -> CorpusManifest:
    """Generate the whole corpus and return the manifest.

    Parameters
    ----------
    spec:
        What to generate. See `CorpusSpec`.
    gold_dir:
        Where `qa_gold.jsonl` is written. Defaults to `<out_dir>/../evals/datasets`
        only when it can be resolved; callers normally pass it explicitly.
    write_pdfs:
        When False, deals and ground truth are built but no PDF is rendered.
        Tests that only care about the financial identities use this, which cuts
        their runtime from minutes to milliseconds.
    """
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    packages_root = spec.out_dir / PACKAGES_SUBDIR

    manifest = CorpusManifest(seed=spec.seed, n_deals=spec.n_deals, years=spec.years)
    defect_index: dict[str, list[str]] = {d: [] for d in ALL_DEFECTS}

    for i in range(spec.n_deals):
        deal: Deal = build_deal(spec, i)

        if write_pdfs:
            page_index = render_package(deal, packages_root / deal.deal_id)
        else:
            # Without rendering there is no measured page index, so fall back to
            # the known fixed layout of the statements document. Ground truth is
            # still internally consistent; it is simply not verified against
            # actual output. Any test that asserts on page numbers must render.
            page_index = {
                "02_financial_statements_comparative.pdf": {
                    "income_statement": 1,
                    "balance_sheet": 2,
                    "cash_flow": 3,
                }
            }

        facts = build_gold_facts(deal, page_index)

        manifest.deals.append(summarise_deal(deal))
        manifest.facts.extend(
            {
                "fact_id": f.fact_id,
                "deal_id": f.deal_id,
                "question": f.question,
                "answer_value": f.answer_value,
                "answer_unit": f.answer_unit,
                "answer_text": f.answer_text,
                "fact_type": f.fact_type,
                "source_document": f.source_document,
                "source_page": f.source_page,
                "answerable": f.answerable,
                "defect_tag": f.defect_tag,
                "notes": f.notes,
            }
            for f in facts
        )
        for d in deal.defects:
            defect_index[d].append(deal.deal_id)

    manifest.defect_index = defect_index

    (spec.out_dir / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")

    if gold_dir is not None:
        gold_dir.mkdir(parents=True, exist_ok=True)
        # JSON Lines rather than one big JSON array: eval harnesses stream it,
        # and a single changed question produces a one-line diff in code review
        # instead of a reformatted file.
        lines = [json.dumps(f, sort_keys=True) for f in manifest.facts]
        (gold_dir / GOLD_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return manifest
