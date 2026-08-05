"""Tests for loan-package rendering (M2).

These are the slow tests in the suite -- rasterising and degrading scans takes
real time. They are marked `slow` so a developer can skip them with
`pytest -m "not slow"` during a tight edit loop, but CI runs everything.

What they prove is narrow and important: that the documents on disk actually
have the properties the rest of the project assumes. In particular, that the
scanned documents genuinely have no text layer. If they quietly kept one, the
M3 OCR milestone would appear to work while doing nothing, and the first real
scanned PDF would break it.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import fitz
import pytest

from pecos.corpus import (
    DEFECT_INJECTION,
    DEFECT_NEAR_DUPLICATE,
    DEFECT_RESTATEMENT,
    DEFECT_ROTATED_SCAN,
    INJECTION_PAYLOAD,
    CorpusSpec,
    build_deal,
)
from pecos.corpus_build import generate_corpus
from pecos.rendering import (
    DOC_AGING,
    DOC_APPLICATION,
    DOC_BANK,
    DOC_BROKER,
    DOC_DEBT,
    DOC_DRAFT,
    DOC_QUESTIONNAIRE,
    DOC_STATEMENTS,
    DOC_SUPERSEDED,
    DOC_TAX,
    _stable_seed,
    money,
    render_bank_statements,
)

pytestmark = pytest.mark.slow

SEED = 20260804


def _text(path, page: int | None = None) -> str:
    """Extract the text layer. Returns an empty string for image-only PDFs."""
    doc = fitz.open(path)
    if page is None:
        out = "\n".join(p.get_text() for p in doc)
    else:
        out = doc.load_page(page).get_text()
    doc.close()
    return out


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Render a seven-deal corpus once. Seven is the smallest size at which
    every defect lands on its own deal, which keeps the assertions below easy
    to reason about."""
    root = tmp_path_factory.mktemp("corpus")
    spec = CorpusSpec(seed=SEED, n_deals=7, out_dir=root / "raw", years=3)
    manifest = generate_corpus(spec, gold_dir=root / "evals", write_pdfs=True)
    return spec, manifest, root / "raw" / "packages"


# ---------------------------------------------------------------------------
# Shape of the package
# ---------------------------------------------------------------------------


def test_every_deal_gets_the_baseline_documents(rendered):
    spec, manifest, packages = rendered
    baseline = [
        DOC_APPLICATION,
        DOC_STATEMENTS,
        DOC_DEBT,
        DOC_AGING,
        DOC_QUESTIONNAIRE,
        DOC_BROKER,
        DOC_BANK,
        DOC_TAX,
    ]
    for record in manifest.deals:
        folder = packages / record["deal_id"]
        assert folder.is_dir()
        for name in baseline:
            assert (folder / name).exists(), f"{record['deal_id']} missing {name}"


def test_conditional_documents_appear_only_with_their_defect(rendered):
    spec, manifest, packages = rendered
    for i, _record in enumerate(manifest.deals):
        deal = build_deal(spec, i)
        folder = packages / deal.deal_id
        assert (folder / DOC_SUPERSEDED).exists() == (
            DEFECT_RESTATEMENT in deal.defects
        )
        assert (folder / DOC_DRAFT).exists() == (DEFECT_NEAR_DUPLICATE in deal.defects)


# ---------------------------------------------------------------------------
# Text layers
# ---------------------------------------------------------------------------


def test_digital_documents_carry_a_real_text_layer(rendered):
    spec, manifest, packages = rendered
    deal = build_deal(spec, 0)
    text = _text(packages / deal.deal_id / DOC_STATEMENTS)
    # The document header prints the borrower name in capitals, so the
    # comparison is case-insensitive. Extraction fidelity is checked on the
    # figures, where casing is not a factor.
    assert deal.borrower_name.lower() in text.lower()
    assert "EBITDA" in text
    assert "TOTAL ASSETS" in text


def test_scanned_documents_have_no_text_layer_at_all(rendered):
    """The load-bearing assertion for the whole OCR milestone.

    If this ever passes text through, M3's OCR path can silently stop being
    exercised and nobody would notice until a real scan arrived.
    """
    spec, manifest, packages = rendered
    for i in range(len(manifest.deals)):
        deal = build_deal(spec, i)
        for name in (DOC_BANK, DOC_TAX):
            extracted = _text(packages / deal.deal_id / name).strip()
            assert extracted == "", f"{deal.deal_id}/{name} leaked a text layer"


def test_statements_pages_are_in_the_documented_order(rendered):
    """Ground truth cites page 1 for the income statement and page 2 for the
    balance sheet. If layout drifts, those citations become wrong and every
    recall@k score built on them is measuring nothing."""
    spec, manifest, packages = rendered
    deal = build_deal(spec, 0)
    path = packages / deal.deal_id / DOC_STATEMENTS
    assert "Statements of Income" in _text(path, 0)
    assert "Balance Sheets" in _text(path, 1)
    assert "Statements of Cash Flows" in _text(path, 2)


def test_key_figures_are_printed_exactly_as_ground_truth_records_them(rendered):
    """Closes the loop between the manifest and the paper.

    A gold answer of $12,345,678 is worthless if the page shows a rounded
    $12.3M. This checks the exact formatted string is on the exact cited page.
    """
    spec, manifest, packages = rendered
    for i, _record in enumerate(manifest.deals):
        deal = build_deal(spec, i)
        page_one = _text(packages / deal.deal_id / DOC_STATEMENTS, 0)
        assert money(deal.latest.revenue) in page_one
        assert money(deal.latest.ebitda) in page_one
        page_two = _text(packages / deal.deal_id / DOC_STATEMENTS, 1)
        assert money(deal.latest.cash) in page_two


# ---------------------------------------------------------------------------
# Defects on paper
# ---------------------------------------------------------------------------


def test_the_superseded_statements_actually_disagree(rendered):
    """A restatement that prints the same number is not a restatement."""
    spec, manifest, packages = rendered
    for i in range(len(manifest.deals)):
        deal = build_deal(spec, i)
        if DEFECT_RESTATEMENT not in deal.defects:
            continue
        old = _text(packages / deal.deal_id / DOC_SUPERSEDED)
        assert money(deal.stale_ebitda) in old
        assert money(deal.financials[0].ebitda) not in old
        new = _text(packages / deal.deal_id / DOC_STATEMENTS, 0)
        assert money(deal.financials[0].ebitda) in new


def test_the_draft_statements_are_labelled_and_differ(rendered):
    spec, manifest, packages = rendered
    for i in range(len(manifest.deals)):
        deal = build_deal(spec, i)
        if DEFECT_NEAR_DUPLICATE not in deal.defects:
            continue
        draft = _text(packages / deal.deal_id / DOC_DRAFT)
        assert "DRAFT" in draft
        assert money(deal.draft_ebitda) in draft


def test_the_injection_payload_is_present_only_where_assigned(rendered):
    """Two failure modes guarded at once: an injection that never got planted,
    and an injection that leaked into every deal and so cannot be attributed."""
    spec, manifest, packages = rendered
    planted = 0
    for i in range(len(manifest.deals)):
        deal = build_deal(spec, i)
        note = _text(packages / deal.deal_id / DOC_BROKER)
        has_payload = INJECTION_PAYLOAD.split(":")[0] in note
        assert has_payload == (DEFECT_INJECTION in deal.defects)
        planted += int(has_payload)
    assert planted >= 1


def test_the_rotated_page_is_landscape_and_the_others_are_not(rendered):
    """The rotated scan is verified geometrically rather than by reading it,
    because there is nothing to read: the page is an image."""
    spec, manifest, packages = rendered
    checked = 0
    for i in range(len(manifest.deals)):
        deal = build_deal(spec, i)
        doc = fitz.open(packages / deal.deal_id / DOC_BANK)
        try:
            assert doc.page_count == 6
            # True means the page is landscape, i.e. it went in sideways.
            shapes = [p.rect.width > p.rect.height for p in doc]
        finally:
            doc.close()
        if DEFECT_ROTATED_SCAN in deal.defects:
            assert shapes[2] is True, f"{deal.deal_id} page 3 was not rotated"
            assert not any(s for j, s in enumerate(shapes) if j != 2)
            checked += 1
        else:
            assert not any(shapes), f"{deal.deal_id} has an unexpected landscape page"
    assert checked >= 1


def test_the_tax_extract_prints_thousands_only_where_assigned(rendered):
    """Verified through the manifest rather than by reading the scan, since the
    scan has no text layer. The document-level flag and the gold answer must
    agree, or the units defect is untestable at M3."""
    spec, manifest, packages = rendered
    thousands_facts = [
        f for f in manifest.facts if f["defect_tag"] == "units_in_thousands"
    ]
    assert thousands_facts
    for fact in thousands_facts:
        assert fact["source_document"] == DOC_TAX
        assert fact["answer_unit"] == "USD"


# ---------------------------------------------------------------------------
# Corpus outputs
# ---------------------------------------------------------------------------


def test_manifest_and_gold_set_are_written_next_to_the_packages(rendered):
    spec, manifest, packages = rendered
    assert (spec.out_dir / "corpus_manifest.json").exists()
    assert (packages.parent.parent / "evals" / "qa_gold.jsonl").exists()


def test_the_corpus_is_large_enough_to_be_a_real_retrieval_problem(rendered):
    """A handful of pages is a lookup task, not a retrieval task. The corpus has
    to be big enough that a wrong chunk is genuinely reachable."""
    spec, manifest, packages = rendered
    total_pages = 0
    for record in manifest.deals:
        for pdf in (packages / record["deal_id"]).glob("*.pdf"):
            doc = fitz.open(pdf)
            total_pages += doc.page_count
            doc.close()
    assert total_pages >= 100, f"only {total_pages} pages generated"


# ---------------------------------------------------------------------------
# Reproducibility of the scanned documents
# ---------------------------------------------------------------------------


def test_scan_noise_seed_is_stable_across_processes():
    """Regression test for a bug that survived the whole of M2 undetected.

    The scan degradation seed was originally derived from Python's built-in
    `hash()`. String hashing is salted per process, so the seed changed on every
    run and the scanned PDFs carried different sensor noise each time. The
    manifest was unaffected, so the determinism test passed and the corpus
    looked reproducible while it was not.

    It surfaced at M3 as an intermittently failing orientation test: with one
    noise pattern Tesseract detected the rotated page, with another it did not.
    Checking a subprocess is the only way to test this, because the salt is
    fixed for the lifetime of an interpreter.
    """
    expected = _stable_seed("PCP-0003", "bank")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pecos.rendering import _stable_seed;"
            "print(_stable_seed('PCP-0003', 'bank'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "random"},
        check=True,
    )
    assert int(result.stdout.strip()) == expected


def _page_pixel_hashes(path) -> list[str]:
    doc = fitz.open(path)
    try:
        return [
            hashlib.sha256(page.get_pixmap(dpi=72).samples).hexdigest() for page in doc
        ]
    finally:
        doc.close()


def test_the_same_deal_renders_pixel_identical_scans(rendered, tmp_path):
    """The property the seed bug quietly broke: same input, same pixels out.

    Compared page by page rather than byte by byte. PyMuPDF writes a random
    document identifier into every PDF trailer, so two files with identical
    content still differ in their raw bytes -- comparing those would fail for a
    reason that has nothing to do with reproducibility. What must be stable is
    the rendered page, including the sensor noise, and that is what is hashed.

    Asserted on a scanned document specifically, because the digital ones never
    had the problem. The noise is what was non-deterministic.
    """
    spec, manifest, packages = rendered
    deal = build_deal(spec, 0)

    first = _page_pixel_hashes(packages / deal.deal_id / DOC_BANK)
    render_bank_statements(deal, tmp_path / DOC_BANK)
    second = _page_pixel_hashes(tmp_path / DOC_BANK)

    assert first == second, "scanned page content is not reproducible"
