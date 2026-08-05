"""Document ingestion: OCR, layout, tables and page provenance (M3).

WHAT THIS MODULE IS FOR
-----------------------
A loan package is a folder of PDFs of wildly different quality. Some were
exported from accounting software and carry a clean text layer. Some were
printed, signed, scanned on a desk scanner and emailed back, and contain no
text at all -- only pixels.

Everything downstream of here assumes it is working with text. This module is
the boundary where that assumption is made true, and where the cost of making
it true is recorded honestly.

THE THREE JOBS
--------------
**1. Route, do not guess.** Each page is inspected for a text layer. Pages that
have one are parsed directly. Pages that do not are rendered, OCR'd, and marked
as such. Running OCR on a digital page is slower and strictly worse -- OCR
introduces errors a clean text layer does not have. Running direct extraction on
a scanned page silently returns nothing at all, which is worse still, because it
looks like a document with no content rather than a failure.

**2. Correct orientation before reading.** A page that went through the feeder
sideways produces, verbatim from Tesseract, output like
`") S89U9INDDO Spun} JUaIONJNSU]"`. That is not low-quality text. It is noise
that will be embedded, indexed and retrieved as though it meant something. The
planted `rotated_scanned_page` defect exists to make sure this path is exercised.

**3. Keep provenance on every page.** Each record carries the deal, the document,
the page number, which method read it, what rotation was applied, and how
confident the OCR was. A credit memo that cites a figure has to be able to say
which page of which file it came from, and a retrieval eval scored on recall@k
needs the page number to score against.

PAGE IS THE UNIT
----------------
Not the document, not the chunk. A document is too coarse to cite -- "it's in
the financial statements" is not an answer an analyst accepts. A chunk is too
fine and too unstable, because chunk boundaries change whenever the chunking
strategy changes at M4, and every stored citation would break with them.

The page is the natural unit: it is what a human points at, it is stable across
every downstream design change, and it is what the ground-truth manifest already
records.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz  # PyMuPDF -- page rendering and text-layer inspection
import pdfplumber  # digital-native text and table extraction
import pytesseract  # Tesseract OCR bindings
from PIL import Image

# Bumped whenever extraction behaviour changes in a way that would alter output.
# Stored on every page record, so a mixed-version interim directory is
# detectable instead of silently producing inconsistent evals.
EXTRACTOR_VERSION = "m3.1"

# --- Tuning knobs -----------------------------------------------------------
# A page with fewer than this many characters in its text layer is treated as
# scanned. It is not zero because some image-only PDFs carry a stray character
# or two from a stamp or a header, and a strict `== 0` test would route those
# pages to the digital path and return almost nothing.
DIGITAL_TEXT_THRESHOLD = 40

OSD_DPI = 150  # cheap render, only used to decide orientation
OCR_DPI = 200  # quality render, used for the actual read

# Orientation quality gate. See `_orientation_score` for what the number means
# and how it was calibrated. Measured values on this corpus: about 50 for a
# correctly-oriented page, and under 1 for the same page read sideways, so the
# threshold sits in a very wide gap rather than on a knife edge.
MIN_ORIENTATION_SCORE = 20.0
MIN_WORDS_FOR_SCORING = 8

# Words closer together than this fraction of the page width belong to the same
# cell; a wider gap is a column boundary.
COLUMN_GAP_RATIO = 0.035

# A run of at least this many multi-cell lines is treated as a table.
MIN_TABLE_ROWS = 3

# --- Scale detection --------------------------------------------------------
# Matched against page text to catch statements presented in thousands. The
# patterns are deliberately loose because this note is usually set in small type
# on a scanned page, which is exactly where OCR is least reliable -- a strict
# pattern would miss it precisely when it matters most.
_THOUSANDS_PATTERNS = (
    re.compile(r"in\s+thousands", re.IGNORECASE),
    re.compile(r"\(\s*000\s*s?\s*\)", re.IGNORECASE),
    re.compile(r"\$\s*000s", re.IGNORECASE),
    # OCR-tolerant catch-all. Tesseract reads the small type this note is set in
    # as things like "amounts stated 1n thousands", which the strict patterns
    # above miss -- precisely on the scanned pages where the note matters most.
    # The leading keyword is what keeps it from firing on ordinary prose such as
    # "the company serves thousands of customers".
    re.compile(
        r"(?:amounts?|stated|expressed|figures?|dollars?|\$)[^.\n]{0,30}thousands",
        re.IGNORECASE,
    ),
)
_MILLIONS_PATTERNS = (
    re.compile(r"in\s+millions", re.IGNORECASE),
    re.compile(r"\$\s*mm\b", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    """One OCR'd word with its box and confidence, in pixel coordinates.

    `line_id` is Tesseract's own (block, paragraph, line) grouping when the word
    came from OCR, and None when it came from a PDF text layer. It matters
    because Tesseract's layout analysis handles page skew properly, and the
    geometric fallback used for PDF words does not do so as well -- see
    `_group_into_lines`.
    """

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    line_id: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class ExtractedTable:
    """A table recovered from a page.

    `source` records how it was found, because the two methods have very
    different reliability. `pdfplumber` reads real ruling lines and text
    positions from the PDF and is close to exact. `layout_clustering` infers
    structure from OCR word boxes and is a best effort. Downstream code that
    treats them as equally trustworthy will be wrong about scanned tables.
    """

    rows: tuple[tuple[str, ...], ...]
    source: str

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def cells(self) -> list[str]:
        return [c for row in self.rows for c in row if c]


@dataclass(frozen=True)
class PageExtraction:
    """One page, and everything known about how it was read."""

    deal_id: str
    document: str
    page_number: int  # 1-based, matching the ground-truth manifest
    method: str  # "digital" or "ocr"
    rotation_applied: int  # degrees clockwise, 0 / 90 / 180 / 270
    text: str
    word_count: int
    mean_word_confidence: float | None  # None for digital pages
    tables: tuple[ExtractedTable, ...]
    scale_factor: int  # 1, 1_000 or 1_000_000
    scale_evidence: str | None
    extractor_version: str = EXTRACTOR_VERSION

    def to_record(self) -> dict:
        d = asdict(self)
        d["tables"] = [
            {"rows": [list(r) for r in t.rows], "source": t.source} for t in self.tables
        ]
        return d


@dataclass
class IngestSummary:
    """Corpus-level counts, written alongside the extractions.

    Exists so a regression is visible without reading 200 JSON files. If the
    OCR page count drops to zero because Tesseract is missing from a machine,
    that shows up here as a number, not as a mysteriously empty index at M5.
    """

    deals: int = 0
    documents: int = 0
    pages: int = 0
    digital_pages: int = 0
    ocr_pages: int = 0
    rotated_pages: int = 0
    scaled_pages: int = 0
    tables: int = 0
    empty_pages: list[str] = field(default_factory=list)
    mean_ocr_confidence: float | None = None


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def page_has_text_layer(page: fitz.Page) -> bool:
    """Decide whether a page can be read without OCR.

    This is a cheap check -- it reads what the PDF already contains and does not
    render anything -- which is why it is the first thing done to every page.
    """
    return len(page.get_text().strip()) >= DIGITAL_TEXT_THRESHOLD


# ---------------------------------------------------------------------------
# Digital path
# ---------------------------------------------------------------------------


def _clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def _plumber_words(page: pdfplumber.page.Page) -> list[Word]:
    """Convert pdfplumber word boxes into the same `Word` type OCR produces.

    This is what lets one layout algorithm serve both paths. Confidence is set
    to 100 because these positions come from the PDF itself: there is no
    recognition step and therefore nothing to be uncertain about.
    """
    out: list[Word] = []
    for word in page.extract_words():
        left, top = int(word["x0"]), int(word["top"])
        out.append(
            Word(
                text=word["text"],
                left=left,
                top=top,
                width=int(word["x1"]) - left,
                height=int(word["bottom"]) - top,
                confidence=100.0,
            )
        )
    return out


def extract_digital_page(
    page: pdfplumber.page.Page,
) -> tuple[str, list[ExtractedTable]]:
    """Read a page that already has a text layer.

    Tables are extracted separately from the running text rather than being left
    to flatten into it. That separation is the whole reason the
    `table_only_fact` defect is survivable: the customer concentration figure
    exists in no sentence anywhere in the corpus, so if tables collapse into
    prose the number loses its relationship to its row label and the fact
    becomes unretrievable.

    Two strategies, tried in order:

    **Ruling lines first.** When a table is drawn with real borders, pdfplumber
    reads those borders and recovers the grid essentially exactly.

    **Layout clustering second.** Borrower-prepared financial statements are
    usually set without borders -- columns are separated by whitespace and
    nothing else, which is why the ruling-line strategy finds zero tables on
    them. The fallback runs the identical whitespace-clustering algorithm used
    on OCR output, fed with pdfplumber's exact word boxes instead of Tesseract's
    estimated ones.

    pdfplumber's own text-based strategy was tried and rejected: on this corpus
    it split words mid-token, turning "TRINITY BEND" into cells like `RINIT` and
    `Y BEND`. One shared clustering implementation is also less to maintain and
    less to explain than two.
    """
    text = page.extract_text() or ""

    tables: list[ExtractedTable] = []
    for raw in page.extract_tables():
        rows = tuple(
            tuple(_clean_cell(cell) for cell in row) for row in raw if row is not None
        )
        # A one-row "table" is almost always a misfire on a heading rule.
        if len(rows) >= 2:
            tables.append(ExtractedTable(rows=rows, source="pdfplumber_lines"))

    if not tables:
        tables = cluster_words_into_tables(_plumber_words(page), int(page.width))

    return text, tables


# ---------------------------------------------------------------------------
# OCR path
# ---------------------------------------------------------------------------


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    """Rasterise a PDF page to a greyscale image at a given resolution."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")


def propose_rotation(image: Image.Image) -> int:
    """Ask Tesseract which way up the page is.

    Run against a deliberately cheap render, because orientation is a
    four-way choice and does not need detail to get right -- a full-resolution
    render would roughly double the cost of every scanned page for no gain.

    Returns degrees to rotate clockwise. Returns 0 if detection fails, which it
    does on genuinely sparse pages; the confidence check in `ocr_page` is what
    catches that case.
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return int(osd.get("rotate", 0)) % 360
    except Exception:
        # Tesseract raises on pages with too little text to judge. That is not
        # an error worth failing ingestion over -- it is a page we read as-is.
        return 0


def _rotate(image: Image.Image, degrees: int) -> Image.Image:
    if degrees % 360 == 0:
        return image
    # PIL rotates counter-clockwise for positive angles, Tesseract reports the
    # clockwise correction, hence the negation. Getting this sign wrong produces
    # a page that is upside down instead of upright, and the resulting text is
    # confident nonsense rather than an obvious failure.
    return image.rotate(-degrees, expand=True, fillcolor=235)


def _ocr_once(image: Image.Image) -> tuple[str, list[Word], float]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: list[Word] = []
    for i, raw in enumerate(data["text"]):
        token = raw.strip()
        if not token:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        words.append(
            Word(
                text=token,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                confidence=conf,
                line_id=(
                    int(data["block_num"][i]),
                    int(data["par_num"][i]),
                    int(data["line_num"][i]),
                ),
            )
        )
    mean_conf = statistics.mean(w.confidence for w in words) if words else 0.0
    text = _words_to_text(words)
    return text, words, mean_conf


def _words_to_text(words: list[Word]) -> str:
    """Reassemble words into lines using their vertical positions.

    Tesseract's plain-text output already does this, but it is rebuilt here from
    the same word boxes used for table clustering, so text and tables can never
    disagree about what is on the page.
    """
    return "\n".join(
        " ".join(w.text for w in line) for line in _group_into_lines(words)
    )


def _overlap_ratio(a: Word, b: Word) -> float:
    """How much two word boxes overlap vertically, as a fraction of the shorter.

    Overlap is used instead of comparing top coordinates against a fixed
    tolerance because it is scale-free: it asks whether two boxes sit at the
    same height, not whether they begin at the same pixel. That distinction
    matters at every font size on the page.
    """
    top = max(a.top, b.top)
    bottom = min(a.top + a.height, b.top + b.height)
    shorter = min(a.height, b.height)
    if shorter <= 0:
        return 0.0
    return max(0.0, bottom - top) / shorter


def _group_into_lines(words: list[Word]) -> list[list[Word]]:
    """Group words into reading-order lines, tolerant of skew.

    Words are walked **left to right**, and each one joins whichever open line
    its nearest left-hand neighbour sits on. That ordering is the crux.

    The obvious implementation -- sort by vertical position, start a new line
    whenever the top coordinate jumps -- fails on scanned pages. The scans in
    this corpus carry up to 0.7 degrees of skew, and across a 2,000-pixel line
    that is roughly 24 pixels of vertical drift: more than the gap between
    lines. Sorting by top interleaves the end of one line with the start of the
    next, every line fragments into single-word rows, and every table on the
    page vanishes. That is exactly what happened to the rotated bank statement
    page, where the skew runs along the long axis.

    Comparing each word only to its immediate left-hand neighbour makes the
    comparison local. Adjacent words drift by a pixel or two no matter how
    skewed the page is overall, so the grouping tracks the baseline instead of
    fighting it.
    """
    if not words:
        return []

    # Prefer the OCR engine's own line grouping when it is available.
    #
    # Tesseract runs real layout analysis, including skew estimation, before it
    # assigns words to lines. The geometric fallback below cannot match that,
    # and the gap showed up as a live bug: on a bank statement page with a wide
    # label-to-amount column gap and a large skew, the geometric grouping split
    # every row in two. Median line span collapsed, the orientation score fell
    # from 75 to 9, and the page was "corrected" into being sideways.
    #
    # The fallback still runs for pdfplumber words, which carry no line ids --
    # and digital pages have no skew for it to mishandle.
    if all(word.line_id is not None for word in words):
        grouped: dict[tuple[int, int, int], list[Word]] = {}
        for word in words:
            grouped.setdefault(word.line_id, []).append(word)  # type: ignore[arg-type]
        lines = [sorted(line, key=lambda w: w.left) for line in grouped.values()]
        lines.sort(key=lambda line: min(w.top for w in line))
        return lines

    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.left, w.top)):
        # Each line's last element is its rightmost word, because words arrive
        # in left-to-right order.
        best_line, best_overlap = None, 0.0
        for line in lines:
            overlap = _overlap_ratio(line[-1], word)
            if overlap > best_overlap:
                best_line, best_overlap = line, overlap
        if best_line is not None and best_overlap > 0.4:
            best_line.append(word)
        else:
            lines.append([word])

    # Restore reading order: top to bottom, and left to right within each line.
    lines.sort(key=lambda line: min(w.top for w in line))
    return lines


def _orientation_score(words: list[Word], page_width: int) -> float:
    """How much a page looks like text that is the right way up.

    The obvious signal -- mean OCR confidence -- is not sufficient on its own,
    and the reason is worth spelling out because it cost a debugging session.
    Tesseract reads a sideways page as a column of single words and reports
    roughly 95% confidence on each of them. It is confident because it really
    did recognise those characters; it simply has no idea they were meant to be
    read across the page rather than down it. Confidence answers "did I read
    these glyphs correctly", not "was this the right way to read the page".

    The signal that does separate the two cases is **how far a line runs**. On a
    correctly-oriented page a line of text spans most of the page width. On the
    same page read sideways, each "line" is one stacked word and spans almost
    nothing. Measured on the rotated bank statement in this corpus: a median
    line span of 1,344 pixels upright against 19 pixels sideways, a difference
    of seventy times.

    Score is mean confidence multiplied by median line span as a fraction of
    page width, which produces about 50 for a good page and under 1 for a bad
    one.
    """
    if len(words) < MIN_WORDS_FOR_SCORING or page_width <= 0:
        return 0.0
    lines = _group_into_lines(words)
    spans = [
        max(w.left + w.width for w in line) - min(w.left for w in line)
        for line in lines
    ]
    if not spans:
        return 0.0
    coverage = statistics.median(spans) / page_width
    mean_conf = statistics.mean(w.confidence for w in words)
    return mean_conf * coverage


def _read_at(image: Image.Image, degrees: int) -> tuple[str, list[Word], float, float]:
    """OCR one candidate orientation and score it."""
    rotated = _rotate(image, degrees)
    text, words, conf = _ocr_once(rotated)
    return text, words, conf, _orientation_score(words, rotated.width)


def ocr_page(image: Image.Image) -> tuple[str, list[Word], float, int]:
    """OCR a page, correcting its orientation first.

    The strategy is **propose, then compare -- never merely accept**.

    Tesseract's orientation detection is fast and usually right, but it is not
    reliable enough to trust blind. Its own confidence score ran from 0.4 to 9.4
    on pages of this corpus where the answer was correct, so the score carries
    almost no information, and detection also proved sensitive to scan noise:
    the same page with a different noise pattern was sometimes detected and
    sometimes not.

    An earlier version accepted the proposal whenever it cleared a quality
    threshold. That was wrong in a way worth recording, because it looked
    reasonable and produced six silently mis-rotated pages. A page proposed at
    180 degrees scored 26 -- above the threshold -- and was accepted without the
    upright reading, which would have scored 75, ever being computed. A
    threshold answers "is this acceptable"; the question that matters is "is
    this the best available".

    So upright is always evaluated alongside whatever Tesseract proposed, and
    the better of the two wins. Only if both look poor are the remaining two
    orientations tried. The common case costs a single OCR pass.

    Returns (text, words, mean confidence, rotation applied clockwise).
    """
    # Orientation is a four-way choice and needs no detail to get right, so the
    # proposal is made against a half-scale render.
    thumbnail = image.resize((max(1, image.width // 2), max(1, image.height // 2)))
    proposed = propose_rotation(thumbnail)

    results: dict[int, tuple[str, list[Word], float, float]] = {
        proposed: _read_at(image, proposed)
    }
    if proposed != 0:
        results[0] = _read_at(image, 0)

    best_rotation = max(results, key=lambda deg: results[deg][3])
    if results[best_rotation][3] < MIN_ORIENTATION_SCORE:
        # Neither candidate looks like upright text. Try what is left.
        for candidate in (90, 180, 270):
            if candidate not in results:
                results[candidate] = _read_at(image, candidate)
        best_rotation = max(results, key=lambda deg: results[deg][3])

    text, words, conf, _ = results[best_rotation]
    return text, words, conf, best_rotation


def cluster_words_into_tables(
    words: list[Word], page_width: int
) -> list[ExtractedTable]:
    """Recover table structure from OCR word boxes.

    There is no ruling-line information to work with on a scan, so structure is
    inferred from whitespace: words on the same line are grouped, and a
    horizontal gap wider than a threshold is treated as a column boundary. Runs
    of consecutive multi-column lines become a table.

    This is genuinely approximate and is labelled `layout_clustering` in the
    output so nothing downstream mistakes it for the exact structure pdfplumber
    recovers from a digital page.
    """
    if not words:
        return []

    gap_threshold = page_width * COLUMN_GAP_RATIO

    rows: list[tuple[str, ...] | None] = []
    for ordered in _group_into_lines(words):
        cells: list[list[str]] = [[ordered[0].text]]
        # Pairwise walk over adjacent words. Both slices are trimmed to the same
        # length so `strict=True` stays meaningful -- it is guarding against a
        # future edit that breaks the pairing, not against the offset itself.
        for prev, word in zip(ordered[:-1], ordered[1:], strict=True):
            gap = word.left - (prev.left + prev.width)
            if gap > gap_threshold:
                cells.append([word.text])
            else:
                cells[-1].append(word.text)
        joined = tuple(" ".join(c) for c in cells)
        rows.append(joined if len(joined) >= 2 else None)

    tables: list[ExtractedTable] = []
    run: list[tuple[str, ...]] = []
    for row in rows + [None]:  # trailing None flushes the final run
        if row is None:
            if len(run) >= MIN_TABLE_ROWS:
                tables.append(
                    ExtractedTable(rows=tuple(run), source="layout_clustering")
                )
            run = []
        else:
            run.append(row)
    return tables


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------


def detect_scale(text: str) -> tuple[int, str | None]:
    """Find a units note and return the multiplier it implies.

    This is the defence against the planted `units_in_thousands` defect, and it
    is worth being precise about why it matters. A figure read as 32,041 when
    the page means $32,041,000 is not a small error. It is not flagged by any
    confidence score, it does not look wrong, and it changes a lending decision
    by three orders of magnitude. Of all the failure modes in this corpus it is
    the one most likely to reach a credit committee undetected.

    Returns (multiplier, the matched note) so the evidence travels with the
    page and a reviewer can check the call rather than trust it.
    """
    for pattern in _THOUSANDS_PATTERNS:
        match = pattern.search(text)
        if match:
            return 1_000, _context(text, match.start(), match.end())
    for pattern in _MILLIONS_PATTERNS:
        match = pattern.search(text)
        if match:
            return 1_000_000, _context(text, match.start(), match.end())
    return 1, None


def _context(text: str, start: int, end: int, window: int = 45) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return " ".join(text[lo:hi].split())


# ---------------------------------------------------------------------------
# Document and package ingestion
# ---------------------------------------------------------------------------


def ingest_document(pdf_path: Path, deal_id: str) -> list[PageExtraction]:
    """Ingest one PDF, page by page, routing each page independently.

    Routing is per page rather than per document on purpose. A real loan package
    contains files where an accountant exported ten clean pages and then appended
    two scanned signature sheets. Classifying at the document level would send
    every page of such a file down one wrong path.
    """
    document = pdf_path.name
    pages: list[PageExtraction] = []

    fitz_doc = fitz.open(pdf_path)
    try:
        plumber_doc = pdfplumber.open(str(pdf_path))
        try:
            for index in range(fitz_doc.page_count):
                fitz_page = fitz_doc.load_page(index)

                if page_has_text_layer(fitz_page):
                    text, tables = extract_digital_page(plumber_doc.pages[index])
                    method, rotation, mean_conf = "digital", 0, None
                    word_count = len(text.split())
                else:
                    image = render_page(fitz_page, OCR_DPI)
                    text, words, mean_conf, rotation = ocr_page(image)
                    tables = cluster_words_into_tables(words, image.width)
                    method, word_count = "ocr", len(words)

                scale_factor, scale_evidence = detect_scale(text)

                pages.append(
                    PageExtraction(
                        deal_id=deal_id,
                        document=document,
                        page_number=index + 1,
                        method=method,
                        rotation_applied=rotation,
                        text=text,
                        word_count=word_count,
                        mean_word_confidence=(
                            round(mean_conf, 2) if mean_conf is not None else None
                        ),
                        tables=tuple(tables),
                        scale_factor=scale_factor,
                        scale_evidence=scale_evidence,
                    )
                )
        finally:
            plumber_doc.close()
    finally:
        fitz_doc.close()

    return pages


def ingest_package(package_dir: Path, deal_id: str) -> list[PageExtraction]:
    """Ingest every PDF in one deal folder, in filename order.

    Sorted so the output is stable: an unordered directory listing would make
    the interim files differ between machines for no reason and destroy the
    determinism check.
    """
    pages: list[PageExtraction] = []
    for pdf_path in sorted(package_dir.glob("*.pdf")):
        pages.extend(ingest_document(pdf_path, deal_id))
    return pages


def ingest_corpus(
    packages_root: Path,
    out_dir: Path,
    deal_ids: list[str] | None = None,
) -> IngestSummary:
    """Ingest the whole corpus and write one JSONL file per deal.

    One file per deal rather than one for the corpus, because M5 indexes and
    M7 answers are both scoped to a single deal -- a lending question is always
    about one borrower. Per-deal files mean a single deal can be re-ingested
    after a fix without touching anything else.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in packages_root.iterdir() if p.is_dir())
    if deal_ids is not None:
        wanted = set(deal_ids)
        folders = [f for f in folders if f.name in wanted]

    summary = IngestSummary()
    confidences: list[float] = []

    for folder in folders:
        pages = ingest_package(folder, folder.name)
        documents = {p.document for p in pages}

        summary.deals += 1
        summary.documents += len(documents)
        summary.pages += len(pages)
        for page in pages:
            if page.method == "digital":
                summary.digital_pages += 1
            else:
                summary.ocr_pages += 1
                if page.mean_word_confidence is not None:
                    confidences.append(page.mean_word_confidence)
            if page.rotation_applied:
                summary.rotated_pages += 1
            if page.scale_factor != 1:
                summary.scaled_pages += 1
            summary.tables += len(page.tables)
            if not page.text.strip():
                summary.empty_pages.append(
                    f"{page.deal_id}/{page.document}#{page.page_number}"
                )

        lines = [json.dumps(p.to_record(), sort_keys=True) for p in pages]
        (out_dir / f"{folder.name}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    summary.mean_ocr_confidence = (
        round(statistics.mean(confidences), 2) if confidences else None
    )
    (out_dir / "ingest_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def load_pages(jsonl_path: Path) -> list[dict]:
    """Read back a deal's extractions. Used by tests and by M4."""
    return [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
