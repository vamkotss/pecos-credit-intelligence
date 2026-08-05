"""Structure-aware chunking with source anchors (M4).

WHAT A CHUNK HAS TO CARRY
-------------------------
M3 produced page records. This module turns them into the units a retriever
indexes. Three things have to survive that transformation, and each one exists
because a specific planted defect punishes losing it.

**1. The anchor back to the page.** A credit memo that states an EBITDA figure
must be able to say which page of which file it came from, and recall@k needs a
page to score against. Every chunk therefore carries deal, document and page --
the same triple the ground-truth manifest uses, so extraction and evaluation
speak the same language with no translation layer.

**2. Table structure.** The customer concentration percentage appears in no
sentence anywhere in the corpus. If a table flattens into running prose, the
number loses its row label and becomes unretrievable no matter how good the
retriever is. Tables are therefore chunked as tables, never split mid-row, and
their header row is repeated when one has to be split.

**3. Document-level status.** This is the one that is easy to get wrong.

THE NEAR-DUPLICATE PROBLEM
--------------------------
One deal carries a DRAFT of its financial statements sitting beside the final
version. The two documents are near-identical -- same borrower, same layout,
same wording, one figure changed. Chunks from them will be almost indistinguishable
to an embedding model, because they *are* almost the same text. Cosine similarity
cannot separate them, and no amount of retriever tuning will fix that, because
the signal simply is not in the text being compared.

What separates them is not content but **status**: one document is final and one
is a draft. That fact lives at the document level, and if chunking drops it at
the boundary -- as naive chunking does, emitting bare strings -- it is gone for
good. So every chunk carries the status and an authority rank derived from it,
and M5's retrieval can prefer authoritative sources when text similarity ties.

The same mechanism handles the restatement defect, where superseded statements
disagree with the comparative set about the same fiscal year.

DESIGN RULE
-----------
Chunking reads page records and nothing else. It never touches the manifest.
Ground truth is used to *score* chunking in tests, never to produce it -- a
chunker that consulted the answer key would measure nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Bumped when chunk boundaries or metadata change. Stored on every chunk, so a
# mixed-version index is detectable rather than silently producing inconsistent
# retrieval scores.
CHUNKER_VERSION = "m4.1"

# --- Size tuning ------------------------------------------------------------
# Characters, not tokens, deliberately. A tokeniser would pin this module to one
# model family and add a dependency for a value that only has to be roughly
# right. English financial prose runs about 4 characters per token, so the
# target below is broadly 220 tokens and the ceiling about 350.
TARGET_CHARS = 900
MAX_CHARS = 1_400
OVERLAP_CHARS = 150

# A PROSE chunk shorter than this is almost always a stray heading or page
# furniture, and indexing it adds noise without adding anything retrievable.
#
# The limit deliberately does not apply to tables. An early version applied it
# to both, and it silently dropped a two-row table rendering to 39 characters --
# which is precisely the content the `table_only_fact` defect punishes losing.
# A short table is still a structured fact; a short line of prose usually is not.
MIN_PROSE_CHARS = 40

# --- Document status --------------------------------------------------------
STATUS_FINAL = "final"
STATUS_DRAFT = "draft"
STATUS_SUPERSEDED = "superseded"

# Authority ranks, used by M5 to break ties that text similarity cannot.
# Final outranks everything. A superseded document sits above a draft because it
# was at least issued to a third party at the time; a draft never was.
AUTHORITY = {STATUS_FINAL: 3, STATUS_SUPERSEDED: 2, STATUS_DRAFT: 1}

# --- Source trust -----------------------------------------------------------
# Who produced the document, which is a distinct question from whether it is the
# current version. A lender weighs a bank-issued statement differently from a
# borrower's own spreadsheet, and differently again from a broker's cover note.
TRUST_BORROWER = "borrower_prepared"
TRUST_BANK = "bank_issued"
TRUST_TAX = "tax_filing"
TRUST_THIRD_PARTY = "third_party"


@dataclass(frozen=True)
class DocumentProfile:
    """What kind of document this is, and how much weight it carries."""

    kind: str
    status: str
    trust: str

    @property
    def authority(self) -> int:
        return AUTHORITY[self.status]


# Filename gives the first guess. It is a starting point, not the answer -- see
# `profile_document` for why the page text gets a vote.
_FILENAME_PROFILES: dict[str, DocumentProfile] = {
    "01_loan_application.pdf": DocumentProfile(
        "loan_application", STATUS_FINAL, TRUST_BORROWER
    ),
    "02_financial_statements_comparative.pdf": DocumentProfile(
        "financial_statements", STATUS_FINAL, TRUST_BORROWER
    ),
    "03_financial_statements_superseded.pdf": DocumentProfile(
        "financial_statements", STATUS_SUPERSEDED, TRUST_BORROWER
    ),
    "04_debt_schedule.pdf": DocumentProfile(
        "debt_schedule", STATUS_FINAL, TRUST_BORROWER
    ),
    "05_ar_aging_and_concentration.pdf": DocumentProfile(
        "receivables", STATUS_FINAL, TRUST_BORROWER
    ),
    "06_borrower_questionnaire.pdf": DocumentProfile(
        "questionnaire", STATUS_FINAL, TRUST_BORROWER
    ),
    "07_broker_email_thread.pdf": DocumentProfile(
        "correspondence", STATUS_FINAL, TRUST_THIRD_PARTY
    ),
    "08_bank_statements.pdf": DocumentProfile(
        "bank_statements", STATUS_FINAL, TRUST_BANK
    ),
    "09_tax_return_extract.pdf": DocumentProfile("tax_return", STATUS_FINAL, TRUST_TAX),
    "10_financial_statements_draft.pdf": DocumentProfile(
        "financial_statements", STATUS_DRAFT, TRUST_BORROWER
    ),
}

_DRAFT_MARKERS = (
    re.compile(r"\bDRAFT\b"),
    re.compile(r"subject\s+to\s+change", re.IGNORECASE),
    re.compile(r"not\s+for\s+(third\s+party\s+)?distribution", re.IGNORECASE),
)
_SUPERSEDED_MARKERS = (
    re.compile(r"\bsupersed", re.IGNORECASE),
    re.compile(r"refer\s+to\s+the\s+comparative", re.IGNORECASE),
    re.compile(r"restated\s+figures", re.IGNORECASE),
    re.compile(r"issued\s+copy", re.IGNORECASE),
)


def profile_document(document: str, page_text: str = "") -> DocumentProfile:
    """Classify a document from its name, corroborated by its own text.

    Filename alone would be brittle in a way worth avoiding. Filenames in this
    corpus are generated and therefore perfectly regular; a real loan package
    arrives with whatever the borrower's accountant happened to save the file
    as, and `Statements FINAL v3 (2).pdf` is not a schema.

    So the text gets a vote, and it can only ever *downgrade* status. A page
    stamped DRAFT is a draft regardless of what the file is called. A page that
    says it was superseded is superseded. Nothing in the text can promote a
    document to final, because a draft that merely fails to say so is still a
    draft, and the safe error is to under-trust rather than over-trust.
    """
    profile = _FILENAME_PROFILES.get(
        document, DocumentProfile("unknown", STATUS_FINAL, TRUST_BORROWER)
    )
    if not page_text:
        return profile

    if any(pattern.search(page_text) for pattern in _DRAFT_MARKERS):
        return DocumentProfile(profile.kind, STATUS_DRAFT, profile.trust)
    if any(pattern.search(page_text) for pattern in _SUPERSEDED_MARKERS):
        if profile.status == STATUS_FINAL:
            return DocumentProfile(profile.kind, STATUS_SUPERSEDED, profile.trust)
    return profile


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit, and everything needed to cite and rank it."""

    chunk_id: str
    deal_id: str
    document: str
    page_number: int
    chunk_index: int
    chunk_type: str  # "prose" or "table"
    text: str
    section: str | None

    # Document-level metadata, carried forward rather than dropped. This is what
    # lets retrieval separate a draft from a final when the text cannot.
    doc_kind: str
    doc_status: str
    authority: int
    source_trust: str

    # Extraction provenance, inherited from the M3 page record.
    extraction_method: str
    mean_word_confidence: float | None
    scale_factor: int
    scale_evidence: str | None
    table_source: str | None = None
    chunker_version: str = CHUNKER_VERSION

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def context_header(self) -> str:
        """A one-line preamble prepended before embedding.

        A bare chunk of numbers is nearly meaningless to an embedding model:
        "Revenue 32,041,248" could come from any document of any vintage. The
        header restores the context a human gets for free from looking at the
        page, and it is what lets a query mentioning "the draft" or "the tax
        return" find its way to the right chunks at all.

        Marked explicitly when a document is not final, so the distinction is
        visible to the embedding as well as to the metadata filter.
        """
        parts = [
            self.document,
            f"page {self.page_number}",
            self.doc_kind.replace("_", " "),
        ]
        if self.section:
            parts.append(self.section)
        if self.doc_status != STATUS_FINAL:
            parts.append(self.doc_status.upper())
        if self.scale_factor != 1:
            parts.append(f"figures in units of {self.scale_factor:,}")
        return " | ".join(parts)

    @property
    def embedding_text(self) -> str:
        return f"{self.context_header}\n{self.text}"

    def to_record(self) -> dict:
        record = asdict(self)
        record["char_count"] = self.char_count
        record["word_count"] = self.word_count
        record["context_header"] = self.context_header
        return record


@dataclass
class ChunkSummary:
    """Corpus-level counts, written beside the chunks."""

    deals: int = 0
    pages: int = 0
    chunks: int = 0
    prose_chunks: int = 0
    table_chunks: int = 0
    draft_chunks: int = 0
    superseded_chunks: int = 0
    scaled_chunks: int = 0
    mean_chunk_chars: float = 0.0
    max_chunk_chars: int = 0
    pages_with_no_chunks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

# Headings a financial package actually uses. Matched case-insensitively against
# whole lines. A curated list beats a general "looks like a title" heuristic
# here: the alternative fires on every short line, and short lines are exactly
# what a table of figures is full of.
_KNOWN_HEADINGS = (
    "Statements of Income",
    "Balance Sheets",
    "Statements of Cash Flows",
    "Schedule of Existing Indebtedness",
    "Ageing summary",
    "Customer detail",
    "Collateral",
    "Commercial Loan Application",
    "Borrower Questionnaire",
    "Accounts Receivable Ageing and Customer Detail",
    "Financial Statements",
    "Correspondence",
    "Commercial Analysis Checking",
    "U.S. CORPORATION INCOME TAX RETURN",
)


def find_heading(line: str) -> str | None:
    """Return the canonical heading a line represents, if any."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    lowered = stripped.lower()
    for heading in _KNOWN_HEADINGS:
        if heading.lower() in lowered:
            return heading
    return None


def _heading_positions(text: str) -> list[tuple[int, str]]:
    """Character offset of every heading occurrence, in document order."""
    positions: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        heading = find_heading(line)
        if heading:
            positions.append((offset, heading))
        offset += len(line)
    return positions


def _section_at(positions: list[tuple[int, str]], offset: int) -> str | None:
    """The most recent heading at or before a character offset."""
    current = None
    for position, heading in positions:
        if position <= offset:
            current = heading
        else:
            break
    return current


# ---------------------------------------------------------------------------
# Table serialisation
# ---------------------------------------------------------------------------


def _normalise_row(cells: list[str]) -> str:
    return " ".join(" ".join(cells).split()).lower()


def render_table_rows(rows: list[list[str]]) -> str:
    """Serialise table rows as a pipe table.

    Pipe format rather than a bare join, because the delimiter is what preserves
    the association between a label and its figure. `Red River Distribution |
    1,803,387 | 34.2%` still reads as three fields after it has passed through an
    embedding model and come back out in a prompt; `Red River Distribution
    1803387 34.2` does not.
    """
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)


def chunk_table(
    rows: list[list[str]], max_chars: int = MAX_CHARS
) -> list[list[list[str]]]:
    """Split a table into row groups that fit, never breaking a row.

    The header row is repeated at the top of every group. Without that, the
    second half of a split table is a wall of unlabelled numbers -- and a wall of
    unlabelled numbers is worse than useless in a lending context, because it
    retrieves on the figures and then cannot say what they measure.
    """
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body:
        return [[header]]

    header_text = render_table_rows([header])
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_chars = len(header_text)

    for row in body:
        row_chars = len(render_table_rows([row])) + 1
        # A single row longer than the ceiling still gets its own group: rows are
        # never split, so the ceiling yields rather than the structure.
        if current and current_chars + row_chars > max_chars:
            groups.append([header, *current])
            current, current_chars = [], len(header_text)
        current.append(row)
        current_chars += row_chars

    if current:
        groups.append([header, *current])
    return groups


# ---------------------------------------------------------------------------
# Prose chunking
# ---------------------------------------------------------------------------


def _pack_lines(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Greedily pack lines into chunks, splitting only at line boundaries.

    Returns (start offset, text) pairs so each chunk can be traced back to its
    position on the page and assigned the right section heading.

    Splitting only on line boundaries is the whole point. A financial statement
    line is an atomic fact -- `EBITDA 2,418,000` -- and a chunker that splits
    mid-line to hit a character target can leave a label in one chunk and its
    figure in another. Both chunks then retrieve badly and neither answers
    anything.
    """
    chunks: list[tuple[int, str]] = []
    current: list[str] = []
    current_start = lines[0][0] if lines else 0
    current_chars = 0

    for offset, line in lines:
        line_chars = len(line) + 1
        if current and current_chars + line_chars > TARGET_CHARS:
            chunks.append((current_start, "\n".join(current)))
            # Overlap: carry back whole trailing lines up to the budget, so a
            # fact that sits near a boundary appears in both neighbours.
            carried: list[str] = []
            carried_chars = 0
            for previous in reversed(current):
                if carried_chars + len(previous) > OVERLAP_CHARS:
                    break
                carried.insert(0, previous)
                carried_chars += len(previous) + 1
            current = carried
            current_chars = carried_chars
            current_start = offset - carried_chars
        if not current:
            current_start = offset
        current.append(line)
        current_chars += line_chars

    if current:
        chunks.append((current_start, "\n".join(current)))
    return chunks


# ---------------------------------------------------------------------------
# Page chunking
# ---------------------------------------------------------------------------


def as_page_record(page: object) -> dict:
    """Accept either an M3 `PageExtraction` or the dict it serialises to.

    Ingestion returns dataclasses; the JSONL on disk holds dicts. Making the
    chunker take both means tests can chunk straight from `ingest_document`
    without a write-and-reread round trip, which is the difference between a
    fast test and a slow one -- and a slow test is one that gets skipped.
    """
    if isinstance(page, dict):
        return page
    to_record = getattr(page, "to_record", None)
    if callable(to_record):
        return to_record()
    raise TypeError(f"cannot read a page record from {type(page).__name__}")


def chunk_page(page: object, profile: DocumentProfile | None = None) -> list[Chunk]:
    """Turn one M3 page record into chunks.

    Tables are emitted first, then whatever prose is left once table content has
    been removed.

    That removal step matters. The extracted page text already contains every
    figure that also appears in the tables, so emitting both unfiltered would
    index the same numbers twice: once with structure and once without. The
    unstructured copy is strictly worse and would compete with the good one in
    retrieval. Lines that reproduce a table row are therefore dropped from the
    prose stream, leaving headings, notes and narrative -- the content tables
    genuinely do not carry.
    """
    page = as_page_record(page)
    text = page.get("text", "")
    tables = page.get("tables", []) or []
    profile = profile or profile_document(page["document"], text)
    headings = _heading_positions(text)

    chunks: list[Chunk] = []
    index = 0

    def make(
        chunk_type: str,
        body: str,
        section: str | None,
        table_source: str | None = None,
    ) -> None:
        nonlocal index
        if not body.strip():
            return
        if chunk_type == "prose" and len(body.strip()) < MIN_PROSE_CHARS:
            return
        chunks.append(
            Chunk(
                chunk_id=(
                    f"{page['deal_id']}::{page['document']}::"
                    f"p{page['page_number']:03d}::c{index:03d}"
                ),
                deal_id=page["deal_id"],
                document=page["document"],
                page_number=page["page_number"],
                chunk_index=index,
                chunk_type=chunk_type,
                text=body,
                section=section,
                doc_kind=profile.kind,
                doc_status=profile.status,
                authority=profile.authority,
                source_trust=profile.trust,
                extraction_method=page["method"],
                mean_word_confidence=page.get("mean_word_confidence"),
                scale_factor=page.get("scale_factor", 1),
                scale_evidence=page.get("scale_evidence"),
                table_source=table_source,
            )
        )
        index += 1

    # --- Tables -----------------------------------------------------------
    table_row_forms: set[str] = set()
    for table in tables:
        rows = [list(row) for row in table["rows"]]
        for row in rows:
            table_row_forms.add(_normalise_row(row))

        # Anchor the table to the heading above where its content appears.
        first_cell = next((c for c in rows[0] if c.strip()), "")
        position = text.find(first_cell) if first_cell else -1
        section = _section_at(headings, position if position >= 0 else 0)

        for group in chunk_table(rows):
            make("table", render_table_rows(group), section, table.get("source"))

    # --- Prose ------------------------------------------------------------
    remaining: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and _normalise_row([stripped]) not in table_row_forms:
            remaining.append((offset, stripped))
        offset += len(line)

    if remaining:
        for start, body in _pack_lines(remaining):
            make("prose", body, _section_at(headings, start))

    return chunks


def chunk_deal(pages: list[object]) -> list[Chunk]:
    """Chunk every page of one deal.

    Document status is resolved once per document by pooling the text of all its
    pages, then applied to every chunk from that document. A DRAFT stamp usually
    appears only on the first page, and it would be wrong for page 2 of a draft
    to be treated as final simply because the stamp was not repeated on it.
    """
    records = [as_page_record(page) for page in pages]
    pooled: dict[str, list[str]] = {}
    for page in records:
        pooled.setdefault(page["document"], []).append(page.get("text", ""))
    profiles = {
        document: profile_document(document, "\n".join(texts))
        for document, texts in pooled.items()
    }

    chunks: list[Chunk] = []
    for page in sorted(records, key=lambda p: (p["document"], p["page_number"])):
        chunks.extend(chunk_page(page, profiles[page["document"]]))
    return chunks


def chunk_corpus(extractions_dir: Path, out_dir: Path) -> ChunkSummary:
    """Chunk every ingested deal, writing one JSONL file per deal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = ChunkSummary()
    sizes: list[int] = []

    for path in sorted(extractions_dir.glob("*.jsonl")):
        pages = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not pages:
            continue
        chunks = chunk_deal(pages)

        produced = {(c.document, c.page_number) for c in chunks}
        for page in pages:
            if (page["document"], page["page_number"]) not in produced:
                summary.pages_with_no_chunks.append(
                    f"{page['deal_id']}/{page['document']}#{page['page_number']}"
                )

        summary.deals += 1
        summary.pages += len(pages)
        summary.chunks += len(chunks)
        for chunk in chunks:
            sizes.append(chunk.char_count)
            if chunk.chunk_type == "table":
                summary.table_chunks += 1
            else:
                summary.prose_chunks += 1
            if chunk.doc_status == STATUS_DRAFT:
                summary.draft_chunks += 1
            if chunk.doc_status == STATUS_SUPERSEDED:
                summary.superseded_chunks += 1
            if chunk.scale_factor != 1:
                summary.scaled_chunks += 1

        lines = [json.dumps(c.to_record(), sort_keys=True) for c in chunks]
        (out_dir / path.name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if sizes:
        summary.mean_chunk_chars = round(sum(sizes) / len(sizes), 1)
        summary.max_chunk_chars = max(sizes)

    (out_dir / "chunk_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def load_chunks(jsonl_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
