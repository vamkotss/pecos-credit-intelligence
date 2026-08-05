# ADR 0001 — Synthetic loan corpus with a ground-truth manifest

**Status:** Accepted · **Date:** 2026-08-04

## Context

The pipeline needs commercial loan packages: scanned financial statements, tax
returns, rent rolls, loan agreements. Three sources were possible.

1. **Real filings** (SEC EDGAR, public company 10-Ks). Free and genuinely messy,
   but wrong shape — audited public filings are clean, structured, and already
   have a text layer. They contain none of the defects that make private-company
   lending documents hard.
2. **A public scanned-document dataset** (FUNSD, DocVQA, CORD). Real scan noise,
   but the documents are receipts and forms, not multi-year financial statements,
   and the annotations do not include the financial semantics this project needs.
3. **A seeded generator** producing synthetic packages plus a manifest recording
   every planted figure, its page, its fiscal year, and its defect class.

## Decision

Build the generator (option 3).

## Rationale

The decisive argument is not realism, it is **knowable ground truth**. This
project's headline claim is "zero ungrounded figures." That claim is only
testable if something outside the pipeline already knows the right answer for
every number on every page. With real documents, verifying a single memo means
a human reading 400 pages; with 40 packages that is not a portfolio project,
it is a job.

A generator also lets defects be planted at *controlled rates*, so an evaluation
can report "recall on rotated pages" separately from "recall on clean pages" —
which is a far more interesting result than a single blended score.

The same reasoning drove P2's seeded defect generator and P4's three planted
leakage traps. This is the third application of a pattern that has held up.

## Consequences

**Accepted cost.** A reviewer may say "synthetic data is easy mode." The README
answers this directly: the generator renders to real PDFs, rasterises them,
degrades them with scan artefacts, and rotates a fraction of pages, so the
ingestion pipeline sees genuine image noise and genuine OCR failure — not a
convenient text layer. The *documents* are synthetic; the *difficulty* is not.

**Accepted risk.** Generated documents can be unrealistically uniform, which
would flatter the retriever. Mitigation: the generator randomises layout family,
statement format, terminology, and entity naming per borrower, and M3 ships an
extraction quality report showing OCR character error rate is non-trivial.

**Rejected shortcut.** Not generating a manifest and instead labelling answers by
hand. That produces perhaps 30 labelled questions; the manifest produces
thousands, and regenerates for free when the corpus changes.
