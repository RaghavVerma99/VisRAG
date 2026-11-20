"""Tests for `scripts/probe_corpus.py`.

The probe decides the ingestion stack, so it has to be right about the corpus
before any parsing code is written. These tests pin its behaviour on a known
synthetic document.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType


def test_fixture_is_well_formed(probe_corpus: ModuleType, fixture_pdf: Path) -> None:
    assert fixture_pdf.exists()
    stats = probe_corpus.probe_pdf(fixture_pdf)

    assert stats.error is None
    assert stats.pages == 2
    assert stats.total_chars > 0
    assert [p.page for p in stats.page_stats] == [1, 2]


def test_text_less_page_trips_the_ocr_gate(probe_corpus: ModuleType, fixture_pdf: Path) -> None:
    """Page 2 has no text layer, so exactly one page must count as OCR-needed."""
    stats = probe_corpus.probe_pdf(fixture_pdf)

    assert stats.text_less_pages == 1
    assert stats.ocr_share == 0.5


def test_table_and_figure_regions_are_detected(
    probe_corpus: ModuleType, fixture_pdf: Path
) -> None:
    stats = probe_corpus.probe_pdf(fixture_pdf)
    page_one = stats.page_stats[0]

    # A ruled grid and a drawn "chart" both leave vector drawing operations.
    assert page_one.draw_ops > 0
    assert page_one.chars > 0


def test_probe_is_read_only(probe_corpus: ModuleType, fixture_pdf: Path) -> None:
    """Ingestion must never modify source PDFs (root AGENTS.md §7)."""
    before = fixture_pdf.read_bytes()
    probe_corpus.probe_pdf(fixture_pdf)
    assert fixture_pdf.read_bytes() == before


def test_missing_file_reports_error_instead_of_raising(
    probe_corpus: ModuleType, tmp_path: Path
) -> None:
    stats = probe_corpus.probe_pdf(tmp_path / "does-not-exist.pdf")

    assert stats.error is not None
    assert stats.pages == 0
    assert stats.ocr_share == 0.0
