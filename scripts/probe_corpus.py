"""Report corpus characteristics that decide the ingestion stack.

Read-only: opens PDFs, never writes to them. Used once per corpus to answer:

  1. Do we need OCR?        -> share of pages with almost no extractable text
  2. Do we need camelot?    -> can PyMuPDF `find_tables` already find them
  3. Are there figures?     -> the multimodal differentiator only exists if yes
  4. How slow is rendering? -> crop cost per page at the target DPI

Usage:
    python scripts/probe_corpus.py data/raw
    python scripts/probe_corpus.py data/raw --json data/processed/corpus_probe.json
    python scripts/probe_corpus.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

# A page with fewer than this many characters is treated as "needs OCR".
TEXT_LESS_CHARS = 50
# A page with at least this many vector paths carries a chart, not table rules.
VECTOR_FIGURE_DRAW_OPS = 15


@dataclass
class PageStats:
    page: int
    chars: int
    text_blocks: int
    images: int
    tables: int
    draw_ops: int


@dataclass
class PdfStats:
    path: str
    name: str
    size_mb: float
    pages: int
    page_stats: list[PageStats] = field(default_factory=list)
    render_ms_per_page: float = 0.0
    error: str | None = None

    @property
    def text_less_pages(self) -> int:
        return sum(1 for p in self.page_stats if p.chars < TEXT_LESS_CHARS)

    @property
    def ocr_share(self) -> float:
        return self.text_less_pages / self.pages if self.pages else 0.0

    @property
    def images(self) -> int:
        return sum(p.images for p in self.page_stats)

    @property
    def tables(self) -> int:
        return sum(p.tables for p in self.page_stats)

    @property
    def pages_with_vector_graphics(self) -> int:
        """Pages carrying enough vector drawing to be a chart, not a rule line."""
        return sum(1 for p in self.page_stats if p.draw_ops >= VECTOR_FIGURE_DRAW_OPS)

    @property
    def pages_with_images(self) -> int:
        return sum(1 for p in self.page_stats if p.images)

    @property
    def total_chars(self) -> int:
        return sum(p.chars for p in self.page_stats)


def probe_pdf(path: Path, render_pages: int = 3, dpi: int = 150) -> PdfStats:
    stat = PdfStats(path=str(path), name=path.name, size_mb=0.0, pages=0)
    try:
        stat.size_mb = round(path.stat().st_size / (1024 * 1024), 2)
        doc = pymupdf.open(path)
    except Exception as exc:  # report, never crash the sweep
        stat.error = f"{type(exc).__name__}: {exc}"
        return stat

    with doc:
        stat.pages = doc.page_count
        for index in range(doc.page_count):
            page = doc[index]
            try:
                tables = len(page.find_tables().tables)
            except Exception:  # find_tables is best-effort
                tables = 0
            stat.page_stats.append(
                PageStats(
                    page=index + 1,
                    chars=len(page.get_text("text").strip()),
                    text_blocks=len(page.get_text("blocks")),
                    images=len(page.get_images(full=True)),
                    tables=tables,
                    draw_ops=len(page.get_drawings()),
                )
            )

        # Render timing on a sample of pages, at the DPI crops will use.
        sample = min(render_pages, doc.page_count)
        if sample:
            start = time.perf_counter()
            for index in range(sample):
                pix = doc[index].get_pixmap(dpi=dpi)
                _ = pix.samples
            stat.render_ms_per_page = round((time.perf_counter() - start) * 1000 / sample, 1)

    return stat


def collect(root: Path, render_pages: int, dpi: int) -> list[PdfStats]:
    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {root}", file=sys.stderr)
    return [probe_pdf(p, render_pages=render_pages, dpi=dpi) for p in pdfs]


def print_report(stats: list[PdfStats]) -> None:
    if not stats:
        return
    header = (
        f"{'document':<34}{'pages':>6}{'chars':>9}{'OCR%':>7}"
        f"{'imgs':>6}{'tbls':>6}{'ms/pg':>7}"
    )
    print(header)
    print("-" * len(header))
    for s in stats:
        if s.error:
            print(f"{s.name[:33]:<34}  ERROR: {s.error}")
            continue
        print(
            f"{s.name[:33]:<34}{s.pages:>6}{s.total_chars:>9}"
            f"{s.ocr_share * 100:>6.0f}%{s.images:>6}{s.tables:>6}"
            f"{s.render_ms_per_page:>7.1f}"
        )

    ok = [s for s in stats if not s.error]
    if not ok:
        return

    total_pages = sum(s.pages for s in ok)
    text_less = sum(s.text_less_pages for s in ok)
    images = sum(s.images for s in ok)
    tables = sum(s.tables for s in ok)
    ocr_share = text_less / total_pages if total_pages else 0.0

    print("\nGates:")
    verdict_ocr = "KEEP PaddleOCR" if ocr_share >= 0.20 else "DROP PaddleOCR from v0"
    print(f"  OCR need        {ocr_share * 100:>5.0f}% text-less pages -> {verdict_ocr}")
    verdict_tables = "keep camelot as fallback" if tables else "no tables detected"
    print(f"  Tables          {tables:>5} detected by PyMuPDF -> {verdict_tables}")
    vector_pages = sum(s.pages_with_vector_graphics for s in ok)
    if images or vector_pages:
        verdict_fig = f"multimodal path justified (raster {images}, vector {vector_pages} pages)"
    else:
        verdict_fig = "NO FIGURES - recheck corpus"
    print(
        f"  Figures         {images:>5} raster images, {vector_pages} vector pages"
        f" -> {verdict_fig}"
    )
    print(f"  Corpus size     {len(ok):>5} PDFs / {total_pages} pages")


def self_test(tmp: Path) -> Path:
    """Build a tiny synthetic PDF so the probe can be validated without a corpus."""
    out = tmp / "fixture_text_table_figure.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "Synthetic fixture: revenue was 12.4M in Q3.", fontsize=12)
    page.insert_text((72, 120), "A second line of extractable body text.", fontsize=12)

    # A ruled grid that find_tables() should recognise as a table.
    x0, y0, w, h = 72, 160, 120, 24
    for row in range(3):
        for col in range(2):
            rect = pymupdf.Rect(x0 + col * w, y0 + row * h, x0 + (col + 1) * w, y0 + (row + 1) * h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
    page.insert_text((x0 + 6, y0 + 16), "Quarter", fontsize=9)
    page.insert_text((x0 + w + 6, y0 + 16), "Revenue", fontsize=9)
    page.insert_text((x0 + 6, y0 + h + 16), "Q3", fontsize=9)
    page.insert_text((x0 + w + 6, y0 + h + 16), "12.4M", fontsize=9)

    # A vector bar chart plus a caption. Enough paths to read as a chart
    # rather than a table rule (the probe separates them by draw-op count).
    axis_y = 360
    page.draw_line(pymupdf.Point(72, axis_y), pymupdf.Point(280, axis_y), width=1.2)
    for grid in range(1, 4):
        y = axis_y - grid * 25
        page.draw_line(pymupdf.Point(72, y), pymupdf.Point(280, y), width=0.4)
    for index, height in enumerate((40, 62, 35, 78, 55, 90, 48, 70, 30, 85, 52, 66)):
        x = 78 + index * 17
        page.draw_rect(
            pymupdf.Rect(x, axis_y - height, x + 12, axis_y),
            color=(0.2, 0.2, 0.6),
            fill=(0.55, 0.6, 0.85),
        )
    page.insert_text((72, 378), "Figure 1: synthetic revenue trend.", fontsize=9)

    # Page 2: no text layer at all -> simulates a scanned page.
    doc.new_page()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    doc.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/raw", help="directory to scan")
    parser.add_argument("--json", dest="json_path", help="write raw stats to this path")
    parser.add_argument("--render-pages", type=int, default=3, help="pages timed per PDF")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--self-test", action="store_true", help="probe a generated fixture")
    args = parser.parse_args()

    if args.self_test:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self_test(Path(tmp))
            stats = collect(fixture.parent, args.render_pages, args.dpi)
    else:
        root = Path(args.root)
        if not root.exists():
            print(f"Path does not exist: {root}", file=sys.stderr)
            return 2
        stats = collect(root, args.render_pages, args.dpi)

    print_report(stats)

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                [{**asdict(s), "ocr_share": round(s.ocr_share, 3)} for s in stats],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
