"""Ingestion entry point: a PDF path becomes a list of Elements.

Read-only with respect to the source PDF. The four stages are deliberately
separate modules so each can be tested and replaced independently:

    tables  -> PyMuPDF find_tables, filtered for over-detection
    figures -> vector drawings + raster images, clustered into regions
    layout  -> body text and caption pairing
    crops   -> PNGs rendered from bboxes into data/processed/<doc_id>/

Re-ingesting the same PDF produces the same ids and overwrites the same crop
files (ADR-012 rule 7).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import pymupdf

from src.config import get_settings
from src.ingestion.crops import save_crops
from src.ingestion.elements import Element
from src.ingestion.layout import (
    derive_doc_id,
    detect_figures,
    extract_text_blocks,
    partition_captions,
)
from src.ingestion.tables import extract_tables, link_cross_page_tables

LOG = logging.getLogger(__name__)


def parse_pdf(
    path: Path,
    *,
    processed_dir: Path | None = None,
    crop_dpi: int = 150,
    doc_id: str | None = None,
) -> list[Element]:
    """Parse one PDF into elements, writing figure/table crops to disk.

    Args:
        path: source PDF, never modified.
        processed_dir: root for crop files; defaults to ``data/processed``.
        crop_dpi: render resolution for crops.
        doc_id: override the filename-derived identifier (tests / callers).

    Returns:
        Elements in page order. Figure and table elements have ``image_path``
        set; all elements have a stable, content-derived ``id`` and a bbox.
    """
    document_id = doc_id or derive_doc_id(path)
    crop_root = (processed_dir or get_settings().data_processed) / document_id
    elements: list[Element] = []

    started = time.perf_counter()
    with pymupdf.open(path) as document:
        for index in range(document.page_count):
            page_num = index + 1
            page_started = time.perf_counter()
            page = document[index]

            tables = extract_tables(page, document_id, page_num)
            table_bboxes = [table.bbox for table in tables]
            figures = detect_figures(page, document_id, page_num, table_bboxes)
            targets = tables + figures
            occupied = [target.bbox for target in targets]

            texts = extract_text_blocks(page, document_id, page_num, occupied)
            captions, body = partition_captions(texts, targets)
            for caption_text, target in captions:
                target.caption = caption_text.text

            page_elements = save_crops(page, targets + body, crop_root, crop_dpi)
            elements.extend(page_elements)

            counts = Counter(element.element_type.value for element in page_elements)
            LOG.info(
                "%s p%d: %s in %.2fs",
                document_id,
                page_num,
                ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "empty",
                time.perf_counter() - page_started,
            )

    elements = link_cross_page_tables(elements)

    # Crop filenames are content-derived. A re-ingest after a heuristic change
    # must not leave orphaned images behind in processed/ (ADR-012 rule 7).
    wanted = {f"{element.id}.png" for element in elements if element.image_path is not None}
    stale = [path for path in crop_root.glob("*.png") if path.name not in wanted]
    for path in stale:
        path.unlink()
    if stale:
        LOG.info("%s: removed %d stale crop(s)", document_id, len(stale))

    LOG.info(
        "%s: %d elements from %s in %.2fs",
        document_id,
        len(elements),
        path.name,
        time.perf_counter() - started,
    )
    return elements
