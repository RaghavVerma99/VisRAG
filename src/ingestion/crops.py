"""Render bbox crops to disk and attach their paths to elements.

Crops are rendered, not extracted with ``get_images()``: most figures in the
probe corpus are vector drawings (ADR-014). Only FIGURE and TABLE elements get
a crop; raw image bytes never enter an Element or a vector-store payload.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from src.ingestion.elements import Element, ElementType

LOG = logging.getLogger(__name__)


def save_crops(
    page: pymupdf.Page,
    elements: list[Element],
    out_dir: Path,
    dpi: int = 150,
) -> list[Element]:
    """Write PNG crops for visual elements; return elements with paths set.

    The filename is the content-derived ``Element.id``, so re-ingesting a
    document overwrites the same files instead of accumulating duplicates.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    updated: list[Element] = []
    for element in elements:
        if element.element_type not in (ElementType.FIGURE, ElementType.TABLE):
            updated.append(element)
            continue
        path = out_dir / f"{element.id}.png"
        clip = pymupdf.Rect(*element.bbox)
        pixmap = page.get_pixmap(clip=clip, dpi=dpi, alpha=False)
        pixmap.save(path)
        updated.append(element.model_copy(update={"image_path": path}))
        LOG.debug(
            "%s p%d: crop %s (%dx%d)",
            element.doc_id,
            element.page,
            path.name,
            pixmap.width,
            pixmap.height,
        )
    return updated
