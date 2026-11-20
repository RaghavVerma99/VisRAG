"""Layout heuristics: text blocks, vector/raster figures, captions, headers.

Deliberately small and deterministic. A future ``unstructured`` pass can sit
behind a capability gate (ADR-009); this module is the v0 path and is pure
PyMuPDF geometry plus regexes, with no model download.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf

from src.ingestion.elements import (
    BBox,
    Element,
    ElementMetadata,
    ElementType,
    normalise_bbox,
)

LOG = logging.getLogger(__name__)

CAPTION_RE = re.compile(r"^(figure|fig\.?|table)\s*\d+\b", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")

HEADER_FRACTION = 0.08
FOOTER_FRACTION = 0.92
CAPTION_MAX_GAP = 40.0
MIN_FIGURE_AREA = 400.0
MAX_PAGE_COVERAGE = 0.70
DRAW_MIN_SIDE = 1.0
CLUSTER_PADDING = 10.0
CROP_PADDING = 4.0
#: Panels of one composite figure are often separated by a small gutter. Merge
#: them so a caption links to the whole figure, not to one arbitrary panel.
#: The Transformer paper has side-by-side panels 100pt apart sharing one
#: caption; the ResNet paper has panels 14pt apart. Both are one figure.
FIGURE_MERGE_H_GAP = 120.0
FIGURE_MERGE_V_GAP = 60.0


def derive_doc_id(path: Path) -> str:
    """Stable, human-readable document id from the filename."""
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return slug or "document"


def _coverage(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner`` covered by ``outer`` (0..1)."""
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if area <= 0 or ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / area


def _is_header_footer(bbox: BBox, text: str, page_rect: pymupdf.Rect) -> bool:
    """Drop page furniture that would otherwise pollute retrieval."""
    if bbox[3] <= page_rect.height * HEADER_FRACTION:
        return True
    if bbox[1] >= page_rect.height * FOOTER_FRACTION:
        return True
    return bool(PAGE_NUMBER_RE.match(text.strip()))


def _union_rect(rects: list[pymupdf.Rect], padding: float, page_rect: pymupdf.Rect) -> BBox:
    x0 = max(page_rect.x0, min(r.x0 for r in rects) - padding)
    y0 = max(page_rect.y0, min(r.y0 for r in rects) - padding)
    x1 = min(page_rect.x1, max(r.x1 for r in rects) + padding)
    y1 = min(page_rect.y1, max(r.y1 for r in rects) + padding)
    return normalise_bbox((x0, y0, x1, y1))


def _rebuild_with_bbox(element: Element, bbox: BBox) -> Element:
    """Rebuild an element so its content-derived id matches its new bbox."""
    return Element(
        doc_id=element.doc_id,
        page=element.page,
        element_type=element.element_type,
        bbox=bbox,
        text=element.text,
        structured=element.structured,
        caption=element.caption,
        image_path=element.image_path,
        metadata=element.metadata,
    )


def merge_nearby_figures(elements: list[Element]) -> list[Element]:
    """Merge panels that sit within a small gutter into one figure.

    Measured on the ResNet paper page 4: five panels of Figure 3 were otherwise
    emitted as five crops with one caption attached to only one of them.
    """
    merged = list(elements)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                x_gap = max(a.bbox[0] - b.bbox[2], b.bbox[0] - a.bbox[2], 0.0)
                y_gap = max(a.bbox[1] - b.bbox[3], b.bbox[1] - a.bbox[3], 0.0)
                if x_gap > FIGURE_MERGE_H_GAP or y_gap > FIGURE_MERGE_V_GAP:
                    continue
                bbox = normalise_bbox(
                    (
                        min(a.bbox[0], b.bbox[0]),
                        min(a.bbox[1], b.bbox[1]),
                        max(a.bbox[2], b.bbox[2]),
                        max(a.bbox[3], b.bbox[3]),
                    )
                )
                merged = [m for k, m in enumerate(merged) if k not in (i, j)]
                merged.append(_rebuild_with_bbox(a, bbox))
                changed = True
                break
            if changed:
                break
    return merged


def _cluster_drawings(rects: list[pymupdf.Rect]) -> list[list[pymupdf.Rect]]:
    """Group drawing rects that touch or are within ``CLUSTER_PADDING``."""
    parents = list(range(len(rects)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    expanded = [
        pymupdf.Rect(
            r.x0 - CLUSTER_PADDING,
            r.y0 - CLUSTER_PADDING,
            r.x1 + CLUSTER_PADDING,
            r.y1 + CLUSTER_PADDING,
        )
        for r in rects
    ]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if expanded[i].intersects(expanded[j]):
                parents[find(j)] = find(i)

    groups: dict[int, list[pymupdf.Rect]] = {}
    for index, rect in enumerate(rects):
        groups.setdefault(find(index), []).append(rect)
    return list(groups.values())


def detect_figures(
    page: pymupdf.Page,
    doc_id: str,
    page_num: int,
    table_bboxes: list[BBox],
) -> list[Element]:
    """Find figure regions from vector drawings and embedded raster images.

    Crops are rendered from the bbox later (``crops.py``): the probe corpus has
    18 vector-chart pages but only 5 raster images, so ``get_images()`` alone
    would miss almost every figure.
    """
    page_area = page.rect.width * page.rect.height
    occupied = [
        (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2) for bbox in table_bboxes
    ]
    elements: list[Element] = []

    drawings: list[pymupdf.Rect] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if not isinstance(rect, pymupdf.Rect):
            continue
        if rect.width < DRAW_MIN_SIDE or rect.height < DRAW_MIN_SIDE:
            continue
        if any(_coverage((rect.x0, rect.y0, rect.x1, rect.y1), box) > 0.5 for box in occupied):
            continue
        drawings.append(rect)

    for group in _cluster_drawings(drawings):
        if len(group) < 8:
            continue
        bbox = _union_rect(group, CROP_PADDING, page.rect)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < MIN_FIGURE_AREA or area > page_area * MAX_PAGE_COVERAGE:
            LOG.debug(
                "%s p%d: dropped vector figure cluster (%d paths, %.0f pt2)",
                doc_id,
                page_num,
                len(group),
                area,
            )
            continue
        elements.append(
            Element(
                doc_id=doc_id,
                page=page_num,
                element_type=ElementType.FIGURE,
                bbox=bbox,
                metadata=ElementMetadata(model_version="pymupdf-drawings"),
            )
        )

    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            if not isinstance(rect, pymupdf.Rect):
                continue
            bbox = normalise_bbox(
                (
                    max(page.rect.x0, rect.x0 - CROP_PADDING),
                    max(page.rect.y0, rect.y0 - CROP_PADDING),
                    min(page.rect.x1, rect.x1 + CROP_PADDING),
                    min(page.rect.y1, rect.y1 + CROP_PADDING),
                )
            )
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < MIN_FIGURE_AREA:
                continue
            if any(
                _coverage(bbox, (e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3])) > 0.5
                for e in elements
            ):
                continue  # already covered by a vector figure cluster
            elements.append(
                Element(
                    doc_id=doc_id,
                    page=page_num,
                    element_type=ElementType.FIGURE,
                    bbox=bbox,
                    metadata=ElementMetadata(model_version="pymupdf-images"),
                )
            )

    figures = merge_nearby_figures(elements)
    LOG.info(
        "%s p%d: figures %d from %d drawings",
        doc_id,
        page_num,
        len(figures),
        len(drawings),
    )
    return figures


def extract_text_blocks(
    page: pymupdf.Page,
    doc_id: str,
    page_num: int,
    occupied: list[BBox],
) -> list[Element]:
    """Extract body text, dropping headers, footers and text inside figures/tables."""
    elements: list[Element] = []
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:  # block_type 1 is an embedded image
            continue
        text = " ".join(str(block[4]).split())
        if not text:
            continue
        bbox = normalise_bbox(tuple(float(value) for value in block[:4]))
        if _is_header_footer(bbox, text, page.rect):
            continue
        if any(_coverage(bbox, region) > 0.5 for region in occupied):
            continue  # the table markdown or figure crop already carries it
        elements.append(
            Element(
                doc_id=doc_id,
                page=page_num,
                element_type=ElementType.TEXT,
                bbox=bbox,
                text=text,
            )
        )
    return elements


def _best_caption_target(caption: Element, targets: list[Element]) -> Element | None:
    """Nearest figure/table whose horizontal span contains the caption."""
    centre = (caption.bbox[0] + caption.bbox[2]) / 2
    best: Element | None = None
    best_gap = CAPTION_MAX_GAP + 1.0
    for target in targets:
        if not (target.bbox[0] - 10 <= centre <= target.bbox[2] + 10):
            continue
        gap = max(target.bbox[1] - caption.bbox[3], caption.bbox[1] - target.bbox[3], 0.0)
        if gap < best_gap:
            best, best_gap = target, gap
    return best if best_gap <= CAPTION_MAX_GAP else None


def partition_captions(
    texts: list[Element], targets: list[Element]
) -> tuple[list[tuple[Element, Element]], list[Element]]:
    """Split text into (caption, target) pairs and ordinary body text.

    A caption found on the PDF is written into ``target.caption`` and is *not*
    emitted as a second TEXT element — this is the ``FIGURE.caption`` branch of
    the contract, and avoids retrieving the same evidence twice (root §5).
    An unmatched caption stays as ordinary TEXT so no content is lost.
    """
    captions: list[tuple[Element, Element]] = []
    body: list[Element] = []
    for text in texts:
        target = _best_caption_target(text, targets) if CAPTION_RE.match(text.text) else None
        if target is None:
            body.append(text)
        else:
            captions.append((text, target))
    return captions, body
