"""Table detection and markdown extraction with PyMuPDF (ADR-009).

PyMuPDF 1.x ``find_tables()`` is geometric: it needs no ONNX model and no
network. It over-detects on this corpus (39 regions in 46 pages), so the raw
regions are filtered by size and row/column shape before they become evidence.
The rejected regions are logged, not silently discarded.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections import Counter

import pymupdf

from src.ingestion.elements import Element, ElementMetadata, ElementType, normalise_bbox

LOG = logging.getLogger(__name__)

#: A detection narrower than this is usually an axis label or a ruled margin,
#: not a table. Measured against the probe corpus: the spurious 48pt regions in
#: the ResNet paper fall below it, while every real table is >= 236pt wide.
MIN_TABLE_WIDTH = 100.0
MIN_TABLE_HEIGHT = 20.0
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2


def clean_cell(value: object) -> str:
    """Flatten one table cell into a single markdown-safe line."""
    if value is None:
        return ""
    return " ".join(str(value).split()).replace("|", "\\|")


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render rows as a markdown table, using the first row as the header."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(padded[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def extract_tables(page: pymupdf.Page, doc_id: str, page_num: int) -> list[Element]:
    """Return the table elements on one page.

    One element per detected table on this page; a table spanning pages is
    emitted once per page by the caller and linked with ``table_group``.
    """
    try:
        # PyMuPDF 1.28 prints a layout-package suggestion to stdout on every
        # call. It is noise in a batch log, so swallow only that print.
        with contextlib.redirect_stdout(io.StringIO()):
            found = page.find_tables().tables
    except Exception as exc:  # find_tables is best-effort; never fail ingestion
        LOG.warning("%s p%d: table detection failed: %s", doc_id, page_num, exc)
        return []

    kept: list[Element] = []
    for raw in found:
        bbox = normalise_bbox(tuple(float(value) for value in raw.bbox))
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        rows = [[clean_cell(cell) for cell in row] for row in (raw.extract() or [])]
        rows = [row for row in rows if any(row)]
        cols = max((len(row) for row in rows), default=0)

        if (
            len(rows) < MIN_TABLE_ROWS
            or cols < MIN_TABLE_COLS
            or width < MIN_TABLE_WIDTH
            or height < MIN_TABLE_HEIGHT
        ):
            LOG.debug(
                "%s p%d: dropped table candidate %sx%s (%d rows, %d cols)",
                doc_id,
                page_num,
                round(width),
                round(height),
                len(rows),
                cols,
            )
            continue

        kept.append(
            Element(
                doc_id=doc_id,
                page=page_num,
                element_type=ElementType.TABLE,
                bbox=bbox,
                structured=rows_to_markdown(rows),
                metadata=ElementMetadata(model_version="pymupdf-find_tables"),
            )
        )

    LOG.info(
        "%s p%d: tables %d kept / %d detected",
        doc_id,
        page_num,
        len(kept),
        len(found),
    )
    return kept


def _column_count(table: Element) -> int:
    if not table.structured:
        return 0
    return max(0, table.structured.splitlines()[0].count("|") - 1)


def link_cross_page_tables(elements: list[Element]) -> list[Element]:
    """Link tables that continue on the next page (root §5).

    Conservative by design: only a single table per page, with an almost
    identical horizontal span and the same column count, is treated as a
    continuation. A wrong link is worse than a missing one — retrieval must
    return the whole group, and a false group would contaminate citations.
    """
    tables = [element for element in elements if element.element_type is ElementType.TABLE]
    per_page = Counter((table.doc_id, table.page) for table in tables)
    by_doc: dict[str, list[Element]] = {}
    for table in tables:
        by_doc.setdefault(table.doc_id, []).append(table)

    for doc_id, doc_tables in by_doc.items():
        previous: Element | None = None
        group_number = 0
        for table in sorted(doc_tables, key=lambda item: item.page):
            linkable = (
                previous is not None
                and table.page == previous.page + 1
                and per_page[(doc_id, previous.page)] == 1
                and per_page[(doc_id, table.page)] == 1
                and abs(table.bbox[0] - previous.bbox[0]) <= 10
                and abs(table.bbox[2] - previous.bbox[2]) <= 10
                and _column_count(table) == _column_count(previous) > 0
            )
            if not linkable:
                previous = table
                continue

            assert previous is not None
            group = previous.metadata.table_group or f"{doc_id}:t{group_number + 1}"
            if previous.metadata.table_group is None:
                group_number += 1
            previous.metadata.table_group = group
            previous.metadata.continues = True
            table.metadata.table_group = group
            table.metadata.continues = False
            LOG.info("%s: linked p%d -> p%d as %s", doc_id, previous.page, table.page, group)
            previous = table
    return elements
