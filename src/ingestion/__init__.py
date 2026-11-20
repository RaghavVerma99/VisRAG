"""Ingestion: raw PDFs -> :class:`~src.ingestion.elements.Element` objects.

The data model is re-exported here so downstream modules can write
``from src.ingestion import Element`` without reaching into ``elements.py``.
"""

from src.ingestion.elements import (
    Answer,
    BBox,
    Citation,
    Element,
    ElementMetadata,
    ElementType,
    MatchedRepr,
    RetrievalFilter,
    ScoredElement,
    Usage,
    make_element_id,
    normalise_bbox,
)
from src.ingestion.pdf_parser import parse_pdf

__all__ = [
    "Answer",
    "BBox",
    "Citation",
    "Element",
    "ElementMetadata",
    "ElementType",
    "MatchedRepr",
    "RetrievalFilter",
    "ScoredElement",
    "Usage",
    "make_element_id",
    "normalise_bbox",
    "parse_pdf",
]
