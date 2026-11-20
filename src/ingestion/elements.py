"""Canonical data model for the whole pipeline (ADR-012).

Every module imports these models; none of them redeclares one. The rules that
matter:

* ``Element.id`` is content-derived, so re-ingesting the same PDF is idempotent.
* ``bbox`` is ``(x0, y0, x1, y1)`` in page coordinates, origin top-left, rounded
  to 2 decimal places so ids stay stable across renders.
* There is no ``embedding`` field — vectors live in the vector store.
* There is no ``CAPTION`` element type — a caption is either ``FIGURE.caption``
  or a ``TEXT`` element whose ``metadata.caption_of`` points at its figure.
* A table is never split *within* a page; a table spanning pages is emitted one
  element per page and linked by ``metadata.table_group``.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BBox = tuple[float, float, float, float]

#: Coordinates are rounded to this many decimal places before hashing.
BBOX_PRECISION = 2


class ElementType(StrEnum):
    """What a region of a page is."""

    TEXT = "TEXT"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    FORMULA = "FORMULA"
    HEADER = "HEADER"
    FOOTER = "FOOTER"


class MatchedRepr(StrEnum):
    """Which representation produced a retrieval hit."""

    TEXT = "text"
    CAPTION = "caption"
    IMAGE = "image"


def normalise_bbox(bbox: tuple[float, ...]) -> BBox:
    """Round, validate and freeze a bounding box.

    Raises ``ValueError`` for degenerate or inverted boxes: a citation with a
    zero-area bbox cannot be highlighted in the UI.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 coordinates, got {len(bbox)}")
    x0, y0, x1, y1 = (round(float(c), BBOX_PRECISION) for c in bbox)
    if min(x0, y0) < 0:
        raise ValueError(f"bbox coordinates must be non-negative, got {bbox}")
    if x1 <= x0:
        raise ValueError(f"bbox has non-positive width: x0={x0}, x1={x1}")
    if y1 <= y0:
        raise ValueError(f"bbox has non-positive height: y0={y0}, y1={y1}")
    return (x0, y0, x1, y1)


def make_element_id(doc_id: str, page: int, bbox: BBox) -> str:
    """Stable, content-derived element id.

    Deterministic in ``(doc_id, page, bbox)`` so that re-ingesting a document
    overwrites its elements instead of duplicating them.
    """
    coords = ",".join(f"{c:.{BBOX_PRECISION}f}" for c in bbox)
    digest = hashlib.sha1(f"{doc_id}:{page}:{coords}".encode()).hexdigest()
    return digest[:16]


class ElementMetadata(BaseModel):
    """Typed metadata. Arbitrary keys are rejected on purpose."""

    model_config = ConfigDict(extra="forbid")

    table_group: str | None = None
    continues: bool = False
    caption_of: str | None = None
    ocr_used: bool = False
    model_version: str | None = None


class Element(BaseModel):
    """One retrievable region of a page."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default="",
        description="Content-derived; computed from (doc_id, page, bbox) when empty.",
    )
    doc_id: str = Field(min_length=1)
    page: int = Field(ge=1, description="1-based page number.")
    element_type: ElementType
    bbox: BBox
    text: str = ""
    structured: str | None = Field(
        default=None, description="Markdown for tables, LaTeX for formulas."
    )
    caption: str | None = Field(default=None, description="VLM description; FIGURE/TABLE.")
    image_path: Path | None = Field(default=None, description="Crop on disk; never raw bytes.")
    metadata: ElementMetadata = Field(default_factory=ElementMetadata)

    @field_validator("bbox", mode="after")
    @classmethod
    def _validate_bbox(cls, bbox: tuple[float, ...]) -> BBox:
        return normalise_bbox(bbox)

    @model_validator(mode="after")
    def _finalise(self) -> Self:
        if not self.id:
            self.id = make_element_id(self.doc_id, self.page, self.bbox)

        if self.metadata.continues and not self.metadata.table_group:
            raise ValueError("metadata.continues requires metadata.table_group")

        if self.metadata.caption_of is not None and self.element_type is not ElementType.TEXT:
            raise ValueError("metadata.caption_of is only valid on TEXT elements")

        return self

    def to_payload(self) -> dict[str, object]:
        """Qdrant payload for this element.

        This is a serialisation boundary, so it is deliberately a plain dict.
        ``bbox`` becomes a 4-element list because payloads carry no tuples.
        Vectors are added by the embedding stage, never here.
        """
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "page": self.page,
            "element_type": self.element_type.value,
            "bbox": list(self.bbox),
            "text": self.text,
            "structured": self.structured,
            "caption": self.caption,
            "image_path": str(self.image_path) if self.image_path is not None else None,
            "metadata": self.metadata.model_dump(exclude_none=True, exclude_defaults=True),
        }


class ScoredElement(Element):
    """An element as returned by retrieval."""

    score: float
    matched_repr: MatchedRepr = MatchedRepr.TEXT


class Citation(BaseModel):
    """A pointer to evidence. Cites ``doc_id``/``page``/``bbox``, never a chunk."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    bbox: BBox
    element_id: str = Field(min_length=1)
    element_type: ElementType

    @field_validator("bbox", mode="after")
    @classmethod
    def _validate_bbox(cls, bbox: tuple[float, ...]) -> BBox:
        return normalise_bbox(bbox)

    @classmethod
    def from_element(cls, element: Element) -> Self:
        return cls(
            doc_id=element.doc_id,
            page=element.page,
            bbox=element.bbox,
            element_id=element.id,
            element_type=element.element_type,
        )


class Usage(BaseModel):
    """Token and image accounting for one generation call."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    images: int = Field(default=0, ge=0)


class Answer(BaseModel):
    """A grounded answer, or an explicit abstention."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    model: str = ""
    usage: Usage | None = None

    @model_validator(mode="after")
    def _abstention_cites_nothing(self) -> Self:
        if self.abstained and self.citations:
            raise ValueError("an abstaining answer must not cite evidence")
        return self


class RetrievalFilter(BaseModel):
    """Optional metadata filters for retrieval."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = None
    element_type: ElementType | None = None
    page_min: int | None = Field(default=None, ge=1)
    page_max: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _page_range_is_ordered(self) -> Self:
        if (
            self.page_min is not None
            and self.page_max is not None
            and self.page_min > self.page_max
        ):
            raise ValueError(f"page_min ({self.page_min}) exceeds page_max ({self.page_max})")
        return self

    def matches(self, element: Element) -> bool:
        """True if ``element`` satisfies every constraint that is set."""
        if self.doc_id is not None and element.doc_id != self.doc_id:
            return False
        if self.element_type is not None and element.element_type is not self.element_type:
            return False
        if self.page_min is not None and element.page < self.page_min:
            return False
        return self.page_max is None or element.page <= self.page_max


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
]
