"""Tests for the canonical data model (ADR-012).

These pin the invariants every other module relies on: stable ids, valid
bounding boxes, linked cross-page tables, and answers that cannot both abstain
and cite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.ingestion import (
    Answer,
    Citation,
    Element,
    ElementMetadata,
    ElementType,
    MatchedRepr,
    RetrievalFilter,
    ScoredElement,
    Usage,
    make_element_id,
)

BBOX = (72.0, 240.0, 480.0, 300.0)


def make_element(**overrides: object) -> Element:
    data: dict[str, object] = {
        "doc_id": "acme-2025-q3",
        "page": 12,
        "element_type": ElementType.TEXT,
        "bbox": BBOX,
        "text": "Q3 revenue was 12.4M.",
    }
    data.update(overrides)
    return Element(**data)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_id_is_derived_from_content() -> None:
    element = make_element()
    assert element.id == make_element_id("acme-2025-q3", 12, BBOX)
    assert len(element.id) == 16


def test_id_is_stable_across_equivalent_inputs() -> None:
    """A list bbox and a tuple bbox must produce the same id."""
    from_list = make_element(bbox=[72.0, 240.0, 480.0, 300.0])
    from_tuple = make_element(bbox=(72, 240, 480, 300))
    assert from_list.id == from_tuple.id


def test_id_is_stable_across_float_noise() -> None:
    """Sub-hundredth jitter from a renderer must not mint a new element."""
    exact = make_element(bbox=(72.0, 240.0, 480.0, 300.0))
    jittered = make_element(bbox=(72.0000001, 240.0000002, 480.0000003, 300.0000004))
    assert exact.id == jittered.id


def test_id_changes_with_page_or_document() -> None:
    base = make_element()
    assert make_element(page=13).id != base.id
    assert make_element(doc_id="other-doc").id != base.id


def test_explicit_id_is_preserved() -> None:
    element = make_element(id="fixed-id")
    assert element.id == "fixed-id"


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------


def test_bbox_is_rounded_to_two_decimals() -> None:
    element = make_element(bbox=(72.123456, 240.987654, 480.5, 300.0))
    assert element.bbox == (72.12, 240.99, 480.5, 300.0)


@pytest.mark.parametrize(
    "bbox",
    [
        (72.0, 240.0, 72.0, 300.0),  # zero width
        (72.0, 240.0, 480.0, 240.0),  # zero height
        (480.0, 240.0, 72.0, 300.0),  # inverted
        (-1.0, 240.0, 480.0, 300.0),  # negative origin
        (72.0, 240.0, 480.0),  # wrong arity
    ],
)
def test_invalid_bbox_is_rejected(bbox: tuple[float, ...]) -> None:
    with pytest.raises(ValidationError):
        make_element(bbox=bbox)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_continues_requires_a_table_group() -> None:
    with pytest.raises(ValidationError, match="table_group"):
        make_element(element_type=ElementType.TABLE, metadata=ElementMetadata(continues=True))


def test_cross_page_table_parts_share_a_group() -> None:
    first = make_element(
        element_type=ElementType.TABLE,
        metadata=ElementMetadata(table_group="acme-2025-q3:t1", continues=True),
    )
    second = make_element(
        page=13,
        element_type=ElementType.TABLE,
        metadata=ElementMetadata(table_group="acme-2025-q3:t1"),
    )
    assert first.metadata.table_group == second.metadata.table_group
    assert first.id != second.id


def test_caption_of_is_only_valid_on_text() -> None:
    with pytest.raises(ValidationError, match="caption_of"):
        make_element(element_type=ElementType.FIGURE, metadata=ElementMetadata(caption_of="abc"))


def test_caption_element_links_to_its_figure() -> None:
    caption = make_element(
        element_type=ElementType.TEXT,
        text="Figure 1: revenue by quarter.",
        metadata=ElementMetadata(caption_of="fig-id"),
    )
    assert caption.metadata.caption_of == "fig-id"


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_element(colour="red")


def test_metadata_rejects_arbitrary_keys() -> None:
    with pytest.raises(ValidationError):
        ElementMetadata.model_validate({"random_key": 1})


# ---------------------------------------------------------------------------
# Payload / serialisation
# ---------------------------------------------------------------------------


def test_payload_is_json_serialisable_and_tuple_free() -> None:
    element = make_element(image_path="data/processed/acme/p12.png")
    payload = element.to_payload()

    assert payload["bbox"] == [72.0, 240.0, 480.0, 300.0]
    assert payload["element_type"] == "TEXT"
    assert Path(str(payload["image_path"])) == Path("data/processed/acme/p12.png")
    assert "embedding" not in payload
    json.dumps(payload)  # must not raise


def test_payload_omits_unset_metadata() -> None:
    payload = make_element().to_payload()
    assert payload["metadata"] == {}


# ---------------------------------------------------------------------------
# Retrieval / generation boundary
# ---------------------------------------------------------------------------


def test_scored_element_keeps_element_fields() -> None:
    base = make_element()
    scored = ScoredElement(**base.model_dump(), score=0.87, matched_repr=MatchedRepr.CAPTION)
    assert scored.id == base.id
    assert scored.score == 0.87
    assert scored.matched_repr is MatchedRepr.CAPTION


def test_citation_round_trips_from_element() -> None:
    element = make_element(element_type=ElementType.TABLE)
    citation = Citation.from_element(element)

    assert citation.element_id == element.id
    assert citation.page == element.page
    assert citation.bbox == element.bbox
    assert citation.element_type is ElementType.TABLE


def test_abstaining_answer_cannot_cite() -> None:
    citation = Citation.from_element(make_element())
    with pytest.raises(ValidationError, match="abstain"):
        Answer(text="I don't know.", abstained=True, citations=[citation])


def test_abstaining_answer_is_valid_without_citations() -> None:
    answer = Answer(text="I don't know.", abstained=True)
    assert answer.citations == []
    assert answer.abstained is True


def test_grounded_answer_carries_citations_and_usage() -> None:
    citation = Citation.from_element(make_element())
    answer = Answer(
        text="Q3 revenue was 12.4M.",
        citations=[citation],
        model="gpt-4o",
        usage=Usage(prompt_tokens=900, completion_tokens=40, images=2),
    )
    assert answer.citations[0].element_id == citation.element_id
    assert answer.usage is not None
    assert answer.usage.images == 2


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_filter_matches_every_constraint() -> None:
    element = make_element(page=12, element_type=ElementType.TEXT)
    assert RetrievalFilter().matches(element)
    assert RetrievalFilter(doc_id="acme-2025-q3").matches(element)
    assert RetrievalFilter(element_type=ElementType.TEXT).matches(element)
    assert RetrievalFilter(page_min=10, page_max=15).matches(element)


def test_filter_rejects_mismatches() -> None:
    element = make_element(page=12, element_type=ElementType.TEXT)
    assert not RetrievalFilter(doc_id="other").matches(element)
    assert not RetrievalFilter(element_type=ElementType.FIGURE).matches(element)
    assert not RetrievalFilter(page_min=13).matches(element)
    assert not RetrievalFilter(page_max=11).matches(element)


def test_inverted_page_range_is_rejected() -> None:
    with pytest.raises(ValidationError, match="page_min"):
        RetrievalFilter(page_min=20, page_max=10)
