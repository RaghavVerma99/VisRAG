"""Embedding-stage tests with fake models and an embedded Qdrant.

No model weights are downloaded and no hosted API is called. External calls are
replaced by deterministic fakes (tests/AGENTS.md rule 2).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from src.config import Settings
from src.embedding.caption import (
    CachedCaptioner,
    CaptionCache,
    HostedCaptioner,
    NullCaptioner,
    build_captioner,
)
from src.embedding.image import feature_array
from src.embedding.store import (
    collection_name,
    embed_and_store,
    load_manifest,
    point_id,
    text_representation,
)
from src.embedding.text import normalise_vectors
from src.ingestion.elements import Element, ElementType

BBOX = (72.0, 240.0, 480.0, 300.0)


class FakeTextEmbedder:
    def __init__(self, dimension: int = 4) -> None:
        self.model_version = f"fake-text-v{dimension}"
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        rows = []
        for text in texts:
            seed = sum(bytearray(text.encode("utf-8"))) + 1
            rows.append([float((seed * (i + 1)) % 97) / 97 for i in range(self._dimension)])
        return normalise_vectors(np.asarray(rows, dtype=np.float32))


class FakeImageEmbedder:
    model_version = "fake-image-v1"

    @property
    def image_dimension(self) -> int:
        return 2

    @property
    def text_dimension(self) -> int:
        return 2

    def embed_images(self, paths: list[Path]) -> NDArray[np.float32]:
        rows = [[float(len(path.name) % 7 + 1), 0.5] for path in paths]
        return normalise_vectors(np.asarray(rows, dtype=np.float32))

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        rows = [[0.25, float(len(text) % 7 + 1)] for text in texts]
        return normalise_vectors(np.asarray(rows, dtype=np.float32))


class FakeCaptioner:
    def __init__(self, caption: str = "A described figure.") -> None:
        self.caption = caption
        self.calls: list[str] = []

    def describe(self, image_path: Path | None, doc_id: str) -> str | None:
        self.calls.append(doc_id)
        return self.caption


class FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> NDArray[np.float32]:
        return np.asarray([self.values], dtype=np.float32)


class FakeModelOutput:
    def __init__(self, **attributes: object) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


@pytest.fixture
def qdrant(tmp_path: Path) -> QdrantClient:
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    yield client
    client.close()


def make_element(
    element_type: ElementType = ElementType.TEXT,
    **overrides: object,
) -> Element:
    values: dict[str, object] = {
        "doc_id": "paper-a",
        "page": 1,
        "element_type": element_type,
        "bbox": BBOX,
        "text": "Revenue was 12.4M.",
    }
    values.update(overrides)
    return Element(**values)  # type: ignore[arg-type]


def crop_file(tmp_path: Path, name: str = "crop.png") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake crop bytes")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_feature_array_accepts_tensor_and_output_object() -> None:
    tensor = FakeTensor([3.0, 4.0])
    # transformers <5: a tensor is returned directly.
    assert feature_array(tensor).tolist() == [[3.0, 4.0]]
    # transformers 5.x: a model output with pooler_output.
    assert feature_array(FakeModelOutput(pooler_output=tensor)).tolist() == [[3.0, 4.0]]
    # Prefer the explicit embedding field when both are present.
    assert feature_array(
        FakeModelOutput(image_embeds=tensor, pooler_output=FakeTensor([9.0, 9.0]))
    ).tolist() == [[3.0, 4.0]]


def test_normalise_vectors_is_unit_length() -> None:
    vectors = normalise_vectors(np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1].tolist() == [0.0, 0.0]


def test_point_id_is_stable_and_a_uuid() -> None:
    first = point_id("b39afd99fa59fce1")
    assert first == point_id("b39afd99fa59fce1")
    assert first != point_id("b39afd99fa59fce2")
    assert len(first) == 36
    assert first.count("-") == 4


def test_text_representation_depends_on_element_type() -> None:
    assert text_representation(make_element()) == "Revenue was 12.4M."
    assert (
        text_representation(
            make_element(ElementType.TABLE, text="ignored", structured="| a |\n| --- |\n| 1 |")
        )
        == "| a |\n| --- |\n| 1 |"
    )
    assert (
        text_representation(make_element(ElementType.FIGURE, caption="A bar chart", text=""))
        == "A bar chart"
    )


def test_load_manifest_round_trip(tmp_path: Path) -> None:
    original = make_element(ElementType.TABLE, structured="| a |", image_path=crop_file(tmp_path))
    path = tmp_path / "elements.jsonl"
    path.write_text(original.model_dump_json() + "\n\n", encoding="utf-8")
    loaded = load_manifest(path)
    assert len(loaded) == 1
    assert loaded[0].id == original.id
    assert loaded[0].image_path == original.image_path


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_writes_named_vectors_and_citation_payload(
    qdrant: QdrantClient, tmp_path: Path
) -> None:
    crop = crop_file(tmp_path)
    elements = [
        make_element(page=1),
        make_element(ElementType.TABLE, page=2, structured="| a |\n| --- |\n| 1 |"),
        make_element(ElementType.FIGURE, page=3, text="", caption="A chart", image_path=crop),
    ]
    stats = embed_and_store(
        elements,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        captioner=NullCaptioner(),
        client=qdrant,
        collection="test_elements",
    )

    assert stats.as_dict()["elements"] == 3
    assert stats.text_vectors == 3
    assert stats.image_vectors == 1
    name = collection_name("test_elements", 4, 2)
    assert stats.collection == name
    assert qdrant.collection_exists(name)

    config = qdrant.get_collection(name).config.params.vectors
    assert isinstance(config, dict)
    assert config["text"].size == 4
    assert config["image"].size == 2

    points = qdrant.retrieve(
        name,
        ids=[point_id(element.id) for element in elements],
        with_payload=True,
        with_vectors=True,
    )
    assert len(points) == 3
    by_type = {point.payload["element_type"]: point for point in points}  # type: ignore[index]

    assert by_type["TABLE"].payload["bbox"] == list(BBOX)  # type: ignore[index]
    assert by_type["TABLE"].payload["doc_id"] == "paper-a"  # type: ignore[index]
    assert by_type["TABLE"].payload["page"] == 2  # type: ignore[index]
    assert by_type["TABLE"].payload["repr_type"] == "text"  # type: ignore[index]
    assert "image" not in by_type["TABLE"].vector  # type: ignore[index]
    assert by_type["FIGURE"].payload["has_image"] is True  # type: ignore[index]
    assert by_type["FIGURE"].payload["repr_type"] == "caption"  # type: ignore[index]
    assert "image" in by_type["FIGURE"].vector  # type: ignore[index]
    assert all(
        point.payload["model_version"] == "fake-text-v4|fake-image-v1"  # type: ignore[index]
        for point in points
    )


def test_named_vector_search_returns_figure(qdrant: QdrantClient, tmp_path: Path) -> None:
    crop = crop_file(tmp_path)
    elements = [
        make_element(page=1),
        make_element(ElementType.FIGURE, page=2, text="", caption="Chart", image_path=crop),
    ]
    embed_and_store(
        elements,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        captioner=NullCaptioner(),
        client=qdrant,
        collection="test_search",
    )
    hits = qdrant.query_points(
        collection_name("test_search", 4, 2),
        query=[1.0, 0.0],
        using="image",
        limit=5,
    ).points
    assert [hit.id for hit in hits] == [point_id(elements[1].id)]


def test_reingest_does_not_duplicate_points(qdrant: QdrantClient) -> None:
    args = {
        "text_embedder": FakeTextEmbedder(),
        "image_embedder": FakeImageEmbedder(),
        "captioner": NullCaptioner(),
        "client": qdrant,
        "collection": "test_idempotent",
    }
    first = embed_and_store([make_element()], **args)  # type: ignore[arg-type]
    second = embed_and_store([make_element()], **args)  # type: ignore[arg-type]

    assert first.elements == second.elements == 1
    assert first.collection == second.collection == collection_name("test_idempotent", 4, 2)
    assert qdrant.count(first.collection).count == 1


def test_model_dimension_change_uses_a_new_collection(qdrant: QdrantClient) -> None:
    args = {
        "image_embedder": FakeImageEmbedder(),
        "captioner": NullCaptioner(),
        "client": qdrant,
        "collection": "test_recreate",
    }
    first = embed_and_store([make_element()], text_embedder=FakeTextEmbedder(4), **args)  # type: ignore[arg-type]
    second = embed_and_store([make_element()], text_embedder=FakeTextEmbedder(6), **args)  # type: ignore[arg-type]

    assert first.collection == collection_name("test_recreate", 4, 2)
    assert second.collection == collection_name("test_recreate", 6, 2)
    assert qdrant.collection_exists(first.collection)
    assert qdrant.collection_exists(second.collection)

    first_config = qdrant.get_collection(first.collection).config.params.vectors
    second_config = qdrant.get_collection(second.collection).config.params.vectors
    assert isinstance(first_config, dict)
    assert isinstance(second_config, dict)
    assert first_config["text"].size == 4
    assert second_config["text"].size == 6
    assert qdrant.count(first.collection).count == 1
    assert qdrant.count(second.collection).count == 1


def test_caption_is_added_and_cached(qdrant: QdrantClient, tmp_path: Path) -> None:
    crop = crop_file(tmp_path)
    figure = make_element(
        ElementType.FIGURE,
        text="",
        caption=None,
        image_path=crop,
    )
    captioner = FakeCaptioner()
    stats = embed_and_store(
        [figure],
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        captioner=captioner,
        client=qdrant,
        collection="test_caption",
    )
    assert stats.captions_added == 1
    assert captioner.calls == ["paper-a"]
    point = qdrant.retrieve(
        collection_name("test_caption", 4, 2),
        ids=[point_id(figure.id)],
        with_payload=True,
    )[0]
    assert point.payload["caption"] == "A described figure."  # type: ignore[index]


def test_element_without_representation_is_skipped(qdrant: QdrantClient) -> None:
    empty = make_element(text="   ")
    stats = embed_and_store(
        [empty],
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        captioner=NullCaptioner(),
        client=qdrant,
        collection="test_skip",
    )
    assert stats.skipped == 1
    assert stats.elements == 0
    assert qdrant.count(collection_name("test_skip", 4, 2)).count == 0


# ---------------------------------------------------------------------------
# Caption gate and cache
# ---------------------------------------------------------------------------


def test_hosted_captioning_requires_approval(tmp_path: Path) -> None:
    captioner = HostedCaptioner(Settings(_env_file=None))
    with pytest.raises(PermissionError, match="CAPTION_ALLOW_HOSTED"):
        captioner.describe(crop_file(tmp_path), "paper-a")


def test_build_captioner_defaults_to_no_op() -> None:
    captioner = build_captioner(Settings(_env_file=None))
    assert isinstance(captioner, NullCaptioner)


def test_local_llava_fails_loudly() -> None:
    settings = Settings(_env_file=None, caption_provider="local-llava")
    with pytest.raises(NotImplementedError, match="LLaVA"):
        build_captioner(settings)


def test_caption_cache_avoids_a_second_call(tmp_path: Path) -> None:
    crop = crop_file(tmp_path)
    inner = FakeCaptioner()
    cached = CachedCaptioner(inner, CaptionCache(tmp_path / "cache.json"))

    assert cached.describe(crop, "paper-a") == "A described figure."
    assert cached.describe(crop, "paper-a") == "A described figure."
    assert len(inner.calls) == 1

    # The cache survives a new process-like object.
    again = CachedCaptioner(inner, CaptionCache(tmp_path / "cache.json"))
    assert again.describe(crop, "paper-a") == "A described figure."
    assert len(inner.calls) == 1


def test_text_vector_counts_are_recorded(qdrant: QdrantClient) -> None:
    elements = [make_element(), make_element(page=2)]
    stats = embed_and_store(
        elements,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        captioner=NullCaptioner(),
        client=qdrant,
        collection="test_stats",
    )
    assert Counter(e.element_type.value for e in elements)["TEXT"] == stats.text_vectors
    assert stats.model_version == "fake-text-v4|none"
