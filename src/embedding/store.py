"""Offline embedding stage: Elements -> Qdrant named vectors + payloads.

One collection, two named vectors:

* ``text`` — sentence-transformers over the element's best text representation.
* ``image`` — CLIP over a FIGURE crop only (ADR-013).

Embedding owns the collection schema and writes; retrieval only reads/searches
(ADR-015). Point ids are deterministic UUIDs derived from ``Element.id`` so a
re-ingest overwrites rather than appends.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from src.config import Settings, get_settings
from src.embedding.caption import Captioner, NullCaptioner, build_captioner
from src.embedding.image import CLIPImageEmbedder, ImageEmbedder
from src.embedding.text import (
    SentenceTransformerTextEmbedder,
    TextEmbedder,
    normalise_vectors,
)
from src.ingestion.elements import Element, ElementType, MatchedRepr

LOG = logging.getLogger(__name__)

TEXT_VECTOR = "text"
IMAGE_VECTOR = "image"
DEFAULT_IMAGE_DIM = 512  # CLIP ViT-B/32 projection; used when no figure is present
UPSERT_BATCH = 64


@dataclass(frozen=True)
class EmbeddingStats:
    """Counts from one embedding run, for logs and the CLI."""

    collection: str
    elements: int
    text_vectors: int
    image_vectors: int
    captions_added: int
    skipped: int
    model_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "collection": self.collection,
            "elements": self.elements,
            "text_vectors": self.text_vectors,
            "image_vectors": self.image_vectors,
            "captions_added": self.captions_added,
            "skipped": self.skipped,
            "model_version": self.model_version,
        }


def point_id(element_id: str) -> str:
    """Deterministic UUID for a Qdrant point from a 16-hex Element id."""
    try:
        number = int(element_id, 16)
    except ValueError:
        number = int.from_bytes(hashlib.sha1(element_id.encode()).digest()[:16], "big")
    return str(uuid.UUID(int=number & ((1 << 128) - 1)))


def text_representation(element: Element) -> str:
    """Which field is embedded in the ``text`` vector."""
    if element.element_type is ElementType.TABLE:
        return element.structured or element.text
    if element.element_type is ElementType.FIGURE:
        return element.caption or element.text
    return element.text


def collection_name(base: str, text_dim: int, image_dim: int) -> str:
    """Resolve the physical collection for a vector schema.

    Qdrant cannot resize an existing collection, and local mode cannot reliably
    delete and recreate one. Encoding the dimensions in the name makes a model
    change a new, deterministic collection instead of a destructive migration.
    Retrieval must resolve the same name from the same models.
    """
    return f"{base}_t{text_dim}i{image_dim}"


def _vector_config(text_dim: int, image_dim: int) -> dict[str, models.VectorParams]:
    return {
        TEXT_VECTOR: models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
        IMAGE_VECTOR: models.VectorParams(size=image_dim, distance=models.Distance.COSINE),
    }


def _ensure_collection(
    client: QdrantClient,
    name: str,
    text_dim: int,
    image_dim: int,
    *,
    config_mode: str = "embedded",
) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=_vector_config(text_dim, image_dim),
    )
    # Local Qdrant ignores payload indexes and warns on creation. They only
    # matter for the server mode, so keep embedded runs quiet.
    if config_mode == "server":
        for field, schema in (
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("element_type", models.PayloadSchemaType.KEYWORD),
            ("page", models.PayloadSchemaType.INTEGER),
        ):
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:  # indexes are an optimisation, not a contract
                LOG.debug("%s: no index for %s: %s", name, field, exc)


def _delete_documents(client: QdrantClient, collection: str, doc_ids: set[str]) -> None:
    for doc_id in sorted(doc_ids):
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )


def _payload(element: Element, model_version: str) -> dict[str, object]:
    payload = element.to_payload()
    has_caption = bool(element.caption)
    payload.update(
        {
            "repr_type": (
                MatchedRepr.CAPTION.value if has_caption else MatchedRepr.TEXT.value
            ),
            "has_image": (
                element.element_type is ElementType.FIGURE and element.image_path is not None
            ),
            "model_version": model_version,
        }
    )
    return payload


def embed_and_store(
    elements: list[Element],
    *,
    settings: Settings | None = None,
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
    captioner: Captioner | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> EmbeddingStats:
    """Caption, embed and upsert elements; return what was written.

    Heavy dependencies are injected by the caller or built lazily from
    ``settings``. An injected Qdrant client is not closed by this function; a
    client created here is closed on exit.
    """
    config = settings or get_settings()
    base = collection or config.qdrant_collection
    if not elements:
        return EmbeddingStats(base, 0, 0, 0, 0, 0, "none")

    owns_client = client is None
    store = client or QdrantClient(path=str(config.qdrant_path))
    text_model = text_embedder or SentenceTransformerTextEmbedder(
        config.embed_model, device=config.device
    )
    captions = captioner if captioner is not None else build_captioner(config)
    if captions is None:
        captions = NullCaptioner()

    figures = [element for element in elements if element.element_type is ElementType.FIGURE]
    image_model = image_embedder
    if image_model is None and figures:
        image_model = CLIPImageEmbedder(config.image_model, device=config.device)

    text_dim = text_model.dimension
    # The collection name must not depend on whether this particular batch
    # happens to contain a figure. Use the model's real dimension when known;
    # DEFAULT_IMAGE_DIM is the CLIP ViT-B/32 projection when no model is built.
    image_dim = image_model.image_dimension if image_model is not None else DEFAULT_IMAGE_DIM
    # Only tag the image model when it actually produced vectors.
    image_version = image_model.model_version if image_model is not None and figures else "none"
    model_version = f"{text_model.model_version}|{image_version}"
    name = collection_name(base, text_dim, image_dim)

    try:
        _ensure_collection(store, name, text_dim, image_dim, config_mode=config.qdrant_mode)

        prepared: list[Element] = []
        captions_added = 0
        for element in elements:
            if (
                element.element_type in (ElementType.FIGURE, ElementType.TABLE)
                and not element.caption
                and element.image_path is not None
            ):
                described = captions.describe(element.image_path, element.doc_id)
                if described:
                    element = element.model_copy(update={"caption": described})
                    captions_added += 1
            prepared.append(element)

        text_inputs = [
            (index, text_representation(element))
            for index, element in enumerate(prepared)
            if text_representation(element).strip()
        ]
        text_vectors = (
            text_model.embed([value for _, value in text_inputs]) if text_inputs else None
        )
        if text_vectors is not None:
            text_vectors = normalise_vectors(text_vectors)

        image_inputs: list[tuple[int, Path]] = []
        for index, element in enumerate(prepared):
            if element.element_type is ElementType.FIGURE and element.image_path is not None:
                image_inputs.append((index, element.image_path))
        image_vectors: NDArray[np.float32] | None = None
        if image_model is not None and image_inputs:
            image_vectors = normalise_vectors(
                image_model.embed_images([path for _, path in image_inputs])
            )

        by_index: dict[int, dict[str, list[float]]] = {}
        for slot, (index, _) in enumerate(text_inputs):
            assert text_vectors is not None
            by_index.setdefault(index, {})[TEXT_VECTOR] = text_vectors[slot].tolist()
        for slot, (index, _) in enumerate(image_inputs):
            assert image_vectors is not None
            by_index.setdefault(index, {})[IMAGE_VECTOR] = image_vectors[slot].tolist()

        skipped = [element for index, element in enumerate(prepared) if index not in by_index]
        if skipped:
            LOG.warning("%s: skipped %d element(s) with no representation", name, len(skipped))

        _delete_documents(store, name, {element.doc_id for element in prepared})
        written = 0
        indices = sorted(by_index)
        for start in range(0, len(indices), UPSERT_BATCH):
            batch = indices[start : start + UPSERT_BATCH]
            store.upsert(
                collection_name=name,
                points=[
                    models.PointStruct(
                        id=point_id(prepared[index].id),
                        vector=cast(models.VectorStruct, by_index[index]),
                        payload=_payload(prepared[index], model_version),
                    )
                    for index in batch
                ],
            )
            written += len(batch)

        stats = EmbeddingStats(
            collection=name,
            elements=written,
            text_vectors=len(text_inputs),
            image_vectors=len(image_inputs) if image_vectors is not None else 0,
            captions_added=captions_added,
            skipped=len(skipped),
            model_version=model_version,
        )
        LOG.info("%s: %s", name, stats.as_dict())
        return stats
    finally:
        if owns_client:
            store.close()


def load_manifest(path: Path) -> list[Element]:
    """Read a JSONL manifest produced by ``scripts/ingest.py``."""
    elements: list[Element] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            elements.append(Element.model_validate_json(line))
        except Exception as exc:  # pragma: no cover - defensive CLI guard
            raise ValueError(f"{path}:{line_number}: invalid Element: {exc}") from exc
    return elements
