"""CLIP ViT-B/32 image and query-text embeddings (ADR-013).

Only FIGURE elements are image-embedded; tables keep caption + markdown. The
model is imported lazily so this module is safe to import in unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.embedding.text import normalise_vectors


def feature_array(output: Any) -> NDArray[np.float32]:
    """Extract a 2-D embedding from a CLIP feature call.

    transformers <5 returns a tensor directly. transformers 5.x returns a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds the projected
    (512-d for ViT-B/32) embedding. Accept both so the pinned runtime and older
    environments produce identical vectors.
    """
    tensor = output
    for attribute in ("image_embeds", "text_embeds", "pooler_output"):
        candidate = getattr(output, attribute, None)
        if candidate is not None:
            tensor = candidate
            break
    return np.asarray(tensor.cpu().numpy(), dtype=np.float32)


@runtime_checkable
class ImageEmbedder(Protocol):
    """Paired image/text towers in the same vector space."""

    model_version: str

    @property
    def image_dimension(self) -> int: ...

    @property
    def text_dimension(self) -> int: ...

    def embed_images(self, paths: list[Path]) -> NDArray[np.float32]: ...

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]: ...


class CLIPImageEmbedder:
    """Production CLIP embedder; weights load on first use."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        self.model_name = model_name
        self.model_version = f"clip:{model_name}"
        self.device = device
        # transformers ships no stable public types for these lazy objects and
        # its annotations vary by version, so this third-party boundary is Any.
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            from transformers import CLIPModel, CLIPProcessor

            # transformers 5.x ships an incorrect from_pretrained stub (it asks
            # for a PreTrainedModel where the API takes a model name). Treat the
            # lazily imported classes as Any at this boundary.
            model_cls: Any = CLIPModel
            processor_cls: Any = CLIPProcessor
            self._model = model_cls.from_pretrained(self.model_name).to(self.device)
            self._processor = processor_cls.from_pretrained(self.model_name)
            self._model.eval()
        return self._model, self._processor

    @property
    def image_dimension(self) -> int:
        model, _ = self._load()
        return int(model.config.projection_dim)

    @property
    def text_dimension(self) -> int:
        return self.image_dimension  # CLIP projects both towers to one space

    def embed_images(self, paths: list[Path]) -> NDArray[np.float32]:
        if not paths:
            return np.zeros((0, self.image_dimension), dtype=np.float32)
        import torch
        from PIL import Image

        model, processor = self._load()
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(self.device)
            features = model.get_image_features(**inputs)
        return normalise_vectors(feature_array(features))

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.text_dimension), dtype=np.float32)
        import torch

        model, processor = self._load()

        with torch.no_grad():
            inputs = processor(
                text=list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            features = model.get_text_features(**inputs)
        return normalise_vectors(feature_array(features))
