"""Text embeddings (sentence-transformers) behind a small interface.

The model is loaded lazily so importing this module never downloads weights and
unit tests can inject a deterministic fake. All vectors are L2-normalised by
the embedder; :mod:`src.embedding.store` normalises again at the boundary.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


def normalise_vectors(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Row-wise L2 normalisation; zero rows stay zero."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (array / safe).astype(np.float32)


@runtime_checkable
class TextEmbedder(Protocol):
    """Minimal contract used by the store and by tests."""

    model_version: str

    @property
    def dimension(self) -> int:
        """Size of one embedding vector."""
        ...

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts in input order, L2-normalised."""
        ...


class SentenceTransformerTextEmbedder:
    """Production text embedder. Weights are loaded on first use."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.model_version = f"sentence-transformers:{model_name}"
        self.device = device
        self.batch_size = batch_size
        # sentence-transformers' annotations vary across versions; this is a
        # deliberately loose third-party boundary.
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        # v6 renamed this method; prefer the new name but stay compatible.
        getter = getattr(model, "get_embedding_dimension", None)
        if getter is None:
            getter = model.get_sentence_embedding_dimension
        return int(getter())

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return normalise_vectors(vectors)
