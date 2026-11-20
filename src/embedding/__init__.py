"""Embedding stage: Elements -> vectors in Qdrant.

Public entry point: :func:`src.embedding.store.embed_and_store`.
"""

from src.embedding.store import EmbeddingStats, embed_and_store, load_manifest

__all__ = ["EmbeddingStats", "embed_and_store", "load_manifest"]
