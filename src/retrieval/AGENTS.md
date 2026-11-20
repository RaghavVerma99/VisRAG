# AGENTS.md — Retrieval module

Scoped instructions for `src/retrieval/`. Root `AGENTS.md` applies.

## Responsibility
Given a user query, return the top-ranked `Element`s (text, tables, figures)
with scores. Online (per-query) stage. Quality here determines everything
downstream.

## Files
- `vector_store.py` — Qdrant read/search wrapper and collection resolution.
  The schema and all writes belong to embedding (ADR-015), not this module.
- `hybrid.py` — **entry point**: `retrieve(query: str, top_k: int) -> list[ScoredElement]`.
  Dense (text + CLIP text encoders) + sparse (BM25) search.
- `fusion.py` — reciprocal rank fusion (RRF) to merge ranked lists.
- `rerank.py` — multimodal reranker (VLM scores query↔image/text relevance).

## Rules
1. **Embed the query twice**: sentence-transformers for text slots, CLIP text
   encoder for image slots. Search both named vectors (`text` and `image`) in the
   **same** dimension-encoded collection resolved from the current models
   (ADR-015). The base name alone is not the physical collection.
2. **Hybrid = dense + sparse.** Always combine vector similarity with BM25
   keyword scores; fuse with RRF (`score = Σ 1/(k + rank)`, k≈60).
3. **Metadata filters** supported: `doc_id`, `element_type`, `page` range — expose
   as optional args to `retrieve`. Use the payload field names embedding writes
   (`element_type`, not `type`).
4. **Rerank after fusion**: top 2·top_k candidates go to the reranker; return
   final top_k. Reranker can be a cheap cross-encoder for text and a VLM for
   image crops.
5. **Deduplicate** by `Element.id` — the same element may appear from caption
   and image vectors; keep the highest score.
6. **Return score + payload**; the UI and generation need bbox/image_path.
7. **Log query, top scores, and element ids** (not full text) for eval/debug.
8. **Threshold**: if top score is below a calibrated minimum, return empty so
   generation can abstain ("I don't know") rather than fabricate.

## Testing
- Fixture index with known text+figure; assert correct element in top-3 for a
  targeted query; dedup works; low-score query returns empty.
- `pytest tests/test_retrieval.py`
