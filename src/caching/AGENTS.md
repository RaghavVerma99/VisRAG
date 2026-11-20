# AGENTS.md — Caching module

Scoped instructions for `src/caching/`. Root `AGENTS.md` applies.

## Responsibility
Reuse expensive results across queries to cut latency/cost. Online stage, but
must not change answers' correctness.

## Files (per root §4)
- `semantic_cache.py` — embedding-keyed cache for answers (and optionally
  retrieval results). Redis in production, in-process dict for dev.

## Rules
1. **Semantic key**: key on the query embedding (cosine similarity ≥ threshold,
   e.g. 0.95), NOT exact string match — near-duplicate questions should hit.
2. **Also key on context hash**: include a hash of retrieved element ids and
   model versions. If the index or models change, old answers must miss. v0
   caches answers only, after retrieval (ADR-006); a retrieval-level cache is a
   later decision, not a v0 requirement.
3. **Don't cache abstentions incorrectly**: a "don't know" is fine to cache but
   must be invalidated when new documents are ingested.
4. **Invalidation**: bump a `cache_version` on every re-ingest / re-embed /
   model change; include it in the key. Provide `clear()` for tests.
5. **Never cache secrets or private data** in a shared Redis beyond the
   project's configured namespace; set TTL (e.g., 24h).
6. **Cache is transparent to eval**: `retrieve`/`generate` stay pure. Eval must
   bypass the cache (config flag `CACHE_ENABLED=false`) to measure true quality.
7. **Log hits/misses** for cost reporting.

## Testing
- Identical query hits; paraphrase hits above threshold; different context
  misses; version bump invalidates. `pytest tests/test_cache.py`
