# AGENTS.md — Tests module

Scoped instructions for `tests/`. Root `AGENTS.md` applies.

## Responsibility
Keep pipeline stages honest. Every module change needs a passing test.

## Files
- `test_ingestion.py`, `test_embedding.py`, `test_retrieval.py`,
  `test_generation.py`, `test_caching.py`, `test_evaluation.py`, `test_ui.py`.

## Rules
1. **Use fixtures, not real private PDFs.** Build a tiny synthetic PDF (one
   text block, one table, one figure) in `tests/fixtures/`. Never commit source
   documents from `data/raw/`.
2. **Mock external calls**: LLM, VLM, Redis, and Qdrant get fakes/stubs — no
   real API calls or network in unit tests.
3. **Test determinism**: set seeds; assert stable ordering where relevant.
4. **Cover the invariants**: figure+caption pairing, table not split, bbox
   preserved, idempotent ingest, cache invalidation, abstention.
5. Run `pytest -q` before declaring done; fix failures, don't skip them.

## Testing the tests
- Each test file maps to one `src/<module>/AGENTS.md` rules section.
