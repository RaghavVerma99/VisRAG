# Evaluation

Goal: measure RAG quality with evidence, not vibes. Run offline with the cache
disabled so results reflect the real pipeline.

## Eval set
`data/eval_set/questions.jsonl` — one item per line. See
[`eval-set.md`](eval-set.md) for the authoring workflow and checklist.
```json
{"qid": "q1", "question": "What was Q3 revenue?", "answerable": true, "relevant": [{"doc_id": "acme-2025-q3", "page": 12, "bbox": [72.0, 240.0, 480.0, 300.0]}], "keywords": ["12.4M"]}
```
- Labels are `(doc_id, page, bbox)`, **not** element ids. Ids are content
  hashes (ADR-012) and hand-writing them is brittle; the runner resolves each
  label to an `Element.id` at eval time.
- `answerable: false` marks the abstention split.
- `k` comes from `TOP_K` (default 8) and is recorded in the report; recall is
  reported at `k` and at `k/2` so rank sensitivity is visible.
- ≥20 questions; several must require a figure/table (a text-only system fails them).
- Include a labeled "unanswerable" split to test abstention.

## Metrics (report separately)
- **Retrieval:** recall@k, MRR, and image recall@k (correct figure/table in top-k).
  Break down by element type (text vs table vs figure).
- **Generation:** faithfulness (are cited claims supported), citation correctness
  (do cited elements contain the answer), answer relevancy, abstention accuracy.

## Running
```bash
CACHE_ENABLED=false python scripts/evaluate.py
```
Outputs a timestamped report with model versions, config hash, and seeds.
Keep the first run as a baseline; every later run notes regressions/improvements.

## Honesty rules
- Quote only numbers actually produced by the runner; label any example data.
- Never tune on a held-out test split.
- Cache disabled during eval (semantic reuse would inflate scores).
