# AGENTS.md — Evaluation module

Scoped instructions for `src/evaluation/`. Root `AGENTS.md` applies.

## Responsibility
Measure retrieval and answer quality on a labeled set; document failures.
This rigor is the project's differentiator.

## Files (per root §4)
- `metrics.py` — retrieval + generation metrics.
- `runner.py` — **entry point**: `run_eval() -> EvalReport`.
- `docs/evaluation.md` and `docs/failure-taxonomy.md` (root docs) — write-ups.

## Rules
1. **Labeled eval set** in `data/eval_set/`: each item = question + correct
   element id(s) + expected answer keywords. Aim for ≥20 questions, several
   requiring a figure/table to answer.
2. **Retrieval metrics**: recall@k, MRR, and **image recall@k** (did the correct
   figure/table appear in top k). Report separately by element type.
3. **Answer metrics**: faithfulness (is every claim supported by retrieved
   evidence), answer relevancy, abstention correctness. Use RAGAS where it
   applies; custom checks for image grounding.
4. **Deterministic**: set seeds, record model versions + config hash with every
   run; write report to `data/eval_set/latest_report.json` (gitignored or
   committed per user choice).
5. **Bypass cache** during eval (`CACHE_ENABLED=false`) — measure true pipeline.
6. **Failure analysis**: classify every wrong case into the taxonomy in
   `docs/failure-taxonomy.md` (e.g., bad chunking, OCR error, retrieval miss,
   lost-in-the-middle, caption detail loss, hallucination). Update counts.
7. Do not tune on the test set hidden aside; keep a held-out split if adding
   more than 30 questions.

## Testing
- Metrics compute correctly on a tiny hand-made set; runner produces a report.
- `pytest tests/test_evaluation.py`
