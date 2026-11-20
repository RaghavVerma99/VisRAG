# AGENTS.md — Documentation

Scoped instructions for `docs/`. Root `AGENTS.md` applies.

## Responsibility
Keep architecture and evaluation write-ups accurate and in sync with code.

## Files
- `architecture.md` — component diagram, data flow, invariants.
- `ingestion.md` — operator's guide: stages, thresholds, measured baseline, limits.
- `embedding.md` — operator's guide: caption gate, collection naming, storage layout, limits.
- `evaluation.md` — methodology, metrics definitions, how to run, how to read results.
- `eval-set.md` — how to author and validate the committed label set.
- `failure-taxonomy.md` — classes of observed failures with counts and examples.

## Rules
1. Docs must match current code. After a pipeline change, update the relevant doc
   in the same change.
2. `failure-taxonomy.md` is a living document: every eval run that finds a new
   failure class gets a new entry with an example element id and a count.
3. No invented numbers — quote only metrics actually produced by
   `src/evaluation/runner.py`. Label any example data clearly.
4. Keep prose plain and concise; diagrams as fenced mermaid blocks.
