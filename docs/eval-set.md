# Building the eval set

The eval set is the only artifact that decides whether the system is good. It
is human-authored source data, committed to `data/eval_set/questions.jsonl`.
Generated reports are not committed. Metrics definitions and honesty rules are
in [`evaluation.md`](evaluation.md); this guide is only about authoring labels.

## Required composition

- **At least 20 questions** across the documents in `data/raw/`.
- **At least 5 answerable only from a figure or table.** A text-only system
  must fail these. Write them from the crop or the markdown, not from prose.
- **At least 5 unanswerable questions** (`answerable: false`) to measure
  abstention. They must be plausible, not obviously absurd.
- Cover different pages and element types; do not cluster every label on page 1.

## Format

One JSON object per line:

| Field | Type | Meaning |
|---|---|---|
| `qid` | string | Unique, stable id (`q1`, `q2`, …) |
| `question` | string | The exact question sent to the pipeline |
| `answerable` | boolean | `false` marks the abstention split |
| `relevant` | array | Zero or more `(doc_id, page, bbox)` labels |
| `relevant[].doc_id` | string | Slugified filename from ingestion |
| `relevant[].page` | integer | **1-based** page number |
| `relevant[].bbox` | array of 4 floats | `[x0, y0, x1, y1]` in page coordinates |
| `keywords` | array of strings | Short answer strings used for checking |

Illustrative shape only — **not real labels; do not commit these lines**:

```json
{"qid":"q1","question":"What value does the chart on page 3 report for Q3?","answerable":true,"relevant":[{"doc_id":"paper-a","page":3,"bbox":[72.0,240.0,480.0,300.0]}],"keywords":["12.4M"]}
{"qid":"q2","question":"What was the 2029 revenue forecast?","answerable":false,"relevant":[],"keywords":[]}
```

Use `.jsonl`, not `.json`: generated `.json` reports are gitignored, while
`questions.jsonl` is the committed source of truth.

## Finding a bbox

1. Run `python scripts/ingest.py data/raw` first.
2. Open `data/processed/elements.jsonl` and find the element that contains the
   answer (look at `text`, `structured` or `caption`).
3. Copy its `doc_id`, `page` and `bbox` into the label.

Do **not** write an `Element.id` into a label. Ids are content hashes computed
from `(doc_id, page, bbox)` and may change if an ingestion heuristic changes;
the runner resolves labels to ids at eval time (ADR-012).

For an unanswerable question, `relevant` must be empty.

## Checklist before committing

- [ ] Every `qid` is unique and questions are unambiguous.
- [ ] Counts meet the composition above (≥20 total, ≥5 figure/table-only, ≥5 unanswerable).
- [ ] Every `bbox` came from the generated manifest and is `[x0, y0, x1, y1]`.
- [ ] Every answerable question has at least one `relevant` label.
- [ ] Every unanswerable question has none.
- [ ] No source PDFs were copied into `data/eval_set/`.
- [ ] `CACHE_ENABLED=false python scripts/evaluate.py` resolves every label; fix unresolved ones before trusting a score.

Keep the first run as the baseline. Record regressions and new failure classes
in [`failure-taxonomy.md`](failure-taxonomy.md) with an example element id and
a count. Quote only numbers actually produced by the runner.
