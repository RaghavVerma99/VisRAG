# Ingestion: operator's guide

Ingestion turns source PDFs into retrievable `Element` objects and PNG crops.
It is a read-only batch stage: source files are never modified, and every
output under `data/processed/` can be regenerated.

Code: [`src/ingestion/`](../src/ingestion/). Decisions: ADR-009 and ADR-014 in
[`DECISIONS.md`](../DECISIONS.md). Architecture context:
[`architecture.md`](architecture.md).

## Run it

```bash
python scripts/probe_corpus.py data/raw --json data/processed/corpus_probe.json
python scripts/ingest.py data/raw
```

Outputs:

| Path | Contents |
|---|---|
| `data/processed/elements.jsonl` | One `Element` per line, JSON-serialised |
| `data/processed/<doc_id>/<element_id>.png` | Crop for every FIGURE and TABLE |
| stdout | Per-page counts by element type, timing, and any rejected table |

`doc_id` is the lowercased, slugified filename (`Attention Is All You Need.pdf`
→ `attention-is-all-you-need`). The CLI stops if two files would produce the
same id, because citations must be unambiguous.

## What each stage does

| Stage | Rule | Why |
|---|---|---|
| Text | `get_text("blocks")`; drop blocks above 8% or below 92% of page height, and bare page numbers | Page furniture pollutes retrieval |
| Text | Skip text >50% covered by a table or figure bbox | The markdown or crop already carries it; no duplicate evidence |
| Tables | Keep regions ≥100pt wide, ≥20pt high, ≥2 non-empty rows and ≥2 columns | PyMuPDF over-detects ruled margins as tables |
| Tables | Empty rows removed; first row becomes markdown header | Keeps `structured` clean |
| Figures | Cluster vector drawings ≥8 paths; area ≥400pt²; ≤70% of page | Separates charts from rules and glyphs |
| Figures | Merge panels within 120pt horizontally / 60pt vertically | One caption can cover several panels (Transformer p4, ResNet p4) |
| Figures | Add raster image rectangles; skip ones already covered by a vector figure | Handles scanned-in charts without duplicates |
| Captions | `^(Figure\|Fig\.?\|Table)\s*\d+\b`, within 40pt of the target | Writes into `FIGURE.caption` / `TABLE.caption`, not a second TEXT element |
| Cross-page tables | Link only if each page has one table, same column count, x-span within 10pt | A false link would contaminate citations |
| Crops | `get_pixmap(clip=bbox, dpi=150)` to PNG | Most figures are vector drawings; `get_images()` misses them |
| Re-ingest | Ids and filenames reused; orphaned PNGs deleted | Idempotency (ADR-012 rule 7) |

Thresholds live in `src/ingestion/tables.py` (`MIN_TABLE_*`) and
`src/ingestion/layout.py` (`CAPTION_MAX_GAP`, `MIN_FIGURE_AREA`,
`FIGURE_MERGE_*`, fractions). Change them only with a measured reason and a new
ADR if the pipeline contract changes (root §7).

## Measured baseline

These numbers came from running the probe and ingestion over the three public
arXiv PDFs in `data/raw/`. They are corpus facts, **not evaluation metrics**.

| Measure | Result |
|---|---|
| Documents / pages | 3 / 46 |
| Text-less pages | 0 (0%) — OCR excluded from v0 |
| Raster images / vector-chart pages | 5 / 18 |
| Raw table regions / kept after filtering | 39 / 25 |
| Elements emitted | 816: 779 TEXT, 25 TABLE, 12 FIGURE |
| PDF captions attached | 10 of 12 figures, 18 of 25 tables |
| Cross-page table groups linked | 0 in this corpus (rule covered by a synthetic test) |

## Verify a change

```bash
pytest -q tests/test_ingestion.py   # fixture: text + table + figure + empty page
pytest -q                            # full suite
ruff check . && mypy src scripts
python scripts/ingest.py data/raw   # must complete without duplicate-id errors
```

To compare two ingests deterministically, diff the `id` column of
`elements.jsonl`; a heuristic change that mints new ids for unchanged pages is
a regression.

## Known limits

- Crop **visual quality** is not machine-verified. Open a few PNGs before
  treating a figure as eval evidence.
- The 25 kept tables are filtered heuristically, not human-labelled. The
  over-detection concern is reduced, not closed (README open decision #9).
- Figures and tables without a PDF caption stay uncaptioned until the VLM
  caption stage exists.
- No OCR and no LaTeX/formula extraction in v0. `ocr.py` is intentionally
  absent (ADR-009).
- A linked `table_group` must later be returned as a whole by retrieval; that
  stage is not implemented yet.
