# AGENTS.md — Ingestion module

Scoped instructions for `src/ingestion/`. Root `AGENTS.md` applies too; where
this conflicts with an older root version, follow the current root file.

## Responsibility
Turn raw PDFs into a clean list of `Element` objects, each visual element with
a crop on disk. This is the highest-risk stage: garbage here is unrecoverable
downstream. See ADR-009 (capability gate) and ADR-014 (v0 filters).

## Files
- `elements.py` — the canonical `Element` model (root §5, ADR-012).
- `pdf_parser.py` — **entry point**: `parse_pdf(path, *, processed_dir=None,
  crop_dpi=150, doc_id=None) -> list[Element]`. Orchestrates the other modules;
  never modifies the source PDF.
- `layout.py` — text blocks, vector/raster figure detection, caption pairing,
  header/footer removal, figure-panel merging. Pure PyMuPDF + regexes.
- `tables.py` — `find_tables()` to markdown with over-detection filters and
  conservative cross-page grouping.
- `crops.py` — renders bbox crops to `data/processed/<doc_id>/<id>.png` and sets
  `element.image_path`.
- `ocr.py` — **not implemented**. The probe found 0% text-less pages (ADR-009);
  PaddleOCR stays an unused extra for a future scanned corpus.

## Rules
1. **Keep figure + caption together.** A caption found in the PDF goes into the
   figure/table `caption` field and is not emitted as a duplicate TEXT element.
   An unmatched caption remains TEXT. Composite panels are merged into one
   figure before caption matching (ResNet p4, Transformer p4).
2. **Never split a table within a page.** A table spanning pages is emitted once
   per page and linked by `metadata.table_group = "<doc_id>:t<n>"`, with
   `continues=True` on every part except the last. Only link when each page has
   one table, the column count matches and the x-span is within 10pt — a false
   link is worse than a missing one. Retrieval must return the whole group.
3. **Store crops on disk**, referenced by `element.image_path`; never embed raw
   image bytes in data structures or vector-store payloads. Render from the
   page bbox with `get_pixmap(clip=bbox, dpi=150)`: most figures are vector
   drawings, so `get_images()` alone misses them. Re-ingest deletes orphaned
   crop files.
4. **Drop headers/footers/page numbers** from retrievable text, and do not emit
   text already carried by a table markdown or figure crop.
5. **OCR only when needed.** No OCR in v0.
6. **Preserve bbox** `(x0, y0, x1, y1)` in page coordinates — citations and UI
   highlights depend on it. `doc_id` is the slugified filename; the CLI rejects
   duplicate ids.
7. **Idempotent**: re-ingesting the same PDF produces the same ids and overwrites
   the same crops. `Element.id` is `sha1(f"{doc_id}:{page}:{bbox}")[:16]`.
8. Log per-page counts by element type and timing; never log full document text.

## Testing
- `tests/test_ingestion.py` uses the synthetic fixture from `tests/conftest.py`.
- Assert types, bbox validity, crop files, caption pairing, table filtering,
  cross-page grouping, idempotency and stale-crop cleanup.
- Run `pytest -q tests/test_ingestion.py`; full suite must stay green.
