# AGENTS.md — Ingestion module

Scoped instructions for `src/ingestion/`. Root `AGENTS.md` applies too; where
this conflicts with an older root version, follow the current root file.

## Responsibility
Turn raw PDFs (digital or scanned) into a clean list of `Element` objects, each
with a crop image on disk. This is the highest-risk stage: garbage here is
unrecoverable downstream.

## Files
- `elements.py` — the `Element` model (canonical schema, defined in root §5).
- `pdf_parser.py` — **entry point**: `parse_pdf(path: Path) -> list[Element]`.
  Renders pages with PyMuPDF and partitions with `unstructured`.
- `layout.py` — region/layout detection (LayoutLMv3 fallback for complex pages).
- `ocr.py` — PaddleOCR wrapper for scanned regions.
- `tables.py` — camelot extraction to markdown.
- `crops.py` — save bbox image crops to `data/processed`, set `element.image_path`.

## Rules
1. **Keep figure + caption together.** A figure and its caption must share
   `page` and have nearby `bbox`; never split them across documents/chunks.
2. **Never split a table within a page.** If a table spans pages, emit one
   `TABLE` element per page, each with its own bbox, and link them with
   `metadata.table_group = "<doc_id>:t<n>"`, plus `metadata.continues = True`
   on every part except the last. Retrieval returns the whole group together
   (root §5, ADR-012).
3. **Store crops on disk**, referenced by `element.image_path`; never embed raw
   image bytes in data structures or vector-store payloads. Render the crop from
   the page bbox (`get_pixmap(clip=bbox, dpi=150)`): most figures in the probe
   corpus are **vector drawings**, so `get_images()` alone misses them.
4. **Drop headers/footers/page numbers** from retrievable text — they pollute
   retrieval. Tag them or omit them.
5. **OCR only when needed.** Run PaddleOCR only for regions with no embedded
   text layer; avoid double-OCR on digital PDFs.
6. **Preserve bbox** (x0,y0,x1,y1) in page coordinates — citations and UI
   highlights depend on it.
7. **Idempotent**: re-ingesting the same PDF must not duplicate elements.
   `Element.id` is `sha1(f"{doc_id}:{page}:{bbox}")[:16]` (ADR-012); the store
   upserts on that id, so re-ingest overwrites instead of appending.
8. Log per-page counts by element type, OCR-used flag, and timing.

## Testing
- Fixture PDF with one text block, one table, one figure.
- Assert types, bbox presence, crop files exist, caption pairing, no dupes.
- `pytest tests/test_ingestion.py`
