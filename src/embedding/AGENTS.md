# AGENTS.md — Embedding module

Scoped instructions for `src/embedding/` (singular, per root `AGENTS.md`).
Root file applies; where conflict, root wins.

## Responsibility
Turn `Element` objects into vectors + descriptions and write them to the vector
store with full metadata. Offline (batch) stage.

## Files (per root §4)
- `store.py` — entry point `embed_and_store(elements: list[Element]) -> None`.
- `text.py` — sentence-transformers text embeddings.
- `image.py` — CLIP ViT-B/32 image embeddings, FIGURE elements only (ADR-013).
- `caption.py` — VLM captions for figures; cached by image hash.

## Rules
1. **Representations differ by type** (ADR-013). A `FIGURE` gets a VLM caption
   (stored in `element.caption`, embedded as text) **and** a CLIP image
   embedding. A `TABLE` gets a caption plus markdown in `element.structured`
   and is **not** image-embedded — a table crop is glyphs the markdown already
   encodes. Tag payload `repr_type in {"text","image","caption"}`.
2. **Text elements**: embed `element.text` (or `structured` for tables); keep
   raw text in payload for prompt assembly.
3. **Caption before embed**; cache captions by image hash so unchanged crops
   are not re-described.
4. **Payload must include** for citations/rerank/UI: `doc_id, page, bbox, type,
   text, caption, image_path, repr_type, model_version`.
5. **Model version tag** on every vector; re-embed on model change must
   overwrite, not append.
6. **Captioning is hosted by default** (ADR-010): `CAPTION_PROVIDER=openai`
   with a small vision model, cached by image hash. Local LLaVA is opt-in for
   sensitive documents or offline runs. Hosted captioning must be refused
   unless the document set is explicitly approved (root §9).
7. **Idempotent**: delete old vectors for a `doc_id` before writing new ones.
8. **L2-normalize** vectors so cosine similarity is consistent.

## Testing
- Vector dims match model; payload fields present; caption cached on 2nd call;
  old vectors removed on re-ingest. `pytest tests/test_embedding.py`
