# AGENTS.md — Embedding module

Scoped instructions for `src/embedding/` (singular, per root `AGENTS.md`).
Root file applies; where conflict, root wins.

## Responsibility
Turn `Element` objects into vectors + descriptions and write them to the vector
store with full metadata. Offline (batch) stage. Decisions: ADR-010 (caption
gate), ADR-013 (representation per type), ADR-015 (Qdrant schema and write
ownership).

## Files
- `store.py` — **entry point**:
  `embed_and_store(elements, *, settings=None, text_embedder=None,
  image_embedder=None, captioner=None, client=None, collection=None)
  -> EmbeddingStats`. Owns the collection schema and all writes; also
  `load_manifest(path)` for the JSONL from `scripts/ingest.py`, `point_id()` and
  `collection_name()` helpers.
- `text.py` — sentence-transformers text embeddings behind the `TextEmbedder`
  protocol; `normalise_vectors()` is shared with the image path.
- `image.py` — CLIP ViT-B/32 paired towers behind `ImageEmbedder`;
  `feature_array()` bridges the transformers <5 tensor and 5.x output object.
- `caption.py` — `Captioner` protocol, `NullCaptioner`, `HostedCaptioner`,
  `CaptionCache` (sha256 of crop bytes), `CachedCaptioner`, `build_captioner()`.
- `__init__.py` — re-exports the three public store functions.

Driven by `scripts/embed.py` (`--dry-run` validates without loading models,
`--no-captions` keeps only PDF captions).

## Rules
1. **Representations differ by type** (ADR-013). A `FIGURE` gets a VLM caption
   (stored in `element.caption`, embedded as text) **and** a CLIP image
   embedding. A `TABLE` gets markdown in `element.structured` and is **not**
   image-embedded — the crop is glyphs the markdown already encodes. Text uses
   `element.text`. See `text_representation()` for the exact mapping.
2. **Caption before embed**, cached by image hash so unchanged crops are not
   re-described. Hosted captioning requires `CAPTION_ALLOW_HOSTED=true` *and* a
   configured key (root §9); otherwise keep PDF captions. `local-llava` must
   fail loudly, never silently fall back to a hosted API.
3. **Payload must include** for citation/rerank/UI: `doc_id`, `page`, `bbox`,
   `element_type`, `text`, `structured`, `caption`, `image_path`, `id`,
   `repr_type`, `has_image`, `model_version`. Payloads are built from
   `Element.to_payload()` — do not redeclare the schema here.
4. **Model version** on every vector; `model_version` joins text and image
   versions. L2-normalise every vector at the boundary.
5. **Idempotent**: point ids are deterministic UUIDs from `Element.id`; all
   points for a `doc_id` are deleted before upsert. A model/dimension change
   writes to a **new** collection `<base>_t<text_dim>i<image_dim>` instead of
   resizing — old collections remain on disk until a cleanup policy exists.
   `EmbeddingStats.collection` is the hand-off value for retrieval.
6. **Injection over import**: all heavy models and the Qdrant client are
   optional arguments. An injected client is not closed by the function; a
   client created here is closed in `finally`.
7. **Third-party boundaries are `Any`** where transformers/sentence-transformers
   annotations are unstable across versions. Never spread `type: ignore`
   through the module; use one local alias with a comment explaining why.
8. Keep logs at stage level with counts; never log full document text or crop
   bytes.

## Testing
- `tests/test_embedding.py` uses fake embedders/captioners and an embedded
  Qdrant — no weights downloaded, no hosted calls.
- Assert named-vector dimensions, payload/citation fields, caption caching,
  delete-before-upsert, empty-representation skipping, and the caption gate.
- Run `pytest -q tests/test_embedding.py`; the full suite must stay green.
