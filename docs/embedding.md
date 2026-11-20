# Embedding: operator's guide

Embedding turns the ingested `Element` manifest into vectors in Qdrant. It is an
offline batch stage: source PDFs and crops are read-only, and the index can be
rebuilt from `data/processed/elements.jsonl`.

Code: [`src/embedding/`](../src/embedding/). Decisions: ADR-010, ADR-013 and
ADR-015 in [`DECISIONS.md`](../DECISIONS.md). Architecture:
[`architecture.md`](architecture.md). Run ingestion first — see
[`ingestion.md`](ingestion.md).

## Run it

```bash
python scripts/embed.py --dry-run     # validate manifest; no models, no writes
python scripts/embed.py --no-captions # embed with PDF captions only
python scripts/embed.py               # also describe uncaptioned crops (gated)
```

| Output | Location |
|---|---|
| Vectors + payloads | embedded Qdrant at `QDRANT_PATH` (default `.qdrant/`, gitignored) |
| Caption cache | `data/processed/caption_cache.json`, keyed by sha256 of crop bytes |
| Run stats | JSON on stdout: collection, element/vector counts, model version |

The CLI refuses to send crops to a hosted VLM unless `CAPTION_ALLOW_HOSTED=true`
**and** `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` is set. This is a hard safety gate
(root §9), not a warning. With no approval it keeps captions found in the PDF
and logs that fact. `CAPTION_PROVIDER=local-llava` currently fails loudly — it
does not silently go hosted.

## Storage layout

One collection per vector schema, named `<base>_t<text_dim>i<image_dim>`:

| Named vector | Source | Dimension (default models) |
|---|---|---|
| `text` | TEXT: `text`; TABLE: `structured` markdown; FIGURE: `caption` | 384 (all-MiniLM-L6-v2) |
| `image` | FIGURE crop only (ADR-013) | 512 (CLIP ViT-B/32) |

Payloads carry the full citation triplet (`doc_id`, `page`, `bbox`),
`element_type`, `text`, `structured`, `caption`, `image_path`, `repr_type`,
`has_image` and `model_version`. Point ids are UUIDs derived from `Element.id`,
so re-running with the same models overwrites rather than appends. Old points
for each `doc_id` are deleted before the new batch is upserted.

**A model change creates a new collection** (e.g. `elements_v1_t768i512`). Old
collections are not deleted — there is no cleanup policy yet. Retrieval must
resolve the same name from the same models, or it will search the wrong index.
`EmbeddingStats.collection` is the authoritative value.

## Measured baseline

From the real run over the three probe PDFs (throughput facts, **not** retrieval
quality — the corpus has no relevance labels yet):

| Measure | Result |
|---|---|
| Points written | 816 |
| Text vectors | 814 |
| Image vectors | 12 (one per FIGURE) |
| Skipped elements | 0 |
| Captions added | 0 (run with `--no-captions`; no hosted approval) |
| Physical collection | `elements_v1_t384i512` |

Two figures have neither a PDF caption nor body text, so with captions disabled
they carry an image vector only. A caption-enabled run would give them a text
vector too.

## Compatibility notes

- **transformers 5.x** returns CLIP features in a model-output object
  (`pooler_output`); older versions return a tensor. `feature_array()` handles
  both. Verified against `transformers 5.23.0` and the pinned
  `openai/clip-vit-base-patch32` model.
- **sentence-transformers 6.x** renamed `get_sentence_embedding_dimension` to
  `get_embedding_dimension`; `text.py` prefers the new name with a fallback.
- PyTorch on Windows needs VC++ redistributable **14.40+** for wheels ≥2.15
  (see the Quickstart note in the README). The working environment used
  `torch 2.7.1+cpu`; `uv.lock` still pins 2.15 and needs the redist upgrade.

## Verify a change

```bash
pytest -q tests/test_embedding.py
ruff check . && mypy src scripts
python scripts/embed.py --dry-run
python scripts/embed.py --no-captions      # must report 816 points / 12 images
```

Tests use fake embedders and an in-process Qdrant: no weight downloads and no
network calls. To check the store directly:

```bash
python -c "from qdrant_client import QdrantClient; c=QdrantClient(path='.qdrant'); print(c.count('elements_v1_t384i512').count); c.close()"
```

## Known limits

- No BM25/sparse index is written yet; hybrid retrieval is a later stage.
- Cross-page `table_group` hints are stored in the payload but retrieval does
  not yet return the whole group.
- Caption quality is unmeasured, and the hosted captioner has not been run.
- Old dimension-specific collections accumulate; add a retention policy before
  running many model experiments.
- A failed run can leave partial points for a `doc_id`; the next successful run
  deletes and rewrites that `doc_id`, so recovery is a re-run.
