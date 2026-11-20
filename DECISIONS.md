# DECISIONS.md — Architecture Decision Records (ADRs)

Format per ADR: **Context → Decision → Alternatives → Consequences**. Add a new
ADR when you change a design decision; never silently edit code past a decision.

## ADR-001: Use RAG, not fine-tuning
**Context:** Need to answer over private, updating PDFs.
**Decision:** Retrieval-augmented generation.
**Alternatives:** Fine-tuning a model on documents (stale, expensive, cannot cite).
**Consequences:** Answers stay current when docs change; retrieval quality
becomes the main engineering problem.

## ADR-002: Two representations for every visual element
**Context:** Figures/tables must be findable by wording and by appearance.
**Decision:** Each figure/table gets a VLM caption (embedded as text) AND a
CLIP/SigLIP image embedding.
**Alternatives:** Caption-only (loses detail), image-only (misses worded queries).
**Consequences:** Larger index, but covers both "revenue chart" and visual-similarity queries.
**Amended by ADR-013:** image embeddings apply to `FIGURE` only; tables keep a
caption plus markdown and are not image-embedded.

## ADR-003: Pass actual image crops to the generation LLM
**Context:** A caption alone cannot answer detailed chart/table questions.
**Decision:** Send image crops to a vision-capable LLM at generation time.
**Alternatives:** Text-only context (fails on visual facts).
**Consequences:** Requires a vision model; higher token cost; justified by correctness.

## ADR-004: Read-only first
**Context:** Risk of destructive side effects during development.
**Decision:** No external writes (email/post/pay/merge/delete). Ingestion writes
only to `data/processed` and the vector store.
**Alternatives:** Full agent with tools from day one.
**Consequences:** Safe to iterate; approval step added only when write tools are introduced.

## ADR-005: Local embeddings, API generation
**Context:** Balance cost, latency, privacy.
**Decision:** Local sentence-transformers + local LLaVA captions; hosted vision
LLM for final answers, behind an interface.
**Alternatives:** All-local (weaker answers) or all-API (cost + privacy).
**Consequences:** Swap points isolated in `src/embedding/` and `src/generation/`.
**Amended by ADR-010:** captioning is now hosted by default; local LLaVA is the
opt-in path for sensitive documents or offline runs.

## ADR-006: Semantic caching with version invalidation
**Context:** Repeated/paraphrased questions waste LLM calls.
**Decision:** Cache answers by query-embedding similarity + context hash +
`cache_version`; Redis beyond demo.
**Alternatives:** Exact-string cache (few hits), no cache (costly).
**Consequences:** Cheaper, faster; must tune similarity threshold to avoid wrong hits.

## ADR-007: Rigorous, separate evaluation
**Context:** RAG quality claims need evidence.
**Decision:** Report retrieval metrics (recall@k, MRR, image recall@k)
separately from answer metrics (faithfulness, abstention); maintain a failure
taxonomy; disable cache during eval.
**Alternatives:** A single demo accuracy number.
**Consequences:** More work, but the documented rigor is the differentiator.

---

## ADR-008: Packaging and interpreter
**Context:** The stack was declared "Python 3.11+" with "uv or poetry", which
blocks a reproducible install. `unstructured` declares `requires_python
<3.14` and `paddlepaddle` ships no cp314 wheel, so the only interpreter present
on the dev machine (3.14) cannot install the stack.
**Decision:** Pin `>=3.12,<3.13` in `pyproject.toml`. Use **uv** with a
committed `uv.lock`. Heavy dependencies are opt-in extras (`embed`, `store`,
`parse`, `tables`, `ocr`, `gen`, `ui`, `eval`) so a clean checkout installs
fast and CI stays cheap.
**Alternatives:** Unpinned `>=3.11` (unreproducible); poetry (equivalent, but
slower and already not installed); conda (extra toolchain, no benefit here).
**Consequences:** A 3.13+ interpreter fails loudly at install rather than
mysteriously at import. Upgrading Python becomes a deliberate, tested change.
On Windows, PyTorch's CPU wheels additionally require the Visual C++
2015–2022 redistributable **14.40+**; below that `import torch` fails with
`WinError 126` because `VCRUNTIME140_THREADS.dll` is absent. This is an
environment prerequisite, not a code dependency, so it is documented in the
README rather than encoded in `pyproject.toml`.

## ADR-009: PyMuPDF-first ingestion, heavy parsers capability-gated
**Context:** `unstructured`, `camelot` and `PaddleOCR` carry large dependency
surfaces and platform risk. The project has no measured evidence that the
corpus needs them.
**Decision:** v0 ingestion uses PyMuPDF alone: `get_text("blocks")` for text,
`find_tables()` for tables, `get_images()` + `get_drawings()` for figure
regions, `get_pixmap()` for crops. `unstructured`, `camelot` and `PaddleOCR`
stay installable as extras and are promoted into the default path only if
`scripts/probe_corpus.py` shows the corpus requires them (>20% text-less pages
for OCR; tables PyMuPDF cannot recover for camelot).
**Alternatives:** Build on `unstructured` first (fights the interpreter, opaque
failure modes); LayoutLMv3 (needs weights + GPU, unproven need).
**Consequences:** One dependency instead of four for the first working pipeline;
each escalation is justified by a measured corpus property, not a hunch.
PyMuPDF is additionally pinned to **1.x**: the 2.x table detector loads an ONNX
layout model at runtime, which crashed with an access violation on the dev
machine and adds an implicit model download plus per-page CPU cost to
ingestion. The 1.x `find_tables()` is geometric and needs nothing extra.
**Probe result (3 arXiv papers, 46 pages):** 0% text-less pages, so **PaddleOCR
is out of v0 entirely** — the largest single dependency removal available.
PyMuPDF `find_tables()` reported 39 table regions, high enough to suspect
over-detection, so camelot stays as the fallback pending a spot-check. There
were only 5 raster images but 18 pages carrying vector charts: most figures are
*drawn*, not embedded bitmaps, so figure crops must be rendered from the bbox
rather than sourced from `get_images()`.
`paddlepaddle` itself imports fine; `paddleocr` pulls `paddlex` → `torch` and
fails to import on the same VC++ runtime problem. The throwaway venv holding
those three candidates weighed **1.9 GB** — for capabilities this corpus does
not need. `unstructured[pdf]` and `camelot-py` both install and import cleanly
on Python 3.12, so they remain available as extras if a future corpus demands
them.

## ADR-010: Hosted captioning by default
**Context:** ADR-005 defaulted to local LLaVA. The target machine has no NVIDIA
GPU and 14 GB RAM, so CPU captioning of a real corpus is hours-to-days and
LLaVA-7B does not fit alongside the embedding stack.
**Decision:** Caption figures with a hosted VLM (`CAPTION_PROVIDER=openai`,
`gpt-4o-mini`) by default. Keep `local-llava` as an explicit opt-in for
sensitive documents or offline runs. Captions are cached by image hash so a
crop is described once.
**Alternatives:** Force local LLaVA (unusable here); no captions (loses the
worded-query half of ADR-002).
**Consequences:** Captioning is cheap and fast, but sends image crops to a
third party — so the §9 approval gate becomes a code path, not a note: the
pipeline refuses hosted captioning unless `CAPTION_ALLOW_HOSTED=true` is set
for that document set.

## ADR-011: Embedded Qdrant for v0
**Context:** Qdrant was named "primary, Chroma fallback" with no deployment
mode, which leaves tests and local runs ambiguous.
**Decision:** Use Qdrant in embedded/local mode (`QDRANT_MODE=embedded`,
on-disk at `.qdrant/`) for v0. Server mode is a config switch, not a rewrite.
**Alternatives:** Docker Compose from day one (slower inner loop, another
moving part); Chroma (second implementation to maintain for no benefit).
**Consequences:** Tests need no network and no container. The migration path to
server mode is a URL change plus a collection re-upsert.

## ADR-012: Canonical `Element` schema
**Context:** The spec named the primary key both `id` and `element_id`, carried
an `embedding` field that duplicates the vector store, used `type=CAPTION`
alongside a `caption` string, and left `ScoredElement`, `Citation` and `Answer`
undefined — so module boundaries did not typecheck against each other.
**Decision:** One schema, defined in `src/ingestion/elements.py`:
`Element(id, doc_id, page, element_type, bbox, text, structured, caption,
image_path, metadata)`.
- `id` = `sha1(f"{doc_id}:{page}:{bbox}")[:16]`. The name is `id`; `element_id`
  is gone.
- `element_type ∈ {TEXT, TABLE, FIGURE, FORMULA, HEADER, FOOTER}`. No `CAPTION`
  type: a caption is either `FIGURE.caption` or a `TEXT` element with
  `metadata.caption_of`.
- `bbox` is `(x0, y0, x1, y1)` in page coordinates; serialised as a 4-list.
- No `embedding` field. Vectors live in the vector store only.
- `ScoredElement = Element + score + matched_repr`; `Citation` and `Answer` are
  defined alongside so retrieval → generation is type-compatible.
**Alternatives:** Keep `embedding` on the model (numpy across a Pydantic
boundary, huge payloads); allow both id names (guaranteed bug).
**Consequences:** Every module imports one schema; the eval set can label by
`(doc_id, page, bbox)` and resolve to ids instead of hand-writing hashes.

## ADR-013: CLIP for figures only; tables stay textual
**Context:** "CLIP / SigLIP" was left open, but retrieval embeds queries with a
CLIP *text* encoder — SigLIP's text tower is not interchangeable in that flow.
Separately, ADR-002 implied every visual element gets an image embedding, yet a
table crop is mostly glyphs that the caption and markdown already encode.
**Decision:** Use **CLIP ViT-B/32** for both the image and query-text towers.
FIGURE elements get a caption + CLIP image embedding. TABLE elements get a
caption + markdown `structured` text, and are **not** image-embedded.
**Alternatives:** SigLIP (better zero-shot, but no drop-in text-conditioned
retrieval path); embedding table crops (index bloat, near-zero recall gain).
**Consequences:** One encoder pair, consistent vector space, smaller index.
Revisit SigLIP only if a paired text encoder is adopted at the same time.

## ADR-014: Ingestion v0 is PyMuPDF-only, with measured filters
**Context:** ADR-009 closed the capability gate (no OCR needed; 18 vector-chart
pages but only 5 raster images), yet the module specs still described
`unstructured`, LayoutLMv3, camelot and PaddleOCR as the v0 path. A first pass
over the probe corpus confirmed the risk: PyMuPDF reported 39 table regions in
46 pages, including many 48pt-wide ruled margins that are not tables, and split
one composite figure (ResNet p4) into five crops with a single caption.
**Decision:** Implement ingestion with pure PyMuPDF geometry:
- `doc_id` is the lowercased, slugified filename; the CLI rejects duplicate ids.
- Text comes from `get_text("blocks")`; headers, footers and bare page numbers
  are dropped. Text whose bbox is >50% inside a table/figure is not emitted
  twice — the markdown or crop carries it.
- Table candidates must be >=100pt wide, >=20pt high, >=2 non-empty rows and
  >=2 columns. Empty rows are removed before rendering markdown.
- Figures come from clustered vector drawings (>=8 paths, area >=400pt²,
  <=70% of the page) plus raster image rects. Panels within 120pt horizontally
  and 60pt vertically are merged into one figure, so a shared caption links to
  the whole figure (Transformer p4 panels are 100pt apart).
- A PDF caption matching `^(Figure|Fig\.?|Table)\s*\d+\b` is written into the
  target element's `caption`; it is not emitted as a second TEXT element. An
  unmatched caption stays as TEXT.
- Cross-page tables are linked only when each page has exactly one table, with
  the same column count and x-span within 10pt; parts get a shared
  `metadata.table_group` and `continues` on all but the last.
- Crops are rendered from the bbox with `get_pixmap(clip=..., dpi=150)` into
  `data/processed/<doc_id>/<element_id>.png`. Re-ingest overwrites the same
  files and deletes orphans; source PDFs are never modified.
**Alternatives:** `unstructured` partitioning and camelot extraction (both work
on Python 3.12 but add a dependency for no measured gain here); leaving the raw
39 table regions unfiltered (would put ruled margins into retrieval); emitting
caption TEXT elements separately (duplicate evidence).
**Consequences:** First real run over 3 papers / 46 pages produced **816
elements: 779 TEXT, 25 TABLE, 12 FIGURE**; 14 of 25 table candidates were
rejected by the size/shape filter. 10/12 figures and 18/25 tables have a PDF
caption; the rest rely on the later VLM caption pass. No cross-page table was
linked in this corpus — the conservative rule is exercised by a synthetic test
instead. Retrieval must return a linked `table_group` together (future stage).
The 25 kept tables are *not yet human-verified*; the earlier over-detection
concern is reduced, not closed.

## ADR-015: Qdrant schema, collection naming and write ownership
**Context:** ADR-011 chose embedded Qdrant. A probe of `qdrant-client` 1.20 in
local mode confirmed named vectors, filter deletes and UUID point ids work, but
a collection cannot be resized in place and delete/recreate during one process
left stale vector dimensions in memory. Model upgrades are expected (the text
embedder already renamed a method between v5 and v6), so the collection name
cannot be a fixed string. Retrieval must be able to resolve exactly the
collection that embedding wrote.
**Decision:** One physical collection per vector schema, named
`<base>_t<text_dim>i<image_dim>` (e.g. `elements_v1_t384i512`). It carries two
named vectors: `text` (sentence-transformers; tables use markdown, figures use
caption) and `image` (CLIP, FIGURE crops only). Point ids are deterministic
UUIDs derived from `Element.id`; re-ingesting a `doc_id` deletes its old points
before upserting, so runs are idempotent. Every payload carries the full
citation fields (`doc_id`, `page`, `bbox`, `element_type`), the raw text,
`repr_type`, `has_image`, and `model_version`. Embedding owns the schema and all
writes; retrieval only reads and searches. Payload indexes are created in server
mode only — local mode ignores them and warns. Heavy models sit behind injected
interfaces; captioning is gated behind `CAPTION_ALLOW_HOSTED=true` plus a key
(root §9), and `local-llava` fails loudly until implemented. Transformers 5.x
returns CLIP features in a model-output object (`pooler_output`); a
compatibility extractor accepts both that and the older tensor return.
**Alternatives:** A fixed collection name with delete/recreate (unreliable in
local mode and destructive); a manual migration step (easy to skip); one
collection per document (fragments cross-document queries).
**Consequences:** A model change writes to a *new* collection; old collections
remain on disk until a cleanup policy is added — retrieval must resolve the
same name from the same models or it will search an empty/stale collection.
`EmbeddingStats.collection` is the hand-off value. Two figures with neither PDF
caption nor text get an image vector only when captions are disabled. Real run
on the probe corpus: 816 points, 814 text vectors, 12 image vectors in
`elements_v1_t384i512`. The corpus has no human relevance labels yet, so these
are throughput facts, not retrieval-quality metrics.
