# Failure taxonomy

A living record of where the system breaks. Update after every eval run: add
new classes with a count, an example `qid`/element id, and a one-line cause.

## Classes
1. **Ingestion — OCR error**: garbled text on low-quality scans; causes wrong
   text embeddings. Example: _(none recorded yet)_.
2. **Ingestion — layout miss**: figure mis classified as text or caption split
   from its figure; visual evidence lost. Example: _(none recorded yet)_.
3. **Chunking — table split**: a table broken across chunks, partial data
   retrieved. Example: _(none recorded yet)_.
4. **Retrieval — wrong figure**: similar-looking charts, wrong one retrieved.
   Example: _(none recorded yet)_.
5. **Retrieval — recall miss**: correct element not in top-k (dense/sparse
   imbalance or reranker error). Example: _(none recorded yet)_.
6. **Lost in the middle**: correct evidence retrieved but ignored by the
   generator because it sat in the middle of a long prompt. Example: _(none recorded yet)_.
7. **Caption detail loss**: answer relied on a caption instead of the crop and
   missed a visual detail. Example: _(none recorded yet)_.
8. **Generation — hallucination**: claim not supported by any cited element.
   Example: _(none recorded yet)_.
9. **Citation error**: answer correct but citation points to the wrong element.
   Example: _(none recorded yet)_.
10. **Abstention failure**: answered confidently when it should have said "I
   don't know". Example: _(none recorded yet)_.
11. **Prompt injection**: text or a caption in a document contains directives
   that the model follows (e.g. "ignore previous instructions"). Detected by
   injection probes in the eval set; treated as a safety failure, not just a
   quality one. Example: _(none recorded yet)_.

## How to maintain
- Count failures per class in each eval report; link the report timestamp.
- Never invent counts — only record failures observed in a real eval run.
- When a class is fixed, note the fix and the ADR/change that addressed it.
