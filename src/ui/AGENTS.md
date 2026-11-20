# AGENTS.md — UI module

Scoped instructions for `src/ui/`. Root `AGENTS.md` applies.

## Responsibility
Present answers with visible, clickable evidence (the design/trust half).

## Files
- `app.py` — Streamlit app: query box, answer with citations, retrieved
  text passages and figure/table images.

## Rules
1. **Show actual image crops**, never "[image]" placeholders. Render from
   `element.image_path` beside the answer.
2. **Clickable citations** `(doc_id, page, bbox)` link to the evidence panel;
   highlight the bbox on the rendered page where feasible.
3. **Honest confidence**: display a clear low-confidence / "I don't know" state.
4. **Read-only** — never send, post, or modify documents.
5. **Alt text / captions** for every image (accessibility).
6. **Show retrieval trace optionally** (which elements were retrieved + scores)
   for transparency.
7. Keep evidence expandable — show top few by default, not everything at once.

## Testing
- Stub answer + elements render image paths and citation labels; app starts.
