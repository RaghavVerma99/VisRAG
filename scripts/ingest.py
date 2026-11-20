"""Ingest a directory of PDFs into Elements plus PNG crops.

Usage:
    python scripts/ingest.py data/raw
    python scripts/ingest.py data/raw --out data/processed/elements.jsonl

The manifest is JSONL, one Element per line, keyed by its content-derived id.
Nothing is written into the source directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from src.config import get_settings
from src.ingestion import parse_pdf
from src.ingestion.layout import derive_doc_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/raw", help="directory of source PDFs")
    parser.add_argument("--out", default="data/processed/elements.jsonl", help="JSONL manifest")
    parser.add_argument("--processed-dir", default=None, help="root for crop files")
    parser.add_argument("--dpi", type=int, default=150, help="crop resolution")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    # PyMuPDF prints a suggestion on every find_tables() call; it is noise here.
    logging.getLogger("pymupdf").setLevel(logging.WARNING)

    root = Path(args.root)
    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {root}", file=sys.stderr)
        return 2

    processed_dir = (
        Path(args.processed_dir) if args.processed_dir else get_settings().data_processed
    )
    seen: dict[str, Path] = {}
    for path in pdfs:
        doc_id = derive_doc_id(path)
        if doc_id in seen:
            print(
                f"Duplicate doc_id {doc_id!r}: {seen[doc_id]} and {path}. "
                "Rename the files; ids must be unique for citations.",
                file=sys.stderr,
            )
            return 2
        seen[doc_id] = path

    manifest: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for path in pdfs:
        elements = parse_pdf(path, processed_dir=processed_dir, crop_dpi=args.dpi)
        counts.update(element.element_type.value for element in elements)
        manifest.extend(element.model_dump(mode="json") for element in elements)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(pdfs)} PDFs -> {len(manifest)} elements")
    print(", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    print(f"manifest: {out}")
    print(f"crops:    {processed_dir}/<doc_id>/<element_id>.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
