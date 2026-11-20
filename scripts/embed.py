"""Embed ingested elements into Qdrant (offline batch stage).

Usage:
    python scripts/embed.py
    python scripts/embed.py --manifest data/processed/elements.jsonl --no-captions
    python scripts/embed.py --dry-run   # validate the manifest only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import get_settings
from src.embedding.caption import build_captioner
from src.embedding.store import embed_and_store, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/elements.jsonl")
    parser.add_argument("--collection", default=None, help="override QDRANT_COLLECTION")
    parser.add_argument("--no-captions", action="store_true", help="keep PDF captions only")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate manifest, no models/vectors"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Root stays at WARNING: model hubs, urllib3 and PIL log every request and
    # PNG chunk at INFO/DEBUG. Only our own stages are allowed through.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    logging.getLogger("src").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    path = Path(args.manifest)
    if not path.is_file():
        print(f"Manifest not found: {path}. Run scripts/ingest.py first.", file=sys.stderr)
        return 2
    try:
        elements = load_manifest(path)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.dry_run:
        from collections import Counter

        print(f"{len(elements)} elements: {dict(Counter(e.element_type.value for e in elements))}")
        print("manifest OK; no models loaded")
        return 0

    settings = get_settings()
    try:
        captioner = build_captioner(settings, enabled=not args.no_captions)
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        stats = embed_and_store(
            elements,
            settings=settings,
            captioner=captioner,
            collection=args.collection,
        )
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
