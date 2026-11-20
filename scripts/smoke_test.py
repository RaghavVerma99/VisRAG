"""Environment smoke test — run before blaming the pipeline.

Verifies the interpreter, the core imports, an embedded Qdrant round-trip, and
(optionally) a real embedding. Nothing here touches `data/raw`.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --with-models   # downloads the embedding model
    python scripts/smoke_test.py --json out.json

Exit code 0 only if every required check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_PYTHON = (3, 12)


@dataclass
class Result:
    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""


def _run(name: str, fn: Callable[[], str], *, required: bool = True) -> Result:
    try:
        return Result(name, "pass", fn() or "")
    except Exception as exc:  # the point is to report, not raise
        if not required:
            return Result(name, "skip", f"{type(exc).__name__}: {exc}")
        return Result(name, "fail", f"{type(exc).__name__}: {exc}")


def check_python() -> str:
    actual = (sys.version_info.major, sys.version_info.minor)
    if actual != REQUIRED_PYTHON:
        raise RuntimeError(
            f"expected Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}, "
            f"got {actual[0]}.{actual[1]} (see ADR-008)"
        )
    return f"{platform.python_version()} on {platform.platform()}"


def check_core_imports() -> str:
    import importlib

    for module in ("numpy", "pydantic", "pydantic_settings", "pymupdf", "PIL.Image"):
        importlib.import_module(module)

    import pydantic
    import pymupdf

    return f"pymupdf {pymupdf.__version__}, pydantic {pydantic.VERSION}"


def check_qdrant_roundtrip() -> str:
    """Upsert and search a vector in embedded mode — no server, no network."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    with tempfile.TemporaryDirectory() as tmp:
        client = QdrantClient(path=str(Path(tmp) / "qdrant"))
        try:
            client.create_collection(
                collection_name="smoke",
                vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            )
            client.upsert(
                collection_name="smoke",
                points=[
                    PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"kind": "a"}),
                    PointStruct(id=2, vector=[0.0, 1.0, 0.0, 0.0], payload={"kind": "b"}),
                ],
            )
            hits = client.query_points(
                collection_name="smoke", query=[0.9, 0.1, 0.0, 0.0], limit=1
            ).points
        finally:
            client.close()

    if not hits:
        raise RuntimeError("embedded search returned no results")
    payload = hits[0].payload or {}
    if payload.get("kind") != "a":
        raise RuntimeError(f"unexpected top hit: {payload}")
    return "embedded upsert + query ok"


def check_embedding() -> str:
    """Load the configured text model and embed one sentence."""
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    vector = model.encode(["smoke test"], normalize_embeddings=True)
    dims = int(vector.shape[1])
    if dims < 64:
        raise RuntimeError(f"suspicious embedding dimension: {dims}")
    return f"{model_name} -> {dims}-d"


def check_vision_key() -> str:
    """Generation is unusable without a key; report which one is configured."""
    present = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if os.environ.get(k)]
    if not present:
        raise RuntimeError("no vision API key in the environment (see .env.example)")
    return ", ".join(present)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-models",
        action="store_true",
        help="also load the embedding model (downloads weights on first run)",
    )
    parser.add_argument("--json", dest="json_path", help="write results as JSON")
    args = parser.parse_args()

    results: list[Result] = [
        _run("interpreter", check_python),
        _run("core imports", check_core_imports),
        _run("qdrant embedded", check_qdrant_roundtrip),
    ]

    if args.with_models:
        results.append(_run("text embedding", check_embedding))
    else:
        results.append(Result("text embedding", "skip", "pass --with-models to run"))

    results.append(_run("vision API key", check_vision_key, required=False))

    print("Smoke test\n" + "-" * 60)
    for result in results:
        print(f"  {result.status.upper():<5} {result.name:<18} {result.detail}")

    failures = [r for r in results if r.status == "fail"]
    print("-" * 60)
    print(f"{len(failures)} failed, {len(results) - len(failures)} ok")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )

    for failure in failures:
        print(f"\nFAILED: {failure.name} — {failure.detail}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
