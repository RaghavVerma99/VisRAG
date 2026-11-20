"""Shared pytest fixtures.

`scripts/` is not an installable package, so it is loaded by path rather than
imported. This keeps the probe usable as a plain CLI script and testable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def probe_corpus() -> ModuleType:
    """The corpus probe, loaded from `scripts/probe_corpus.py`."""
    return _load_module("probe_corpus", ROOT / "scripts" / "probe_corpus.py")


@pytest.fixture(scope="session")
def fixture_pdf(tmp_path_factory: pytest.TempPathFactory, probe_corpus: ModuleType) -> Path:
    """A synthetic 2-page PDF: text + table + figure, then a text-less page."""
    out = tmp_path_factory.mktemp("fixtures")
    return probe_corpus.self_test(out)
