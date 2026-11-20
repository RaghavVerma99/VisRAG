"""Tests for :mod:`src.config`.

The important one is ``test_env_example_matches_settings_fields``: it makes the
template and the code impossible to drift apart, which is the whole point of
treating ``.env.example`` as the config contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from src.config import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]
SECRET_FIELDS = {"openai_api_key", "anthropic_api_key"}


def parse_env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def settings_from_example() -> Settings:
    return Settings(_env_file=None, **{k.lower(): v for k, v in parse_env_example().items()})


# ---------------------------------------------------------------------------
# Template / code agreement
# ---------------------------------------------------------------------------


def test_env_example_matches_settings_fields() -> None:
    documented = {key.upper() for key in parse_env_example()}
    declared = {name.upper() for name in Settings.model_fields}

    assert documented == declared, (
        f"missing from .env.example: {sorted(declared - documented)}; "
        f"documented but not a field: {sorted(documented - declared)}"
    )


def test_env_example_values_are_the_defaults() -> None:
    """Every value in the template must equal the code default, or drift is silent."""
    from_template = settings_from_example().model_dump(exclude=SECRET_FIELDS)
    defaults = Settings(_env_file=None).model_dump(exclude=SECRET_FIELDS)
    assert from_template == defaults


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def test_defaults_load_without_an_env_file() -> None:
    settings = Settings(_env_file=None)
    assert settings.top_k == 8
    assert settings.abstain_threshold == 0.30
    assert settings.qdrant_mode == "embedded"
    assert settings.cache_enabled is True


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("RERANK_CANDIDATES", "5")
    monkeypatch.setenv("CACHE_ENABLED", "false")

    settings = Settings(_env_file=None)
    assert settings.top_k == 3
    assert settings.cache_enabled is False


def test_server_mode_requires_a_url() -> None:
    with pytest.raises(ValidationError, match="QDRANT_URL"):
        Settings(_env_file=None, qdrant_mode="server")


def test_server_mode_accepts_a_url() -> None:
    settings = Settings(_env_file=None, qdrant_mode="server", qdrant_url="http://localhost:6333")
    assert settings.qdrant_mode == "server"


def test_rerank_candidates_must_cover_top_k() -> None:
    with pytest.raises(ValidationError, match="RERANK_CANDIDATES"):
        Settings(_env_file=None, top_k=10, rerank_candidates=5)


@pytest.mark.parametrize("field", ["abstain_threshold", "cache_similarity"])
def test_probability_bounds_are_enforced(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 1.5})


def test_settings_are_immutable() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.top_k = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_blank_keys_count_as_unconfigured() -> None:
    """An empty ``.env`` value must not masquerade as a working key."""
    settings = Settings(_env_file=None, openai_api_key="", anthropic_api_key="   ")
    assert settings.openai_api_key is None
    assert settings.is_vision_configured is False


def test_a_real_key_configures_vision() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    assert settings.is_vision_configured is True
    assert "sk-test" not in repr(settings)  # SecretStr keeps it out of logs


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_config_hash_is_stable_and_config_sensitive() -> None:
    base = Settings(_env_file=None)
    assert base.config_hash() == Settings(_env_file=None).config_hash()
    changed = Settings(_env_file=None, top_k=3, rerank_candidates=16)
    assert base.config_hash() != changed.config_hash()


def test_config_hash_ignores_secrets() -> None:
    without = Settings(_env_file=None)
    with_key = Settings(_env_file=None, openai_api_key="sk-secret")
    assert without.config_hash() == with_key.config_hash()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
