"""Runtime configuration for JB-Chat."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Central settings, overridable through environment variables."""

    # Where findings are persisted.
    findings_dir: str = field(default_factory=lambda: os.environ.get("JB_FINDINGS_DIR", "findings"))

    # Default target. Format: ``openai://`` (auto-reads env API key) or a raw HTTP URL.
    target: str = field(default_factory=lambda: os.environ.get("JB_TARGET", ""))

    # Model name/handle used when talking to OpenAI-compatible endpoints.
    model: str = field(default_factory=lambda: os.environ.get("JB_MODEL", ""))

    # Default objective for manual/copilot work.
    goal: str = field(default_factory=lambda: os.environ.get("JB_GOAL", "system prompt"))

    # Extra request headers, e.g. ``X-API-Key: abc,JB-Bot: 1`` (for generic web targets).
    extra_headers: dict[str, str] = field(default_factory=lambda: _parse_headers(os.environ.get("JB_HEADERS", "")))

    # API key for OpenAI-compatible endpoints (defaults to OPENAI_API_KEY / ANTHROPIC_API_KEY).
    api_key: str = field(
        default_factory=lambda: os.environ.get("JB_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )

    # Request / response budget.
    timeout_seconds: float = field(default_factory=lambda: float(os.environ.get("JB_TIMEOUT", "60")))
    max_response_chars: int = field(default_factory=lambda: int(os.environ.get("JB_MAX_RESPONSE", "8000")))

    # Sampling. ``chat()`` uses temperature; differential probes stay near-deterministic so the
    # A/B difference reflects the *defence*, not sampling noise.
    temperature: float = field(default_factory=lambda: float(os.environ.get("JB_TEMPERATURE", "0.7")))
    probe_temperature: float = field(
        default_factory=lambda: float(os.environ.get("JB_PROBE_TEMPERATURE", "0.0")))

    # Responsible-testing knobs.
    delay_between_requests: float = field(default_factory=lambda: float(os.environ.get("JB_DELAY", "0.5")))
    max_concurrent: int = field(default_factory=lambda: int(os.environ.get("JB_CONCURRENCY", "1")))
    dry_run: bool = field(default_factory=lambda: _env_bool("JB_DRY_RUN", False))
    authorized_only: bool = field(default_factory=lambda: _env_bool("JB_AUTHORIZED_ONLY", True))

    # Web UI bind address.
    host: str = field(default_factory=lambda: os.environ.get("JB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("JB_PORT", "12000")))


def _parse_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for pair in raw.split(","):
        if ":" in pair:
            key, _, value = pair.partition(":")
            headers[key.strip()] = value.strip()
    return headers