"""Attack engine: connects to a target, sends payloads, manages rate limits."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .payloads import Technique


class TargetError(RuntimeError):
    """Raised when the target cannot be reached or rejects the harness."""


@dataclass(slots=True)
class Target:
    """Parsed target descriptor."""

    kind: str  # "openai" | "web" | "manual"
    url: str = ""
    model: str = ""
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Target":
        raw = settings.target.strip()
        if not raw or raw == "manual":
            return cls(kind="manual")
        if raw.startswith("openai://"):
            # openai://<optional-base-url>?model=name
            rest = raw[len("openai://"):]
            base, _, query = rest.partition("?")
            import urllib.parse
            params = dict(urllib.parse.parse_qsl(query))
            model = params.get("model") or settings.model
            need_default = not base or base == "auto"
            if need_default:
                api_base = "https://api.openai.com/v1"
            else:
                api_base = base.rstrip("/")
                # Allow a bare host[:port][/path] — default to http:// for local targets.
                if not api_base.startswith(("http://", "https://")):
                    api_base = "http://" + api_base
                if not api_base.rstrip("/").endswith("/v1") and "/v1/" not in api_base + "/":
                    api_base = api_base.rstrip("/") + "/v1"
            return cls(
                kind="openai",
                url=f"{api_base}/chat/completions",
                model=model,
                api_key=settings.api_key,
                headers={"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {},
            )
        if raw.startswith("http://") or raw.startswith("https://"):
            return cls(kind="web", url=raw, headers=dict(settings.extra_headers))
        raise TargetError(f"Unsupported target format: {raw!r}")


@dataclass(slots=True)
class AttackResult:
    technique_id: str
    technique_name: str
    category: str
    payload: str
    status: str  # "success" | "refused" | "error" | "unclear"
    score: float  # 0.0 .. 1.0 for success likelihood
    matched_signals: list[str] = field(default_factory=list)
    evidence: str = ""
    response_text: str = ""
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(
            technique_id=self.technique_id,
            technique_name=self.technique_name,
            category=self.category,
            payload=self.payload,
            status=self.status,
            score=self.score,
            matched_signals=self.matched_signals,
            evidence=self.evidence,
            latency_ms=round(self.latency_ms, 1),
            error=self.error,
        )


class AttackEngine:
    """Sends payloads against a Target and returns structured results."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.target = Target.from_settings(settings)
        self._client = httpx.AsyncClient(timeout=settings.timeout_seconds, follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, payload: str, history: list[dict[str, str]] | None = None,
                   temperature: float | None = None) -> AttackResult:
        """One atomic interaction: payload as the latest user turn."""
        if self.settings.dry_run:
            return self._dry_result(payload, history or [])
        if self.target.kind == "manual":
            raise TargetError("Target is 'manual' — drive it yourself and call /analyze with the reply.")
        started = time.monotonic()
        try:
            if self.target.kind == "openai":
                response_text, status_code = await self._chat_openai(payload, history or [], temperature)
            else:
                response_text, status_code = await self._chat_web(payload, history or [])
        except httpx.HTTPStatusError as exc:
            latency = (time.monotonic() - started) * 1000
            return AttackResult(
                technique_id="", technique_name="unknown", category="", payload=payload,
                status="error", score=0.0, error=f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                latency_ms=latency,
            )
        except httpx.HTTPError as exc:
            latency = (time.monotonic() - started) * 1000
            return AttackResult(
                technique_id="", technique_name="unknown", category="", payload=payload,
                status="error", score=0.0, error=f"transport error: {exc.__class__.__name__}: {exc}",
                latency_ms=latency,
            )
        latency = (time.monotonic() - started) * 1000
        return AttackResult(
            technique_id="", technique_name="unknown", category="", payload=payload,
            status="ok", score=0.0, response_text=response_text, latency_ms=latency,
            error="" if status_code < 400 else f"HTTP {status_code}",
        )

    async def _chat_openai(self, payload: str, history: list[dict[str, str]],
                           temperature: float | None = None) -> tuple[str, int]:
        messages = list(history)
        messages.append({"role": "user", "content": payload})
        body: dict[str, Any] = {
            "model": self.target.model or "gpt-4o",
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": 1024,
        }
        headers = {
            "Content-Type": "application/json",
            **self.target.headers,
        }
        resp = await self._client.post(self.target.url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = json.dumps(data, ensure_ascii=False)[: self.settings.max_response_chars]
        return text or "", resp.status_code

    async def _chat_web(self, payload: str, history: list[dict[str, str]]) -> tuple[str, int]:
        # Generic web target: POST JSON. The remote adapter decides the schema.
        body: dict[str, Any] = {
            "messages": [*history, {"role": "user", "content": payload}],
            "prompt": payload,
        }
        headers = {"Content-Type": "application/json", **self.target.headers}
        resp = await self._client.post(self.target.url, json=body, headers=headers)
        resp.raise_for_status()
        text = self._extract_web_text(resp)
        return text, resp.status_code

    def _extract_web_text(self, resp: httpx.Response) -> str:
        ct = resp.headers.get("content-type", "")
        if "json" in ct:
            data = resp.json()
            for key in ("response", "reply", "answer", "output", "completion", "result", "message"):
                if isinstance(data, dict) and key in data:
                    return str(data[key])
            if isinstance(data, dict) and "choices" in data:
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    pass
            return json.dumps(data, ensure_ascii=False)
        return resp.text[: self.settings.max_response_chars]

    def _dry_result(self, payload: str, history: list[dict[str, str]]) -> AttackResult:
        return AttackResult(
            technique_id="", technique_name="dry-run", category="",
            payload=payload, status="ok", score=0.0, response_text="[dry-run]",
            error="dry-run: no live request sent",
        )

    async def run_chain(self, chain: list[tuple[Technique, str]]) -> list[AttackResult]:
        """Send every payload in sequence. Rate-limit aware; safe by design."""
        results: list[AttackResult] = []
        for tech, payload in chain:
            res = await self.chat(payload)
            res.technique_id = tech.id
            res.technique_name = tech.name
            res.category = tech.category
            results.append(res)
            await asyncio.sleep(self.settings.delay_between_requests)
        return results

    async def run_multiturn(self, script) -> list[AttackResult]:
        """Execute a multi-turn TurnScript, threading the model's replies into ``{prev}``.

        Returns one AttackResult per turn so the analyzer can see how the attack
        escalated.  The conversation history is preserved across turns.
        """
        import re

        from .multiturn import TurnScript

        if not isinstance(script, TurnScript):
            raise TypeError("run_multiturn expects a TurnScript")

        history: list[dict[str, str]] = []
        results: list[AttackResult] = []
        previous_reply = ""
        for turn in script.turns:
            payload = re.sub(r"\{prev\}", previous_reply[:1500], turn) if "{prev}" in turn else turn
            res = await self.chat(payload, history=history)
            res.technique_id = f"{script.id}:t{len(results) + 1}"
            res.technique_name = script.name
            res.category = f"multiturn/{script.id}"
            results.append(res)
            history.append({"role": "user", "content": payload})
            if res.response_text:
                history.append({"role": "assistant", "content": res.response_text})
                previous_reply = res.response_text
            await asyncio.sleep(self.settings.delay_between_requests)
        return results