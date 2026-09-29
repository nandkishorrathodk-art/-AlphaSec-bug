"""Findings store: persist attack results and render bug-bounty style reports."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import AttackResult

_SEVERITY_LEVELS = ("informational", "low", "medium", "high", "critical")


@dataclass(slots=True)
class Finding:
    title: str
    target: str
    category: str
    technique_id: str
    technique_name: str
    payload: str
    evidence: str
    score: float
    severity: str = "low"
    status: str = "new"
    notes: str = ""
    created_at: str = field(default_factory=lambda: _now())
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _new_id()
        if self.severity not in _SEVERITY_LEVELS:
            self.severity = "low"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_result(cls, result: AttackResult, target: str) -> "Finding":
        impact = _infer_impact(result)
        return cls(
            title=f"[{result.category or 'injection'}] {result.technique_name or result.technique_id} — {impact}",
            target=target,
            category=result.category,
            technique_id=result.technique_id,
            technique_name=result.technique_name,
            payload=result.payload,
            evidence=result.evidence or result.response_text[:500],
            score=result.score,
            severity=_severity_from_score(result.score),
        )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return f"F-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{abs(hash(object())) % 9973:04d}"


def _infer_impact(result: AttackResult) -> str:
    ev = (result.evidence + " " + " ".join(result.matched_signals)).lower()
    if any(w in ev for w in ("secret", "api key", "token", "password", "credential")):
        return "credential / secret disclosure"
    if any(w in ev for w in ("system prompt", "developer instruction")):
        return "system prompt extraction"
    if any(w in ev for w in ("history", "previous message", "user data", "personal")):
        return "conversation / PII disclosure"
    if any(w in ev for w in ("tool", "function", "get_", "send_")):
        return "tool / internal function disclosure"
    if any(w in ev for w in ("admin", "sudo", "root", "privilege")):
        return "privilege / admin claim"
    return "policy bypass"


def _severity_from_score(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.7:
        return "medium"
    if score >= 0.5:
        return "low"
    return "informational"


class FindingsStore:
    """Loads/saves findings to ``findings_dir`` as JSON files (one per run)."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_run(self, findings: list[Finding], label: str = "run") -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.directory / f"{label}-{stamp}.json"
        path.write_text(json.dumps([f.to_dict() for f in findings], indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def list_runs(self) -> list[Path]:
        return sorted(self.directory.glob("*.json"), reverse=True)

    def load_run(self, path: str | Path) -> list[Finding]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [Finding(**item) for item in data]


REPORT_TEMPLATE = """# Bug Bounty Report

**Title:** {title}

**Target:** {target}  `({category})`

**Severity:** {severity}  
**Score (heuristic):** {score}  
**ID:** {id}  
**Date:** {created_at}

---

## 1. Description

An authorised prompt-injection test on the model powered by the target application.
Using technique `{technique_id}` ({technique_name}), an attacker-controlled message
was able to {impact_description}.

## 2. Steps to Reproduce (PoC)

1. Open the target application's chat interface.
2. Submit the following message as a user:
   ```
   {payload}
   ```
3. Observe the model's reply.

### Observed output (evidence)

```
{evidence}
```

## 3. Impact

- {impact_desc_bullet}

## 4. Recommended Remediation

- Treat user-provided content as untrusted data, never as instructions (robust
  instruction hierarchy / role separation).
- Sanitise and boundary-tag retrieved/indirect content before it reaches the model.
- Apply output filtering and monitor for structured secrets (API keys, JWTs, URLs).
- Run red-team regression suites from this report's technique family before release.

## 5. References

- OWASP Top 10 for LLM Applications — LLM01: Prompt Injection
- Guidelines from the target's disclosure / responsible-use policy
"""


def render_report(finding: Finding, impact_hint: str = "", extra_bullet: str = "") -> str:
    impact_text = impact_hint or _infer_impact_from_finding(finding)
    bullet = extra_bullet or f"The attacker could obtain or confirm {impact_text} without authorization."
    return REPORT_TEMPLATE.format(
        title=finding.title,
        target=finding.target,
        category=finding.category,
        severity=finding.severity,
        score=finding.score,
        id=finding.id,
        created_at=finding.created_at,
        technique_id=finding.technique_id,
        technique_name=finding.technique_name,
        payload=finding.payload,
        evidence=_truncate(finding.evidence, 1500),
        impact_description=impact_text,
        impact_desc_bullet=bullet,
    )


def _infer_impact_from_finding(finding: Finding) -> str:
    ev = (finding.payload + " " + finding.evidence).lower()
    if any(w in ev for w in ("secret", "api key", "token", "password", "credential")):
        return "disclose credentials or secrets"
    if "system prompt" in ev:
        return "extract the model's system prompt / developer instructions"
    if any(w in ev for w in ("history", "previous message", "personal", "pii")):
        return "disclose conversation history or personal data"
    return "bypass the model's policy boundaries in an unintended way"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"

# Re-export for convenience.
__all__ = ["Finding", "FindingsStore", "render_report", "SEVERITY_LEVELS", "AttackResult"]