"""Defensive detectors for the injection surfaces JB-Chat's payloads exercise.

This is the blue-team half of the tool. Where ``payloads.py`` builds attacks and
``analyzer.py`` scores replies, this module inspects untrusted *input* (tool
descriptions, retrieved documents, user text) and flags instruction-injection
patterns before they reach a model.

Complements OWASP MCP03 (tool poisoning) and MCP06 (intent-flow subversion).

Nothing here calls a model or the network.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Unicode ranges abused to hide instructions from a human reviewer.
_ZERO_WIDTH = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
}
_BIDI_CONTROLS = {
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
}
_UNICODE_TAG_RANGE = range(0xE0000, 0xE0080)

# Phrases that signal an embedded instruction aimed at the model.
_INSTRUCTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I), "instruction override"),
    (re.compile(r"\bdisregard\s+(your\s+)?(previous|prior|system)\b", re.I), "instruction override"),
    (re.compile(r"\b(reveal|print|show|leak|expose)\b.{0,40}\b(system\s*prompt|instructions?|secrets?|credentials?)\b", re.I),
     "exfiltration ask"),
    (re.compile(r"\b(read|cat|open)\b.{0,40}(~/\.ssh|id_rsa|\.aws/credentials|\.env\b)", re.I), "sensitive-file ask"),
    (re.compile(r"\b(note|message|instruction|directive)s?\s+(to|for)\s+(the\s+)?(assistant|ai|model|agent)", re.I),
     "assistant-directed note"),
    (re.compile(r"</?(important|system|instruction|secret)s?>", re.I), "embedded directive tag"),
    (re.compile(r"\bdo\s+not\s+(tell|mention|inform|reveal\s+this\s+to)\b.{0,30}\b(user|human)\b", re.I),
     "concealment ask"),
    (re.compile(r"\b(send|transfer|pay|wire)\b.{0,40}\$?\d", re.I), "payment-shaped action"),
)

_SUSPICIOUS_TAGS = re.compile(r"</?(important|system|instruction|secret|admin)s?>", re.I)

_B64_TOKEN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


def _base64_findings(text: str) -> list[Finding]:
    """Decode base64 blobs and re-scan the plaintext for injected instructions."""
    findings: list[Finding] = []
    import base64 as _b64
    for m in _B64_TOKEN.finditer(text):
        token = m.group(0)
        try:
            decoded = _b64.b64decode(token + "=" * (-len(token) % 4)).decode("utf-8")
        except Exception:
            continue
        if not decoded.isprintable():
            continue
        if any(p.search(decoded) for p, _ in _INSTRUCTION_PATTERNS):
            findings.append(Finding("text", "base64-encoded instruction", "high",
                                    "hidden instruction smuggled inside a base64 blob",
                                    snippet=decoded[:80]))
    return findings


@dataclass
class Finding:
    surface: str
    kind: str
    severity: str
    detail: str
    snippet: str = ""


@dataclass
class ScanResult:
    clean: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def max_severity(self) -> str:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        best = "none"
        for f in self.findings:
            if order.get(f.severity, 0) > order[best]:
                best = f.severity
        return best


def _strip_hidden(text: str) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    zw = sum(1 for ch in text if ch in _ZERO_WIDTH)
    bidi = sum(1 for ch in text if ch in _BIDI_CONTROLS)
    tags = sum(1 for ch in text if ord(ch) in _UNICODE_TAG_RANGE)
    clean = "".join(
        ch for ch in text
        if ch not in _ZERO_WIDTH and ch not in _BIDI_CONTROLS and ord(ch) not in _UNICODE_TAG_RANGE
    )
    if zw:
        findings.append(Finding("text", "zero-width characters", "high",
                                f"{zw} zero-width char(s) hide text from human review"))
    if bidi:
        findings.append(Finding("text", "bidi control characters", "high",
                                f"{bidi} bidi control char(s) can reorder displayed text"))
    if tags:
        findings.append(Finding("text", "unicode tag characters", "high",
                                f"{tags} unicode-tag char(s) smuggled past a length filter"))
    return clean, findings


def _homoglyph_findings(text: str) -> list[Finding]:
    """Flag mixed-script words (e.g. Latin + Cyrillic) that read as one word."""
    findings: list[Finding] = []
    for word in re.findall(r"\b\w{4,}\b", text):
        scripts = {unicodedata.name(ch, "").split()[0] for ch in word if ch.isalpha()}
        scripts.discard("")
        if len(scripts) > 1 and {"LATIN", "CYRILLIC"} <= scripts:
            findings.append(Finding("text", "homoglyph mix", "medium",
                                    "mixed-script word can spoof a trusted token", snippet=word))
    return findings


def _instruction_findings(surface: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for pattern, label in _INSTRUCTION_PATTERNS:
        m = pattern.search(text)
        if m:
            findings.append(Finding(surface, label, "high" if surface != "user" else "medium",
                                    "text contains an instruction aimed at the model", snippet=m.group(0)))
    return findings


def scan_text(text: str, surface: str = "user") -> ScanResult:
    """Scan untrusted text (user msg, tool description, retrieved doc)."""
    findings: list[Finding] = []
    visible, hidden = _strip_hidden(text)
    findings += hidden
    findings += _homoglyph_findings(visible)
    findings += _instruction_findings(surface, visible)
    findings += _base64_findings(visible)
    if surface == "tool-description" and _SUSPICIOUS_TAGS.search(visible):
        findings.append(Finding(surface, "tool-description poisoning", "high",
                                "tool description carries an embedded directive"))
    return ScanResult(clean=not findings, findings=findings)


def scan_tool_descriptions(tools: dict[str, str]) -> ScanResult:
    findings: list[Finding] = []
    for name, desc in tools.items():
        res = scan_text(desc, surface="tool-description")
        for f in res.findings:
            f.detail = f"[{name}] {f.detail}"
        findings += res.findings
    return ScanResult(clean=not findings, findings=findings)


def scan_retrieved(text: str) -> ScanResult:
    """Scan retrieved context (RAG) for indirect injection."""
    return scan_text(text, surface="retrieved")


def sanitize(text: str) -> str:
    """Remove hidden/control characters; keep the visible text intact."""
    visible, _ = _strip_hidden(text)
    return visible


_ZERO_WIDTH_BITS = {"\u200b": "0", "\u200c": "1"}


def _recover_zero_width(window: str) -> str:
    bits = "".join(_ZERO_WIDTH_BITS[ch] for ch in window if ch in _ZERO_WIDTH_BITS)
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        out.append(int(bits[i:i + 8], 2))
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def deobfuscate(text: str) -> str:
    """Best-effort reversal of the obfuscation *this toolkit* generates.

    Recovers text hidden with the zero-width bit channel and decodes base64 blobs so
    the result can be compared or scanned as plaintext. Not a general de-obfuscator.
    """
    import base64 as _b64
    recovered: list[str] = []
    window: list[str] = []
    for ch in text:
        if ch in _ZERO_WIDTH_BITS or ch in _ZERO_WIDTH:
            window.append(ch)
            continue
        if window:
            recovered.append(_recover_zero_width("".join(window)))
            window = []
        recovered.append(ch)
    if window:
        recovered.append(_recover_zero_width("".join(window)))
    visible = "".join(recovered)

    def _try(match: re.Match) -> str:
        token = match.group(0)
        try:
            decoded = _b64.b64decode(token + "=" * (-len(token) % 4)).decode("utf-8")
            return decoded if decoded.isprintable() else token
        except Exception:
            return token

    return _B64_TOKEN.sub(_try, visible)


def report(result: ScanResult) -> str:
    if result.clean:
        return "clean: no injection patterns detected"
    lines = [f"{len(result.findings)} finding(s), max severity {result.max_severity}:"]
    for f in result.findings:
        tail = f"  e.g. {f.snippet!r}" if f.snippet else ""
        lines.append(f"  [{f.severity}] {f.surface}/{f.kind}: {f.detail}{tail}")
    return "\n".join(lines)
