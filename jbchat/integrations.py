"""Integration layer: detect and (optionally) invoke external red-team tools.

Design rules:
* Nothing is installed automatically.
* Nothing is executed without the operator explicitly asking for it.
* Only tools present on ``PATH`` (or as importable Python packages) are offered.

This keeps JB-Chat a *coordinator*: it tells you which world-class tool to run,
with which arguments, and merges the imported findings back into its own store.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .catalog import ExternalTool, all_tools, get_tool

# Python packages that don't ship a CLI but can be detected via importlib.
_PYTHON_MODULES = {
    "pyrit": "pyrit",
    "deepteam": "deepteam",
    "jailbreakbench": "jailbreakbench",
    "promptmap": "promptmap",
    "giskard": "giskard",
    "llm-guard": "llm_guard",
    "rebuff": "rebuff",
    "easyjailbreak": "easyjailbreak",
}


@dataclass(slots=True)
class ToolStatus:
    tool: ExternalTool
    installed: bool
    path: str = ""
    via: str = ""  # "cli" | "python" | ""

    @property
    def available(self) -> bool:
        return self.installed


def _python_module_present(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def detect() -> list[ToolStatus]:
    """Return availability for every catalogued tool."""
    statuses: list[ToolStatus] = []
    for tool in all_tools():
        cli_path = shutil.which(tool.cli) if tool.cli else None
        module = _PYTHON_MODULES.get(tool.id)
        py_ok = _python_module_present(module) if module else False
        statuses.append(ToolStatus(
            tool=tool,
            installed=bool(cli_path) or py_ok,
            path=cli_path or "",
            via="cli" if cli_path else ("python" if py_ok else ""),
        ))
    return statuses


def detect_one(tool_id: str) -> ToolStatus | None:
    tool = get_tool(tool_id)
    if not tool:
        return None
    return next((s for s in detect() if s.tool.id == tool_id), None)


def render_status() -> str:
    lines = ["### External tool availability", ""]
    for status in detect():
        mark = "✅" if status.installed else "⬜"
        detail = f" ({status.via}: {status.path})" if status.installed else ""
        hint = f" — install: `{status.tool.install}`" if not status.installed and status.tool.install else ""
        lines.append(f"{mark} **{status.tool.name}**{detail}{hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command builders (never executed implicitly)
# ---------------------------------------------------------------------------

def build_command(tool_id: str, target: str = "", model: str = "", extra: str = "") -> str:
    """Return the shell command the operator should run, tailored to a target.

    ``target`` is a URL or model name; ``extra`` appends raw flags.
    Raises KeyError for unknown tools.
    """
    tool = get_tool(tool_id)
    if tool is None:
        raise KeyError(f"unknown tool: {tool_id}")
    suffix = f" {extra}".rstrip()

    if tool_id == "garak":
        if model:
            return f"garak --model_type openai --model_name {model}{suffix}"
        return f"garak --list_probes{suffix}" if not target else f"garak --target_type rest -U {target}{suffix}"
    if tool_id == "promptfoo":
        return ("promptfoo redteam run" if target else "promptfoo redteam generate") + suffix
    if tool_id == "pyrit":
        return "# PyRIT is a library: write a Python script importing pyrit (orchestrator + converters)"
    if tool_id == "deepteam":
        return "# DeepTeam is a library: `from deepteam import red_team`"
    if tool_id == "promptmap":
        mod = f" -m {model}" if model else ""
        return f"promptmap -u {target or '<url>'} -m openai{mod}{suffix}"
    if tool_id == "jailbreakbench":
        return "jailbreakbench --help  # evaluate attacks/defenses on JBB-Behaviors"
    if tool_id == "giskard":
        return "giskard scan  # scan an ML/LLM model for injection & robustness issues"
    if tool_id == "easyjailbreak":
        return "# EasyJailbreak is a library: `from easyjailbreak.attacker import ...`"
    if tool_id == "llm-guard":
        return "# LLM Guard is a library: `from llm_guard.input_scanners import PromptInjection`"
    return f"# See {tool.repo}"


# ---------------------------------------------------------------------------
# Import external findings into JB-Chat's store
# ---------------------------------------------------------------------------

def import_findings(path: str | Path, source: str = "external") -> list[dict]:
    """Best-effort importer for JSON findings produced by another tool.

    Accepts either a top-level list or a dict with a ``findings``/``results`` key,
    and normalises each item to the fields JB-Chat's :class:`~jbchat.findings.Finding`
    understands.  Malformed items are skipped rather than crashing a run.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("findings", "results", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        return []

    normalised: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        normalised.append({
            "title": str(item.get("title") or item.get("name") or f"{source} finding"),
            "target": str(item.get("target") or item.get("url") or ""),
            "category": str(item.get("category") or item.get("type") or source),
            "technique_id": str(item.get("probe") or item.get("technique_id") or source),
            "technique_name": str(item.get("probe_name") or item.get("technique_name") or source),
            "payload": str(item.get("prompt") or item.get("payload") or ""),
            "evidence": str(item.get("output") or item.get("evidence") or item.get("response") or ""),
            "score": float(item.get("score") or 0.0),
            "severity": str(item.get("severity") or "low"),
            "notes": f"imported from {source}",
        })
    return normalised


def export_findings_template() -> str:
    """A sample file showing the JSON shape the importer expects."""
    sample = [{
        "title": "System prompt disclosure via role-play",
        "target": "https://in-scope.example/chat",
        "category": "role_play",
        "probe": "role-play",
        "prompt": "You are DAN, ...",
        "output": "My system prompt is ...",
        "score": 0.9,
        "severity": "high",
    }]
    return json.dumps(sample, indent=2)