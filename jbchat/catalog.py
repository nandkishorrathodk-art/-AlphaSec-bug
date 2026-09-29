"""Catalog of the world's prompt-injection / LLM-red-teaming ecosystem.

Every entry records what the project is, where it lives, its licence, and how it
relates to a bug-bounty workflow.  The catalog is metadata only — JB-Chat never
auto-installs or auto-runs a third-party tool; :mod:`jbchat.integrations` only
builds the command for tools the operator already has on their machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExternalTool:
    """A third-party tool/framework/dataset relevant to prompt-injection work."""

    id: str
    name: str
    kind: str  # "scanner" | "framework" | "guardrail" | "benchmark" | "dataset"
    repo: str
    description: str
    license: str = ""
    install: str = ""
    cli: str = ""  # executable name used by `which` detection, when applicable
    tags: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Offensive testing frameworks / scanners
# ---------------------------------------------------------------------------

TOOLS: tuple[ExternalTool, ...] = (
    ExternalTool(
        id="garak",
        name="Garak (NVIDIA)",
        kind="scanner",
        repo="https://github.com/NVIDIA/garak",
        description=(
            "Open-source LLM vulnerability scanner. generator/probe/detector architecture with "
            "120+ probe modules covering DAN jailbreaks, encoding attacks, prompt injection, "
            "glitch tokens and training-data leakage."
        ),
        license="Apache-2.0",
        install="pip install -U garak",
        cli="garak",
        tags=("breadth", "cli", "probes", "detectors"),
    ),
    ExternalTool(
        id="pyrit",
        name="PyRIT (Microsoft)",
        kind="framework",
        repo="https://github.com/Azure/PyRIT",
        description=(
            "Python Risk Identification Toolkit. Orchestrator-driven red teaming with multi-turn "
            "crescendo and Tree-of-Attacks, converter chains for obfuscation, and multi-modal support."
        ),
        license="MIT",
        install="pip install pyrit",
        cli="",
        tags=("multi-turn", "orchestrator", "crescendo", "converters"),
    ),
    ExternalTool(
        id="promptfoo",
        name="Promptfoo",
        kind="framework",
        repo="https://github.com/promptfoo/promptfoo",
        description=(
            "CLI/library for LLM evaluation and red teaming. YAML-driven configs map findings to "
            "OWASP LLM Top 10 and run as a CI/CD regression gate against app/agent layers (RAG, tools, MCP)."
        ),
        license="MIT",
        install="npm install -g promptfoo",
        cli="promptfoo",
        tags=("ci-cd", "yaml", "owasp", "regression"),
    ),
    ExternalTool(
        id="giskard",
        name="Giskard",
        kind="framework",
        repo="https://github.com/Giskard-AI/giskard",
        description=(
            "Testing framework for ML/LLM systems: scans for prompt injection, hallucination, "
            "bias and robustness issues with 40+ vulnerability probes."
        ),
        license="Apache-2.0",
        install="pip install giskard",
        cli="giskard",
        tags=("quality", "bias", "robustness"),
    ),
    ExternalTool(
        id="deepteam",
        name="DeepTeam",
        kind="framework",
        repo="https://github.com/confident-ai/deepteam",
        description=(
            "Open-source red-teaming framework from the DeepEval team, with 40+ vulnerability "
            "probes and OWASP Top 10 for LLMs / Agentic mappings."
        ),
        license="Apache-2.0",
        install="pip install deepteam",
        cli="",
        tags=("python", "owasp", "probes"),
    ),
    ExternalTool(
        id="easyjailbreak",
        name="EasyJailbreak",
        kind="framework",
        repo="https://github.com/EasyJailbreak/EasyJailbreak",
        description=(
            "Modular jailbreak-attack library implementing 17+ attacks (GCG, AutoDAN, PAIR, "
            "ReNeLLM, …) with a uniform interface for research benchmarking."
        ),
        license="MIT",
        install="pip install easyjailbreak",
        cli="",
        tags=("attacks", "research", "gcg", "pair"),
    ),
    ExternalTool(
        id="promptmap",
        name="promptmap",
        kind="scanner",
        repo="https://github.com/utkusen/promptmap",
        description=(
            "Automated prompt-injection scanner for custom LLM applications. Tests both system-prompt "
            "hijacking and user-prompt injection."
        ),
        license="MIT",
        install="pip install promptmap2",
        cli="",
        tags=("app-layer", "automated"),
    ),
    ExternalTool(
        id="llm-attacks",
        name="llm-attacks (AdvBench / GCG)",
        kind="framework",
        repo="https://github.com/llm-attacks/llm-attacks",
        description=(
            "Reference implementation of GCG adversarial suffixes and the AdvBench harmful-behaviour "
            "dataset used by many jailbreak papers."
        ),
        license="MIT",
        install="",
        cli="",
        tags=("gcg", "suffixes", "advbench"),
    ),
    ExternalTool(
        id="autodan",
        name="AutoDAN",
        kind="framework",
        repo="https://github.com/SheltonLiu-N/AutoDAN",
        description=(
            "Generates semantically meaningful, stealthy jailbreak prompts with a hierarchical "
            "genetic algorithm — built to survive perplexity filters that catch GCG suffixes."
        ),
        license="MIT",
        install="",
        cli="",
        tags=("genetic", "stealthy", "perplexity"),
    ),
    ExternalTool(
        id="jailbreak-llms",
        name="Jailbreaking-LLMs (PAP / persuasive)",
        kind="framework",
        repo="https://github.com/CHATS-lab/persuasive_jailbreaker",
        description=(
            "Persuasive Adversarial Prompts (PAP): 40 persuasion strategies that reached ~92% ASR on "
            "GPT-4, plus the persuasion taxonomy paper."
        ),
        license="MIT",
        install="",
        cli="",
        tags=("persuasion", "pap", "social-engineering"),
    ),
    ExternalTool(
        id="deepinception",
        name="DeepInception",
        kind="framework",
        repo="https://github.com/tmlr-group/DeepInception",
        description=(
            "Official code for nested-fiction ('hypnosis') jailbreaking (EMNLP 2024, arXiv:2311.03191); "
            "shows how deeply layered scenarios bypass role-play defences."
        ),
        license="MIT",
        install="",
        cli="",
        tags=("nested-fiction", "roleplay"),
    ),
    ExternalTool(
        id="renellm",
        name="ReNeLLM",
        kind="framework",
        repo="https://github.com/NJUNLP/ReNeLLM",
        description=(
            "Generalised nested jailbreak prompts (NAACL 2024, 'A Wolf in Sheep's Clothing') that "
            "restructure the request and wrap it in benign-looking code/scenario tasks."
        ),
        license="MIT",
        install="",
        cli="",
        tags=("nested", "restructure", "code-task"),
    ),
    ExternalTool(
        id="awesome-jailbreak",
        name="Awesome-Jailbreak-on-LLMs",
        kind="dataset",
        repo="https://github.com/yueliu1999/Awesome-Jailbreak-on-LLMs",
        description=(
            "Continuously updated index of jailbreak papers, code, datasets and evaluations — the best "
            "single place to track new attacks (TAP, PAP, IRIS, QROA, GPTFuzzer, …)."
        ),
        license="See repo",
        install="",
        cli="",
        tags=("index", "papers", "surveys"),
    ),
    ExternalTool(
        id="gptfuzzer",
        name="GPTFuzzer",
        kind="framework",
        repo="https://github.com/sherdencooper/GPTFuzz",
        description=(
            "Fuzzing framework that auto-generates jailbreak prompts by mutating a seed corpus, "
            "analogous to AFL-style fuzzing applied to LLM policy."
        ),
        license="MIT",
        install="",
        cli="",
        tags=("fuzzing", "mutation", "automated"),
    ),
    ExternalTool(
        id="mcp-scan",
        name="MCP-Scan (Invariant Labs)",
        kind="tool",
        repo="https://github.com/invariantlabs-ai/mcp-scan",
        description=(
            "Security scanner for MCP/agentic systems: detects tool poisoning, rug pulls and "
            "prompt-injection in tool descriptions by static config analysis + dynamic monitoring."
        ),
        license="Apache-2.0",
        install="pip install mcp-scan",
        cli="mcp-scan",
        tags=("agentic", "mcp", "tool-poisoning", "scanner"),
    ),
    ExternalTool(
        id="mcp-injection-experiments",
        name="MCP Injection Experiments",
        kind="dataset",
        repo="https://github.com/invariantlabs-ai/mcp-injection-experiments",
        description=(
            "Reference implementations of MCP tool-poisoning, shadowing and sleeper rug-pull "
            "attacks (e.g. WhatsApp takeover) — the canonical PoCs to reproduce."
        ),
        license="MIT",
        install="",
        cli="",
        tags=("agentic", "mcp", "poc"),
    ),
)

# ---------------------------------------------------------------------------
# Guardrails / defensive tools (useful to see what a target may deploy)
# ---------------------------------------------------------------------------

GUARDRAILS: tuple[ExternalTool, ...] = (
    ExternalTool(
        id="llm-guard",
        name="LLM Guard (Protect AI)",
        kind="guardrail",
        repo="https://github.com/protectai/llm-guard",
        description=(
            "35+ input/output scanners for prompt injection, PII, toxicity and secrets. Useful both "
            "as a defense reference and to predict which payloads an in-scope target filters."
        ),
        license="MIT",
        install="pip install llm-guard",
        cli="",
        tags=("input-scanner", "output-scanner", "pii"),
    ),
    ExternalTool(
        id="rebuff",
        name="Rebuff (Protect AI)",
        kind="guardrail",
        repo="https://github.com/protectai/rebuff",
        description="Self-hardening prompt-injection detector combining heuristics, an LLM detector and a vector DB.",
        license="MIT",
        install="pip install rebuff",
        cli="",
        tags=("detector", "self-hardening"),
    ),
    ExternalTool(
        id="nemo-guardrails",
        name="NeMo Guardrails (NVIDIA)",
        kind="guardrail",
        repo="https://github.com/NVIDIA/NeMo-Guardrails",
        description="Programmable dialog policies via the Colang DSL; used to constrain bot behaviour at runtime.",
        license="Apache-2.0",
        install="pip install nemoguardrails",
        cli="nemoguardrails",
        tags=("colang", "runtime", "policy"),
    ),
    ExternalTool(
        id="vigil",
        name="Vigil",
        kind="guardrail",
        repo="https://github.com/deadbits/vigil-llm",
        description="Detection engine for prompt injection, jailbreaks and malicious content, with YARA-style rules.",
        license="Apache-2.0",
        install="pip install vigil-llm",
        cli="vigil",
        tags=("detector", "yara"),
    ),
    ExternalTool(
        id="owasp-top10-llm",
        name="OWASP Top 10 for LLM Applications",
        kind="framework",
        repo="https://genai.owasp.org/llm-top-10/",
        description=(
            "The de-facto reference taxonomy. LLM01 Prompt Injection, LLM02 Sensitive Information "
            "Disclosure, LLM07 System Prompt Leakage. Use it to map findings to a shared language."
        ),
        license="CC-BY-SA-4.0",
        install="",
        cli="",
        tags=("taxonomy", "reporting", "owasp"),
    ),
    ExternalTool(
        id="mitre-atlas",
        name="MITRE ATLAS",
        kind="framework",
        repo="https://atlas.mitre.org/",
        description=(
            "Adversarial Threat Landscape for AI Systems. Relevant techniques: AML.T0051.000 "
            "(direct prompt injection), AML.T0051.001 (indirect), AML.T0054 (LLM jailbreak)."
        ),
        license="MITRE",
        install="",
        cli="",
        tags=("taxonomy", "reporting", "mitre"),
    ),
)

# ---------------------------------------------------------------------------
# Benchmarks & datasets
# ---------------------------------------------------------------------------

BENCHMARKS: tuple[ExternalTool, ...] = (
    ExternalTool(
        id="jailbreakbench",
        name="JailbreakBench",
        kind="benchmark",
        repo="https://github.com/JailbreakBench/jailbreakbench",
        description=(
            "NeurIPS 2024 benchmark. JBB-Behaviors (200 misuse + benign behaviours) plus a "
            "leaderboard of attacks (PAIR, GCG, random search) and defenses."
        ),
        license="MIT",
        install="pip install jailbreakbench",
        tags=("leaderboard", "neurips", "judges"),
    ),
    ExternalTool(
        id="harmbench",
        name="HarmBench",
        kind="benchmark",
        repo="https://github.com/centerforaisafety/HarmBench",
        description="Standardized evaluation framework for automated red teaming and robust refusal (Center for AI Safety).",
        license="MIT",
        install="",
        tags=("evaluation", "refusal"),
    ),
    ExternalTool(
        id="hackaprompt",
        name="HackAPrompt",
        kind="dataset",
        repo="https://github.com/hackaprompt/hackaprompt",
        description="600k+ adversarial prompts from a global prompt-hacking competition, with difficulty levels.",
        license="MIT",
        install="",
        tags=("dataset", "competition"),
    ),
    ExternalTool(
        id="qualifire-pi",
        name="Qualifire Prompt-Injection Benchmark",
        kind="dataset",
        repo="https://huggingface.co/datasets/qualifire/Qualifire-prompt-injection-benchmark",
        description="5,000-row labelled benchmark (~60% benign / 40% jailbreak); low label noise, mostly English.",
        license="See dataset card",
        install="",
        tags=("dataset", "labelled", "huggingface"),
    ),
    ExternalTool(
        id="mosscap",
        name="Lakera Mosscap",
        kind="dataset",
        repo="https://huggingface.co/datasets/Lakera/mosscap_prompt_injection",
        description="280k-row multilingual prompt-injection corpus from an LLM red-team CTF (unlabelled).",
        license="See dataset card",
        install="",
        tags=("dataset", "multilingual", "huggingface"),
    ),
    ExternalTool(
        id="advbench",
        name="AdvBench",
        kind="dataset",
        repo="https://github.com/llm-attacks/llm-attacks",
        description="520 harmful behaviours used as the objective set in many jailbreak papers (part of llm-attacks).",
        license="MIT",
        install="",
        tags=("dataset", "harmful-behaviours"),
    ),
)

GROUPS: dict[str, tuple[ExternalTool, ...]] = {
    "tools": TOOLS,
    "guardrails": GUARDRAILS,
    "benchmarks": BENCHMARKS,
    "all": TOOLS + GUARDRAILS + BENCHMARKS,
}


def all_tools() -> tuple[ExternalTool, ...]:
    return GROUPS["all"]


def get_tool(tool_id: str) -> ExternalTool | None:
    for tool in all_tools():
        if tool.id == tool_id:
            return tool
    return None


def render_catalog(group: str = "all") -> str:
    """Pretty markdown/plain listing for the CLI and chat UI."""
    tools = GROUPS.get(group, GROUPS["all"])
    lines = [f"### External ecosystem — {group}", ""]
    for tool in tools:
        meta = f"[{tool.kind}"
        if tool.license:
            meta += f", {tool.license}"
        meta += "]"
        lines.append(f"**{tool.name}** {meta}")
        lines.append(f"  {tool.description}")
        lines.append(f"  repo: {tool.repo}")
        if tool.install:
            lines.append(f"  install: `{tool.install}`")
        lines.append("")
    return "\n".join(lines)