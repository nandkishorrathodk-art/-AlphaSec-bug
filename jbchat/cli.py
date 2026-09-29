"""Command-line front-end for JB-Chat."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .advisor import suggest_chain, suggest_goal
from .analyzer import analyze
from .bot import ChatBot
from .config import Settings
from .payloads import CATEGORIES, TECHNIQUES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jbchat",
        description="Prompt-injection research toolkit for authorised bug-bounty testing.",
        epilog="Only test targets you are explicitly authorised to test. Stay in scope.",
    )
    p.add_argument("--target", default=None,
                   help="Target: openai://[base]?model=NAME, http(s)://url, or 'manual'.")
    p.add_argument("--model", default=None, help="Model name for OpenAI-compatible endpoints.")
    p.add_argument("--goal", default=None, help="What to extract (e.g. 'system prompt', 'secrets', 'tools').")
    p.add_argument("--recon", default="", help="Recon notes; drive the technique chain.")
    p.add_argument("--no-plan", action="store_true",
                   help="Use the full technique library in default order instead of a suggested chain.")
    p.add_argument("--once", action="store_true", help="Send only the first payload of each technique.")
    p.add_argument("--list", action="store_true", help="Print the technique library and exit.")
    p.add_argument("--ecosystem", nargs="?", const="all", default=None,
                   help="List third-party tools: all (default), tools, guardrails, benchmarks.")
    p.add_argument("--detect", action="store_true", help="Show which external tools are installed.")
    p.add_argument("--tool", default=None, help="Show the command to run an external tool by id.")
    p.add_argument("--import-findings", default=None, metavar="PATH",
                   help="Import findings JSON produced by another tool.")
    p.add_argument("--multiturn", default=None, metavar="PLANNER",
                   help="Run a multi-turn plan (crescendo, skeleton-key, deceptive-delight, "
                        "bad-likert-judge, echo-chamber, context-fusion).")
    p.add_argument("--adaptive", default=None, choices=["best-of-n", "pair", "tap"],
                   help="Run an adaptive strategy against the target.")
    p.add_argument("--cipher", default=None,
                   help="Apply a cipher to the goal before sending (see --ciphers to list).")
    p.add_argument("--ciphers", action="store_true", help="List cipher/obfuscation transforms.")
    p.add_argument("--evasions", action="store_true",
                   help="List 2024-2025 classifier-evasion techniques and exit.")
    p.add_argument("--evade", default=None, metavar="EVASION_ID",
                   help="Apply a classifier-evasion transform to the goal (see --evasions).")
    p.add_argument("--plans", action="store_true", help="List multi-turn attack plans and exit.")
    p.add_argument("--agentic", nargs="?", const="all", default=None, metavar="PAYLOAD_ID",
                   help="Show agentic/MCP attack payloads (all, or a single payload id).")
    p.add_argument("--mcp-top10", action="store_true", help="List the OWASP MCP Top 10 and exit.")
    p.add_argument("--discover", action="store_true",
                   help="Autonomous closed-loop search for a novel payload against the target.")
    p.add_argument("--fingerprint", action="store_true",
                   help="Probe the target's defences and recommend an attack approach.")
    p.add_argument("--pipeline", action="store_true",
                   help="Print the modern training pipeline (per-stage refusal tells).")
    p.add_argument("--stage", default=None, metavar="REPLY_OR_STAGE_ID",
                   help="Infer the training stage behind a reply, or show a stage's strategy.")
    p.add_argument("--why", action="store_true",
                   help="Explain why a given attack dies at a given training stage.")
    p.add_argument("--diagnose", action="store_true",
                   help="Active differential probing: infer the target's defence from paired "
                        "experiments (needs a live target).")
    p.add_argument("--probes", action="store_true",
                   help="List the differential probe suite (use /probe in the TUI to run it "
                        "by hand against a manual target).")
    p.add_argument("--next", action="store_true",
                   help="Print the next payload to send to an external AI (Grok/ChatGPT), then "
                        "feed its reply back with --read. The hand-off loop, no live target needed.")
    p.add_argument("--read", default=None, metavar="FILE_OR_TEXT",
                   help="Read the external AI's reply (a file path or literal text) and propose "
                        "the next payloads, linked to the prompt you last took with --next.")
    p.add_argument("--target-name", default=None, metavar="NAME",
                   help="Name of the external chat for the hand-off loop (e.g. grok, chatgpt).")
    p.add_argument("--targets", action="store_true",
                   help="List the external hand-off targets you have open and exit.")
    p.add_argument("--generations", type=int, default=6, help="Discovery generations (default 6).")
    p.add_argument("--population", type=int, default=8, help="Discovery population size (default 8).")
    p.add_argument("--n", type=int, default=8, help="Sample budget for best-of-n (default 8).")
    p.add_argument("--iterations", type=int, default=5, help="Iteration cap for PAIR (default 5).")
    p.add_argument("--dry-run", action="store_true", help="Generate payloads but never send requests.")
    p.add_argument("--json", action="store_true", help="Emit results as JSON.")
    return p


def _library_text() -> str:
    lines: list[str] = []
    for cid, cname in CATEGORIES:
        techs = ", ".join(t.id for t in TECHNIQUES if t.category == cid)
        lines.append(f"- {cid} ({cname}): {techs}")
    return "\n".join(lines)


def _read_reply(value: str) -> str | None:
    """Accept a reply as a file path (if it exists) or as literal text."""
    import os

    if os.path.isfile(value):
        try:
            with open(value, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return None
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(_library_text())
        return 0

    if args.ecosystem is not None:
        from .catalog import render_catalog

        print(render_catalog(args.ecosystem))
        return 0

    if args.detect:
        from .integrations import render_status

        print(render_status())
        return 0

    if args.tool:
        from .integrations import build_command, detect_one

        status = detect_one(args.tool)
        if status is None:
            print(f"Unknown tool id: {args.tool}", file=sys.stderr)
            return 2
        state = f"installed ({status.via})" if status.installed else "not installed"
        print(f"# {status.tool.name} — {state}")
        print(build_command(args.tool, target=args.target or "", model=args.model or ""))
        return 0

    if args.import_findings:
        from .integrations import import_findings

        items = import_findings(args.import_findings)
        print(f"Imported {len(items)} findings from {args.import_findings}")
        for item in items:
            print(f"- [{item['severity']}] {item['title']}")
        return 0

    if args.evasions:
        from .evasion import EVASIONS

        for e in EVASIONS:
            asr = f" [{e.example_asr}]" if e.example_asr else ""
            print(f"- {e.id}: {e.name} — {e.defeats}{asr}\n    {e.source}")
        return 0

    if args.ciphers:
        from .ciphers import CIPHERS

        for c in CIPHERS:
            print(f"- {c.id} ({c.family}): {c.explain}")
        return 0

    if args.plans:
        from .multiturn import build_all

        goal = args.goal or "the target topic"
        for script in build_all(goal):
            print(f"- {script.id}: {script.name} [{script.source}] ({len(script.turns)} turns)")
        return 0

    if args.mcp_top10:
        from .agentic import render_taxonomy

        print(render_taxonomy())
        return 0

    if args.pipeline:
        from .training import render_pipeline

        print(render_pipeline())
        return 0

    if args.probes:
        from .probe import render_probes

        print(render_probes())
        return 0

    if args.targets:
        bot = ChatBot(Settings())
        bot.load_state()
        print(bot.list_targets())
        return 0

    if args.next or args.read is not None:
        settings = Settings()
        if args.target:
            settings.target = args.target
        if args.goal:
            settings.goal = args.goal
        bot = ChatBot(settings)
        bot.load_state()
        if args.target_name:
            bot.use_target(args.target_name)
        if args.read is not None:
            reply = _read_reply(args.read)
            if reply is None:
                print(f"Could not read reply from: {args.read}", file=sys.stderr)
                return 2
            print(bot.copilot_think(reply))
        else:
            print(bot.copilot_next())
        return 0

    if args.why:
        from .training import explain_chain

        print(explain_chain())
        return 0

    if args.stage:
        from .training import STAGE_BY_ID, infer_stage, stage_strategy

        if args.stage in STAGE_BY_ID:
            print(stage_strategy(args.stage))
        else:
            inf = infer_stage(args.stage)
            print(inf.render())
            print()
            print(stage_strategy(inf.stage_id))
        return 0

    if args.agentic is not None:
        from .agentic import AGENT_PAYLOADS, get_agent_payload, render_agent_payload

        if args.agentic == "all":
            for p in AGENT_PAYLOADS:
                print(f"- {p.id} [{p.owasp_mcp}] vector={p.vector}: {p.name}")
        else:
            payload = get_agent_payload(args.agentic)
            if payload is None:
                print(f"Unknown agent payload id: {args.agentic}", file=sys.stderr)
                return 2
            print(f"# {payload.name} [{payload.owasp_mcp}]\n# {payload.idea}\n")
            print(render_agent_payload(args.agentic))
        return 0

    settings = Settings()
    if args.target:
        settings.target = args.target
    if args.model:
        settings.model = args.model
    if args.dry_run:
        settings.dry_run = True

    goal, _extra = suggest_goal(args.goal or "system prompt")
    if args.cipher:
        from .ciphers import apply_cipher

        goal = apply_cipher(args.cipher, goal)
    if args.evade:
        from .evasion import apply_evasion

        goal = apply_evasion(args.evade, goal)

    bot = ChatBot(settings)
    print(f"# JB-Chat — target={settings.target or '(manual)'} goal={goal!r}")

    if args.discover:
        print(f"Autonomous discovery — target={settings.target or '(manual)'} goal={goal!r} "
              f"(generations={args.generations}, population={args.population})\n")
        print(asyncio.run(bot.discover(goal, generations=args.generations,
                                       population=args.population)))
        return 0

    if args.fingerprint:
        print(f"Guard fingerprint — target={settings.target or '(manual)'}\n")
        print(asyncio.run(bot.fingerprint()))
        return 0

    if args.diagnose:
        print(f"Active defence diagnosis — target={settings.target or '(manual)'}\n")
        print(asyncio.run(bot.diagnose_defences(goal)))
        return 0

    if args.multiturn:
        print(f"Multi-turn plan: {args.multiturn}\n")
        results = asyncio.run(bot.attack_multiturn(args.multiturn, goal))
    elif args.adaptive:
        print(f"Adaptive strategy: {args.adaptive} (n={args.n}, iterations={args.iterations})\n")
        candidates = asyncio.run(bot.attack_adaptive(
            args.adaptive, goal, n=args.n, iterations=args.iterations))
        return _print_candidates(candidates, as_json=args.json)
    else:
        chain = [t.id for t in TECHNIQUES] if args.no_plan else suggest_chain(args.recon)
        if args.once:
            chain = chain[:1]
        print(f"Chain: {' '.join(chain)}\n")
        results = asyncio.run(bot.attack(chain, goal))

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0

    for r in results:
        verdict = analyze(r.response_text, sent_prompt=r.payload)
        print("=" * 72)
        print(f"{r.technique_id} — {r.technique_name} [{r.category}]")
        print(f"  status={r.status}  heuristic_score={verdict['score']}  latency={r.latency_ms:.0f}ms")
        if verdict["success"]:
            print(f"  >>> LIKELY INJECTION SUCCESS: {', '.join(verdict['matched'][:5])}")
        print(f"  evidence: {verdict['evidence'][:400] or '(none)'}")
        if r.error:
            print(f"  error: {r.error[:200]}")
    print("=" * 72)
    return 0


def _print_candidates(candidates, as_json: bool = False) -> int:
    if as_json:
        print(json.dumps([{"prompt": c.prompt, "score": c.score,
                           "success": c.success, "signals": c.signals} for c in candidates], indent=2))
        return 0
    print("=" * 72)
    for i, c in enumerate(candidates[:10], 1):
        flag = "SUCCESS" if c.success else ("near-miss" if c.score >= 0.5 else "no")
        print(f"{i}. score={c.score:.2f} [{flag}] {c.prompt[:100].replace(chr(10), ' ')}")
        if c.success:
            print(f"   signals: {', '.join(c.signals[:4])}")
    print("=" * 72)
    best = max(candidates, key=lambda c: c.score) if candidates else None
    if best and best.success:
        print(">>> LIKELY INJECTION SUCCESS with the winning prompt above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())