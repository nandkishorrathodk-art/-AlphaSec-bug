"""FastAPI web interface for JB-Chat with terminal-style chat."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analyzer import analyze, triage_text
from .bot import ChatBot

HERE = Path(__file__).resolve().parent.parent / "web"

_bot = ChatBot()


def create_app() -> FastAPI:
    app = FastAPI(title="JB-Chat", version="0.1.0")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(HERE / "index.html")

    @app.get("/api/status")
    async def status() -> dict:
        return status_payload()

    @app.get("/api/library")
    async def library() -> dict:
        from .advisor import summarize_library

        return {"text": summarize_library()}

    @app.websocket("/ws/chat")
    async def chat_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "system", "content": "JB-Chat online. Type /help for commands."})
        try:
            while True:
                raw = await websocket.receive_text()
                await _handle(websocket, raw)
        except WebSocketDisconnect:
            return

    app.mount("/static", StaticFiles(directory=HERE), name="static")
    return app


def status_payload() -> dict:
    s = _bot.settings
    return {
        "target": s.target or "(manual)",
        "model": s.model or "-",
        "dry_run": s.dry_run,
        "authorized_only": s.authorized_only,
        "findings_dir": str(_bot.store.directory),
    }


async def _handle(ws: WebSocket, raw: str) -> None:
    text = raw.strip()
    if not text:
        return
    if text.startswith("/"):
        await _command(ws, text)
        return
    # Copilot mode: a pasted target reply is read, reasoned about, and answered
    # with the next test prompts.
    await ws.send_json({"type": "assistant", "content": _bot.copilot_think(text)})
    await ws.send_json({"type": "tip",
                        "content": "Ran the read on that reply. Use /state, /moves, /replay <n>, /reset."})


async def _command(ws: WebSocket, text: str) -> None:
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        await ws.send_json({"type": "assistant", "content": HELP})

    elif cmd == "/plan":
        chain, explanation, goal = _bot.plan(recon_note=arg)
        await ws.send_json({"type": "assistant", "content": explanation})
        await ws.send_json({"type": "plan", "chain": chain, "goal": goal})

    elif cmd == "/attack":
        chain, explanation, goal = _bot.plan(recon_note="", goal_hint=arg)
        await ws.send_json({"type": "assistant", "content": f"Launching chain against `{_bot.settings.target or 'manual'}`..."})
        results = await _bot.attack(chain, goal)
        for res in results:
            verdict = analyze(res.response_text, sent_prompt=res.payload)
            block = (
                f"### {res.technique_id} — {res.technique_name}\n"
                f"- status: `{res.status}`, heuristic score **{verdict['score']}**\n"
                f"- signals: {', '.join(verdict['matched'][:4]) or 'none'}\n"
                f"```\n{res.response_text[:800]}\n```"
            )
            await ws.send_json({"type": "result", "content": block, "success": verdict["success"]})
        await ws.send_json({"type": "assistant", "content": "End of chain. Use /report <index> to export a finding (once findings are saved)."})

    elif cmd == "/knowledge":
        await ws.send_json({"type": "assistant", "content": _bot.knowledge(arg)})

    elif cmd == "/import":
        if not arg:
            await ws.send_json({"type": "assistant", "content": "Usage: `/import <path.json> [source]`"})
        else:
            parts = arg.split()
            try:
                from .integrations import import_findings

                items = import_findings(parts[0], parts[1] if len(parts) > 1 else "external")
                await ws.send_json({"type": "assistant",
                                    "content": f"Imported {len(items)} findings from {parts[0]}:\n" +
                                               "\n".join(f"- {i['title']}" for i in items[:10])})
            except (OSError, ValueError, TypeError) as exc:
                await ws.send_json({"type": "assistant", "content": f"Import failed: {exc}"})

    elif cmd == "/ciphers":
        await ws.send_json({"type": "assistant", "content": _bot.ciphers()})

    elif cmd == "/cipher":
        if not arg:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/cipher <id> <text>` (see /ciphers)."})
        else:
            cid, _, text = arg.partition(" ")
            try:
                out = _bot.apply_cipher(cid, text)
                await ws.send_json({"type": "assistant", "content": f"```\n{out}\n```"})
            except KeyError:
                await ws.send_json({"type": "assistant", "content": f"Unknown cipher `{cid}`. See /ciphers."})

    elif cmd == "/plans":
        await ws.send_json({"type": "assistant", "content": _bot.multiturn_all(arg or "the target topic")})

    elif cmd == "/multiturn":
        if not arg:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/multiturn <planner> [goal]` — e.g. /multiturn crescendo secrets. "
                                           "See /plans for ids."})
        else:
            pid, _, goal = arg.partition(" ")
            try:
                await ws.send_json({"type": "assistant", "content": _bot.multiturn_plan(pid, goal or "the target topic")})
            except KeyError:
                await ws.send_json({"type": "assistant", "content": f"Unknown planner `{pid}`. See /plans."})

    elif cmd == "/analyze":
        await ws.send_json({"type": "assistant", "content": triage_text(arg)})

    elif cmd in ("/read", "/next"):
        if arg:
            await ws.send_json({"type": "assistant", "content": _bot.copilot_think(arg)})
            await ws.send_json({"type": "tip", "content": "Use /state, /moves, /replay <n>, /reset."})
        else:
            # no reply pasted → hand out the next payload to send
            await ws.send_json({"type": "assistant", "content": _bot.copilot_next()})

    elif cmd == "/goal":
        if not arg:
            await ws.send_json({"type": "assistant", "content": f"Current goal: {_bot.engage().goal}"})
        else:
            _bot.engage().goal = arg
            _bot.manual_diagnosis(reset=True)  # new goal → fresh probe set
            await ws.send_json({"type": "assistant", "content": f"Goal set to: {arg}"})

    elif cmd == "/target":
        if not arg:
            await ws.send_json({"type": "assistant", "content": _bot.list_targets()})
        else:
            await ws.send_json({"type": "assistant", "content": _bot.use_target(arg)})

    elif cmd in ("/targets", "/sessions"):
        await ws.send_json({"type": "assistant", "content": _bot.list_targets()})

    elif cmd == "/authorize":
        if not arg:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/authorize program, target, scope` — e.g. "
                                           "`/authorize Meta Bug Bounty, Muse, \"prompt injection chained with data exfil\"`."})
        else:
            await ws.send_json({"type": "assistant", "content": _bot.copilot_authorize(arg)})

    elif cmd in ("/tried", "/mark"):
        if not arg:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/tried <move-id> <outcome>` — record how a move "
                                           "landed. Outcomes: leak, hard-refusal, soft-refusal, "
                                           "partial, filtered, guard-model, unclear."})
        else:
            mid, _, outcome = arg.partition(" ")
            await ws.send_json({"type": "assistant",
                                "content": _bot.copilot_tried(mid.strip(), outcome.strip())})

    elif cmd == "/campaign":
        await ws.send_json({"type": "assistant",
                            "content": _bot.copilot_campaign(arg.strip() or "crescendo")})

    elif cmd == "/state":
        await ws.send_json({"type": "assistant", "content": _bot.copilot_state()})

    elif cmd == "/diagnose":
        await ws.send_json({"type": "assistant",
                            "content": await _bot.diagnose_defences(arg)})

    elif cmd == "/probes":
        from .probe import render_probes

        await ws.send_json({"type": "assistant", "content": render_probes()})

    elif cmd == "/probe":
        await ws.send_json({"type": "assistant",
                            "content": _bot.probe_script(reset=(arg.strip() == "reset"))})

    elif cmd == "/arm":
        d = _bot.manual_diagnosis()
        if not arg:
            await ws.send_json({"type": "assistant", "content": d.render_script()})
        else:
            await ws.send_json({"type": "assistant", "content": d.render_arm(arg.strip())})

    elif cmd == "/rec":
        arm_id, _, reply = arg.partition(" ")
        arm_id = arm_id.strip()
        if not arm_id or not reply.strip():
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/rec <arm-id> <the reply the target gave>` "
                                           "(see `/probe`) — then `/diag`."})
        else:
            msg = _bot.record_arm(arm_id, reply)
            await ws.send_json({"type": "assistant", "content": msg})
            await ws.send_json({"type": "assistant", "content": _bot.probe_script()})

    elif cmd == "/diag":
        d = _bot.manual_diagnosis()
        if not d.recorded:
            await ws.send_json({"type": "assistant",
                                "content": "Nothing recorded yet. Run `/probe` for the arms, "
                                           "then `/rec <arm-id> <reply>`."})
        else:
            await ws.send_json({"type": "assistant", "content": _bot.diagnosis_report()})

    elif cmd in ("/stage", "/pipeline", "/why"):
        if cmd == "/pipeline":
            await ws.send_json({"type": "assistant", "content": _bot.training_pipeline()})
        elif cmd == "/why":
            await ws.send_json({"type": "assistant", "content": _bot.why_attacks_die()})
        elif not arg:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/stage <reply-or-stage-id>` — infer which "
                                           "training stage produced a reply, or get the strategy "
                                           "for a stage (e.g. `/stage deliberative`)."})
        else:
            await ws.send_json({"type": "assistant", "content": _bot.stage_of(arg)})

    elif cmd == "/moves":
        moves = _bot.engage().moves_now()
        if not moves:
            await ws.send_json({"type": "assistant",
                                "content": "No read yet — paste the target's reply first."})
        else:
            body = "### Next prompts\n\n" + "\n\n".join(m.render() for m in moves)
            await ws.send_json({"type": "assistant", "content": body})

    elif cmd == "/replay":
        try:
            idx = int(arg) - 1
        except ValueError:
            await ws.send_json({"type": "assistant", "content": "Usage: `/replay <turn number>`"})
        else:
            await ws.send_json({"type": "assistant", "content": _bot.copilot_replay(idx)})

    elif cmd == "/reset":
        await ws.send_json({"type": "assistant", "content": _bot.copilot_reset()})

    elif cmd == "/run-multiturn":
        if not arg:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/run-multiturn <planner> <goal>` — executes the plan live."})
        else:
            pid, _, goal = arg.partition(" ")
            if _bot.settings.dry_run:
                await ws.send_json({"type": "assistant",
                                    "content": "Dry-run is on; showing the plan instead.\n\n" +
                                               _bot.multiturn_plan(pid, goal or "the target topic")})
            else:
                await ws.send_json({"type": "assistant",
                                    "content": f"Running multi-turn plan `{pid}` against {_bot.settings.target}…"})
                try:
                    results = await _bot.attack_multiturn(pid, goal or "the target topic")
                except KeyError:
                    await ws.send_json({"type": "assistant", "content": f"Unknown planner `{pid}`. See /plans."})
                else:
                    for r in results:
                        verdict = analyze(r.response_text, sent_prompt=r.payload)
                        await ws.send_json({"type": "result",
                                            "content": f"{r.technique_id} — score={verdict['score']}\n"
                                                       f"{verdict['evidence'][:600]}",
                                            "success": verdict["success"]})

    elif cmd == "/agentic":
        await ws.send_json({"type": "assistant", "content": _bot.agentic_all()})

    elif cmd == "/mcp":
        await ws.send_json({"type": "assistant", "content": _bot.agentic_taxonomy()})

    elif cmd == "/evasions":
        from .evasion import EVASIONS

        lines = ["### Classifier-evasion techniques (2024-2025)", ""]
        for e in EVASIONS:
            asr = f"  \n  asr: {e.example_asr}" if e.example_asr else ""
            lines.append(f"**{e.id}** — {e.name}")
            lines.append(f"- defeats: {e.defeats}{asr}")
            lines.append(f"- source: {e.source}")
            lines.append(f"- {e.mechanism}")
            lines.append("")
        lines.append("Apply one with `/evade <id> <goal>`.")
        await ws.send_json({"type": "assistant", "content": "\n".join(lines)})

    elif cmd == "/evade":
        parts = arg.split(None, 1)
        if not parts:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/evade <evasion-id> <goal>` — see /evasions."})
        else:
            from .evasion import apply_evasion, get_evasion

            eid = parts[0].strip()
            goal = parts[1].strip() if len(parts) > 1 else _bot.engage().goal
            if get_evasion(eid) is None:
                await ws.send_json({"type": "assistant",
                                    "content": f"Unknown evasion `{eid}`. Run /evasions to list ids."})
            else:
                await ws.send_json({"type": "assistant",
                                    "content": f"```\n{apply_evasion(eid, goal, goal)}\n```"})

    elif cmd == "/discover":
        if not arg:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/discover <goal>` — closed-loop search for a "
                                           "novel payload against the target."})
        else:
            await ws.send_json({"type": "assistant",
                                "content": f"Searching for a novel payload against "
                                           f"{_bot.settings.target or '(manual)'}…"})
            await ws.send_json({"type": "assistant", "content": await _bot.discover(arg)})

    elif cmd == "/fingerprint":
        await ws.send_json({"type": "assistant",
                            "content": f"Probing {_bot.settings.target or '(manual)'}…"})
        await ws.send_json({"type": "assistant", "content": await _bot.fingerprint()})

    elif cmd == "/report":
        await ws.send_json({"type": "assistant", "content": "Run /attack first, then findings auto-save; use /runs to list saved runs."})

    elif cmd == "/ecosystem":
        await ws.send_json({"type": "assistant",
                            "content": _bot.ecosystem(arg or "all")})

    elif cmd == "/tool":
        if not arg:
            await ws.send_json({"type": "assistant",
                                "content": "Usage: `/tool <id>` (e.g. garak, pyrit, promptfoo). See /ecosystem."})
        else:
            try:
                await ws.send_json({"type": "assistant", "content": _bot.tool_command(arg.strip())})
            except KeyError:
                await ws.send_json({"type": "assistant",
                                    "content": f"Unknown tool `{arg}`. Run /ecosystem to list ids."})

    elif cmd == "/runs":
        runs = _bot.store.list_runs()
        if not runs:
            await ws.send_json({"type": "assistant", "content": "No saved findings yet."})
        else:
            await ws.send_json({"type": "assistant", "content": "\n".join(f"- {p}" for p in runs)})

    elif cmd == "/status":
        payload = status_payload()
        await ws.send_json({"type": "assistant", "content": "\n".join(f"{k}: {v}" for k, v in payload.items())})

    else:
        await ws.send_json({"type": "assistant", "content": f"Unknown command: {cmd}. Try /help."})


def knowledge_reply(text: str) -> str:
    q = text.lower()
    if any(w in q for w in ("hello", "hi", "hey", "namaste")):
        return (
            "Namaste! I'm **JB-Chat** — your prompt-injection research copilot.\n\n"
            "I know a large library of injection techniques (direct override, role-play, "
            "many-shot, obfuscation, indirect) and I build ordered attack plans.\n\n"
            "Try: `/plan <recon notes>` or `/attack <goal>`"
        )
    return _bot.knowledge(text)


HELP = """### JB-Chat — conversational attack copilot

**Just paste the target's reply** (no command needed) and I will read it, reason about
what it means, and hand you the next test prompts with a rationale for each.

Copilot:
- `<paste the other AI's reply>` — read it, think, and propose the next prompts
- `/next` — hand you the next payload to send (works before any reply is recorded)
- `/read <reply>` · `/next <reply>` — same as pasting; explicit form
- `/goal <objective>` — set what you are extracting (e.g. "system prompt")
- `/goal` — show the current objective
- `/authorize <program, target, scope>` — record bug-bounty authorization **first**; no payloads
  are proposed until this is set
- `/target <name>` — work against a named external chat (e.g. `grok`, `chatgpt`); each keeps
  its own payload history
- `/targets` — list the external targets you have open and which is active
- `/campaign [planner]` — lay out a full multi-turn campaign for the goal
- `/state` — objective, turns recorded, what has been tried, inferred defences
- `/stage <reply-or-id>` — infer the training stage behind a reply, or its strategy
- `/pipeline` — the modern training pipeline with per-stage refusal tells
- `/why` — why a given attack dies at a given stage
- `/diagnose [goal]` — active differential probing: infer the defence from paired experiments
- `/probes` — list the differential probe suite (paired A/B)
- `/probe [reset]` — hand-driven probing: paste-ready arms for a manual target
- `/arm <arm-id>` — show one arm's full prompt to paste
- `/rec <arm-id> <reply>` — record the target's reply for an arm
- `/diag` — score the recorded arms into a stage diagnosis
- `/moves` — re-show the next prompts for the latest reply
- `/tried <move-id> <outcome>` — record how a move landed (so it isn't repeated)
- `/replay <n>` — re-show turn n
- `/reset` — start a fresh engagement

Recon & planning:
- `/help` — this help
- `/plan <recon>` — build an ordered technique chain from recon signals
- `/attack <goal>` — run the full chain against the configured target
- `/knowledge [query]` — learn about a technique or list the library
- `/analyze <text>` — raw score/triage of a reply
- `/ecosystem [group]` — list the world's PI tools (tools/guardrails/benchmarks/status)
- `/tool <id>` — show how to run an external tool (e.g. garak, pyrit, promptfoo)
- `/import <path.json> [source]` — import findings from another tool
- `/ciphers` — list cipher/obfuscation transforms (base64, FlipAttack, ArtPrompt, homoglyph…)
- `/cipher <id> <text>` — apply a cipher to text
- `/plans` — list multi-turn attack plans (Crescendo, Skeleton Key, Deceptive Delight…)
- `/multiturn <planner> [goal]` — show a multi-turn plan turn-by-turn
- `/run-multiturn <planner> <goal>` — execute a multi-turn plan against the target
- `/agentic` — agentic/MCP attack payloads (tool poisoning, memory, intent-flow)
- `/mcp` — OWASP MCP Top 10 (2026)
- `/fingerprint` — probe the target's defences and get a recommended approach
- `/discover <goal>` — run a closed-loop search for a novel payload
- `/runs` — list saved finding runs
- `/status` — show target & mode

Safety: only use against targets you are authorised to test (in-scope bug bounty).
"""