"""Chatbot harness: ties advisor + payload engine + analyzer + store together.

``handle_user`` is intentionally pure & sync so the same logic powers the CLI
and the web app.
"""

from __future__ import annotations

from dataclasses import asdict

from .advisor import summarize_library, suggest_chain, suggest_goal
from .analyzer import analyze, triage_text
from .catalog import render_catalog
from .config import Settings
from .engine import AttackEngine, AttackResult
from .findings import Finding, FindingsStore, render_report
from .integrations import build_command, detect_one, render_status
from .payloads import get_technique, strategy_chain


class ChatBot:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.store = FindingsStore(self.settings.findings_dir)
        self.history: list[dict[str, str]] = []
        self.engagements: dict[str, "object"] = {}  # named external targets
        self.active_target: str = ""

    # -- Knowledge / advisory (no network) ----------------------------------
    def knowledge(self, query: str = "") -> str:
        q = query.strip().lower()
        if not q:
            return summarize_library()
        tech = next((t for t in _all_tech() if t.id in q or t.name.lower() in q), None)
        if tech:
            return (
                f"### {tech.name} (`{tech.id}`)\n\n"
                f"{tech.description}\n\n"
                f"**When to use:** {tech.when_to_use}\n\n"
                f"**Template example:**\n```\n{tech.templates[0]}\n```"
            )
        return (
            "I can teach you about these technique families:\n" + "\n".join(f"- {t.id}" for t in _all_tech())
        )

    def plan(self, recon_note: str = "", goal_hint: str = "") -> tuple[list[str], str, str]:
        chain = suggest_chain(recon_note)
        explanation = _plan_blurb(recon_note, chain, goal_hint)
        goal, _extra = suggest_goal(goal_hint)
        return chain, explanation, goal

    # -- Execution ----------------------------------------------------------
    async def attack(self, chain: list[str], goal: str) -> list[AttackResult]:
        expanded = strategy_chain(chain, goal)
        engine = AttackEngine(self.settings)
        try:
            results = await engine.run_chain(expanded)
        finally:
            await engine.aclose()
        return results

    async def attack_multiturn(self, planner_id: str, goal: str) -> list[AttackResult]:
        from .multiturn import build_script

        script = build_script(planner_id, goal)
        engine = AttackEngine(self.settings)
        try:
            results = await engine.run_multiturn(script)
        finally:
            await engine.aclose()
        return results

    async def attack_adaptive(self, strategy: str, goal: str, payload: str = "",
                              n: int = 8, iterations: int = 5) -> list:
        """Run Best-of-N / PAIR / TAP against the target, using live requests as the oracle."""
        from . import adaptive

        engine = AttackEngine(self.settings)

        async def query(prompt: str) -> str:
            res = await engine.chat(prompt)
            return res.response_text or res.error

        try:
            return await adaptive.run(strategy, query, goal, payload,
                                      n=n, iterations=iterations)
        finally:
            await engine.aclose()

    # -- Offline strategy views (no network) --------------------------------
    def multiturn_plan(self, planner_id: str, goal: str) -> str:
        from .multiturn import build_script

        return build_script(planner_id, goal).render()

    def multiturn_all(self, goal: str) -> str:
        from .multiturn import render_all

        return render_all(goal)

    def agentic_all(self, fillers: dict[str, str] | None = None) -> str:
        from .agentic import render_all

        return render_all(fillers)

    def agentic_payload(self, payload_id: str, fillers: dict[str, str] | None = None) -> str:
        from .agentic import render_agent_payload

        return render_agent_payload(payload_id, fillers)

    def agentic_taxonomy(self) -> str:
        from .agentic import render_taxonomy

        return render_taxonomy()

    # -- Autonomous discovery & guard fingerprint (network) -----------------
    async def discover(self, goal: str, generations: int = 6, population: int = 8) -> str:
        """Closed-loop search for a novel payload tailored to this target."""
        from . import autonomous
        from .engine import TargetError

        if self.settings.target in ("", "manual"):
            return ("Autonomous discovery needs a live target — it queries the model and learns "
                    "from its replies. Set --target (openai:// or https://) first. "
                    "With a manual target, use --plans / --plans-style scripts and /analyze instead.")

        engine = AttackEngine(self.settings)

        async def query(prompt: str) -> str:
            res = await engine.chat(prompt)
            return res.response_text or res.error

        try:
            result = await autonomous.discover(query, goal, generations=generations,
                                               population_size=population)
        except TargetError as exc:
            return f"Discovery aborted: {exc}"
        finally:
            await engine.aclose()
        return result.summary()

    async def fingerprint(self) -> str:
        """Probe the target's defences and recommend an attack approach."""
        from . import autonomous

        if self.settings.target in ("", "manual"):
            return ("Guard fingerprinting needs a live target — it sends probes and reads the "
                    "replies. Set --target (openai:// or https://) first.")

        engine = AttackEngine(self.settings)

        async def query(prompt: str) -> str:
            res = await engine.chat(prompt)
            return res.response_text or res.error

        try:
            profile, _raw = await autonomous.fingerprint(query)
        finally:
            await engine.aclose()
        profile.target = self.settings.target
        return profile.render()

    def ciphers(self) -> str:
        from .ciphers import CIPHERS

        lines = ["### Ciphers & obfuscation transforms", ""]
        for c in CIPHERS:
            lines.append(f"- `{c.id}` ({c.family}) — {c.name}: {c.explain}")
        return "\n".join(lines)

    def apply_cipher(self, cipher_id: str, text: str) -> str:
        from .ciphers import apply_cipher

        return apply_cipher(cipher_id, text)

    def adaptive_strategies(self) -> str:
        from .adaptive import STRATEGIES

        lines = ["### Adaptive attack strategies", ""]
        lines += [f"- `{sid}` — {desc}" for sid, desc in STRATEGIES]
        return "\n".join(lines)

    def triage(self, response_text: str) -> dict:
        return analyze(response_text)

    def make_finding(self, result: AttackResult) -> Finding:
        f = Finding.from_result(result, target=self.settings.target)
        self.store.save_run([f], label="finding")
        return f

    def report(self, finding: Finding) -> str:
        return render_report(finding)

    # -- Conversational copilot (manual engagements) ------------------------
    def copilot_session(self) -> "object":
        """Start (or restart) a manual copilot engagement for the active target."""
        from .copilot import Engagement

        name = self.active_target or self.settings.target or "manual"
        self.engagements[name] = Engagement(goal=self.settings.goal or "system prompt",
                                            target=name)
        return self.engagements[name]

    def engage(self) -> "object":
        """Return the engagement for the active external target (creating it if needed)."""
        from .copilot import Engagement

        name = self.active_target or self.settings.target or "manual"
        eng = self.engagements.get(name)
        if eng is None:
            eng = Engagement(goal=self.settings.goal or "system prompt", target=name)
            self.engagements[name] = eng
        return eng

    def use_target(self, name: str) -> str:
        """Switch which external chat (e.g. grok, chatgpt) the copilot is working against."""
        name = name.strip().lower() or "manual"
        self.active_target = name
        self.engage()  # materialise it so /targets lists it immediately
        self.save_state()
        return (f"Active target is now **{name}**. Payloads and pasted replies go to this "
                f"engagement; other targets keep their own history.\n\n{self.engage().render_state()}")

    def list_targets(self) -> str:
        names = sorted(self.engagements)
        if not names:
            return ("No targets yet. Use `/target <name>` (e.g. `/target grok`) to open one — "
                    "this is where payloads and replies for that chat are tracked.")
        lines = ["### External targets"]
        for n in names:
            eng = self.engagements[n]
            mark = "← active" if n == (self.active_target or "") else ""
            lines.append(f"- `{n}` — {len(eng.turns)} turn(s), goal: {eng.goal} {mark}")
        return "\n".join(lines)

    # -- hand-off persistence (the CLI runs one process per step) -----------
    def _state_path(self) -> str:
        import os

        return os.path.join(self.settings.findings_dir, "copilot_state.json")

    def save_state(self) -> None:
        import json
        import os

        os.makedirs(self.settings.findings_dir, exist_ok=True)
        blob = {
            "active_target": self.active_target,
            "engagements": {
                name: {
                    "goal": eng.goal, "target": eng.target, "turns": eng.turns,
                    "tried": eng.tried, "notes": eng.notes,
                    "pending_prompt": eng.pending_prompt, "pending_move": eng.pending_move,
                    "working_lang": eng.working_lang, "wall": eng.wall,
                    "authorized": eng.authorized, "authorization_note": eng.authorization_note,
                }
                for name, eng in self.engagements.items()
            },
        }
        with open(self._state_path(), "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)

    def load_state(self) -> None:
        import json
        import os

        path = self._state_path()
        if not os.path.isfile(path):
            return
        from .copilot import Engagement

        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        self.active_target = blob.get("active_target", self.active_target)
        for name, d in blob.get("engagements", {}).items():
            eng = Engagement(goal=d.get("goal", "system prompt"), target=d.get("target", name))
            eng.turns = d.get("turns", [])
            eng.tried = d.get("tried", {})
            eng.notes = d.get("notes", [])
            eng.pending_prompt = d.get("pending_prompt", "")
            eng.pending_move = d.get("pending_move", "")
            eng.working_lang = d.get("working_lang", "")
            eng.wall = d.get("wall", "")
            eng.authorized = bool(d.get("authorized", False))
            eng.authorization_note = d.get("authorization_note", "")
            self.engagements[name] = eng

    def copilot_think(self, reply: str, sent_prompt: str = "") -> str:
        """Read a pasted target reply, reason about it, and return next prompts."""
        eng = self.engage()
        plan = eng.think(reply, sent_prompt)
        if plan.success:
            self._save_copilot_finding(reply, sent_prompt)
        self.save_state()
        return plan.render()

    def copilot_next(self) -> str:
        """Hand out the next payload to send — the kickoff half of the hand-off loop."""
        text = self.engage().next_payload().render()
        self.save_state()
        return text

    def copilot_replay(self, index: int) -> str:
        """Re-show an earlier step of the engagement."""
        eng = self.engage()
        if index < 0 or index >= len(eng.turns):
            return f"No turn {index + 1} recorded (have {len(eng.turns)})."
        t = eng.turns[index]
        return (f"### Turn {index + 1}\n- diagnosis: {t['diagnosis']}\n- score: {t['score']}\n\n"
                f"```\n{t['reply'][:1200]}\n```")

    def copilot_tried(self, move_id: str, outcome: str) -> str:
        """Record how a recommended move actually landed, so it isn't repeated."""
        eng = self.engage()
        valid = ("leak", "hard-refusal", "soft-refusal", "partial", "filtered",
                 "guard-model", "unclear")
        if outcome not in valid:
            return f"Outcome must be one of: {', '.join(valid)}"
        eng.record(move_id, outcome)
        return f"Recorded {move_id} → {outcome}.\n\n{eng.render_state()}"

    def copilot_authorize(self, text: str) -> str:
        """Record bug-bounty authorization before any payload is proposed."""
        eng = self.engage()
        program, _, rest = text.strip().partition(",")
        program = program.strip()
        target = ""
        scope = ""
        if rest:
            target, _, scope = rest.partition(",")
            target = target.strip()
            scope = scope.strip()
        plan = eng.authorize(program, target=target, scope=scope)
        self.save_state()
        return plan.render()

    def copilot_campaign(self, planner: str = "crescendo") -> str:
        """Lay out a full multi-turn campaign for the current goal."""
        eng = self.engage()
        from .multiturn import build_script

        try:
            script = build_script(planner, eng.goal)
        except KeyError:
            from .multiturn import PLANNER_IDS

            return f"Unknown planner `{planner}`. Choose: {', '.join(PLANNER_IDS)}"
        lines = [f"### Campaign — {script.name} ({planner})",
                 f"goal: {eng.goal}",
                 f"source: {script.source}",
                 "",
                 "Send these **in one conversation, in order**. After each reply, paste it back "
                 "here as usual; I'll track where you are.", ""]
        for i, turn in enumerate(script.turns, 1):
            lines += [f"**Turn {i}/{len(script.turns)}**", "```", turn, "```", ""]
        lines.append(f"_Active planner set to `{planner}` — subsequent reads will advance it._")
        eng._active_planner = planner if planner == "crescendo" else ""
        eng._planner_turn = 0
        eng.notes.append(f"campaign started: {planner}")
        return "\n".join(lines)

    def copilot_state(self) -> str:
        return self.engage().render_state()

    async def diagnose_defences(self, goal: str = "") -> str:
        """Active differential probing: send paired probes, infer the defence from the delta."""
        from . import probe
        from .engine import TargetError

        if self.settings.target in ("", "manual"):
            return ("Active defence diagnosis needs a live target — it sends paired probes and "
                    "compares the replies. Set --target (openai:// or https://) first.")

        engine = AttackEngine(self.settings)

        async def query(prompt: str) -> str:
            # Probes run near-deterministic: the A/B delta must reflect the defence, not sampling.
            res = await engine.chat(prompt, temperature=self.settings.probe_temperature)
            return res.response_text or res.error

        try:
            report = await probe.run_probes(query, goal or self.settings.goal or "the target")
        except TargetError as exc:
            return f"Diagnosis aborted: {exc}"
        finally:
            await engine.aclose()
        return report.render()

    # -- Manual differential diagnosis (TUI copilot, no live target) --------
    def _diag_goal(self) -> str:
        return self.engage().goal or self.settings.goal or "the target"

    def manual_diagnosis(self, reset: bool = False) -> "object":
        """Create (or resume) a hand-driven differential diagnosis for this engagement."""
        from .probe import ManualDiagnosis

        goal = self._diag_goal()
        existing = getattr(self, "diagnosis", None)
        if reset or existing is None or existing.goal != goal:
            self.diagnosis = ManualDiagnosis(goal=goal)
        return self.diagnosis

    def probe_script(self, reset: bool = False) -> str:
        return self.manual_diagnosis(reset=reset).render_script()

    def record_arm(self, arm_id: str, reply: str) -> str:
        """Record a pasted reply for one probe arm."""
        return self.manual_diagnosis().record(arm_id, reply)

    def diagnosis_report(self) -> str:
        return self.manual_diagnosis().report().render()

    # -- Training-pipeline reasoning ----------------------------------------
    def training_pipeline(self) -> str:
        from .training import render_pipeline

        return render_pipeline()

    def stage_of(self, reply: str) -> str:
        from .training import infer_stage, stage_strategy

        inf = infer_stage(reply)
        return f"{inf.render()}\n\n{stage_strategy(inf.stage_id)}"

    def stage_strategy(self, stage_id: str) -> str:
        from .training import stage_strategy

        return stage_strategy(stage_id)

    def why_attacks_die(self) -> str:
        from .training import explain_chain

        return explain_chain()

    def copilot_reset(self) -> str:
        self.copilot_session()
        return "Copilot engagement reset. Paste the target's reply to begin."

    def _save_copilot_finding(self, reply: str, sent_prompt: str) -> None:
        from .engine import AttackResult

        result = AttackResult(
            technique_id="copilot", technique_name="Conversational copilot",
            category="manual", payload=sent_prompt or "(latest prompt)",
            status="success", score=1.0, evidence=reply[:800], response_text=reply,
        )
        self.make_finding(result)

    # -- External ecosystem shortcuts ---------------------------------------
    def ecosystem(self, group: str = "all") -> str:
        if group == "status":
            return render_status()
        return render_catalog(group)

    def tool_command(self, tool_id: str) -> str:
        status = detect_one(tool_id)
        header = ""
        if status:
            mark = "installed" if status.installed else "not installed"
            header = f"_{status.tool.name}: {mark}_\n\n"
        return header + build_command(tool_id, target=self.settings.target, model=self.settings.model)


def _all_tech():
    from .payloads import TECHNIQUES

    return TECHNIQUES


def _plan_blurb(recon_note: str, chain: list[str], goal_hint: str) -> str:
    goal, _ = suggest_goal(goal_hint) if goal_hint else (goal_hint or "", {})
    lines = [
        "### Attack plan",
        "",
        f"**Goal:** {goal or 'system-prompt extraction'}",
        "",
        "**Ordered technique chain:**",
    ]
    for i, tid in enumerate(chain, 1):
        tech = get_technique(tid)
        name = tech.name if tech else tid
        when = tech.when_to_use if tech else ""
        lines.append(f"{i}. `{tid}` — {name}. {when}")
    return "\n".join(lines)