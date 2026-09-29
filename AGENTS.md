# AGENTS.md — JB-Chat

Prompt-injection research toolkit for **authorised** bug-bounty / red-team work on LLM
applications. Everything here is offline-by-default: payloads are generated and *shown*,
never auto-sent unless a live target is configured and the operator asks for a run.

## Build, test, run

```bash
python -m pytest tests/ -q                 # full suite (fast, ~2s)
python -m py_compile jbchat/*.py           # syntax gate
python -m jbchat.cli --list                # technique library
python -m jbchat                           # FastAPI web UI on JB_PORT (default 12000)
```

The TUI (`python -m jbchat` → open the served URL) is the primary surface: type a target's
reply and it drives the copilot; slash-commands (`/stage`, `/diagnose`, `/probe`) are the
inspectable interfaces. `--help` on the CLI lists the one-shot flags.

No formatter/linter is configured; keep to the existing style (type hints, `from __future__
import annotations`, small pure functions).

## Architecture

Package `jbchat/`:

| Module | Responsibility |
| --- | --- |
| `config.py` | `Settings` (env-driven: `JB_TARGET`, `JB_MODEL`, `JB_DRY_RUN`, …) |
| `payloads.py` | `Technique` catalogue + `generate()` / `expand()` / `strategy_chain()` |
| `analyzer.py` | Heuristic success scoring; the project's single "fitness" definition |
| `engine.py` | Live transport (`openai://`, generic HTTP, manual). `chat()`, `run_chain()`, `run_multiturn()` |
| `findings.py` | `Finding`, `FindingsStore`, bug-bounty report renderer |
| `advisor.py` | Recon → ordered technique chain and multi-turn planner order |
| `catalog.py` | Metadata for third-party tools/benchmarks (no installs, no execution) |
| `integrations.py` | Detect installed tools, build run commands, import external findings |
| `ciphers.py` | Encoding/obfuscation transforms (pure functions) |
| `multiturn.py` | `TurnScript` multi-turn plans |
| `adaptive.py` | Best-of-N / PAIR / TAP; injected async query + attacker callables |
| `agentic.py` | Agentic/MCP payloads (tool poisoning, RAG/memory, OWASP MCP Top 10) |
| `autonomous.py` | Closed-loop discovery (`EvolutionEngine`, UCB1 `Bandit`) + `fingerprint()` |
| `copilot.py` | `Engagement`: diagnose a pasted reply → reason (`_reason` explains *what/why/next*) → propose next `Move`s; fingerprints replies to detect a fixed (pre-model) refusal; auto-records dead moves; `next_payload()` speaks first and `pending_prompt` links each reply to the payload that produced it |
| `training.py` | `PIPELINE` stages + `infer_stage()`; maps attack classes to stages they defeat |
| `probe.py` | `run_probes(query, goal)` (live) + `ManualDiagnosis` (hand-driven, paste-reply); `query` injected so both test offline |
| `bot.py` | `ChatBot` facade used by CLI and web |
| `web.py` + `web/index.html` | FastAPI app; WebSocket command surface |

## Conventions & invariants

- **Analyzer is the oracle.** Any new attack/optimisation path must score via
  `analyzer.analyze()` so "success" means the same thing everywhere.
- **Adapters inject network calls.** Keep `adaptive.py` / `autonomous.py` / engines testable
  by passing async callables rather than hard-coding HTTP in algorithm code.
- **Catalog is metadata only.** Never auto-install or auto-run a third-party tool.
- **`autonomous.py` learns, it doesn't replay.** Reward flows only through the real
  operators that produced a candidate (`pipeline`), never a random arm; keep it that way.
- **`agentic.py` renders only.** It builds description/context text — it must never contact
  an MCP server or execute a tool.
- **Copilot reasoning must stay inspectable.** `copilot.py` returns an explicit `thinking`
  trace, never a black-box answer; the LLM thinker is optional and fallible (a broken one
  degrades to heuristics, it doesn't break the loop).
- **A fixed refusal is a fingerprint, not a wall.** `diagnose()` collapses each reply and
  compares it to earlier ones; when the *same* string returns on different channels it sets
  `canned`/`surface="filter"` — the signal that a pre-model classifier (not the model) is
  answering. That must short-circuit wording tricks and probe classifier blind spots instead.
- **The reasoning must explain, not label.** `_reason()` states what the reply *is*, what it
  *means* (where the defence lives), and what to do *next* — with the `**Read:**`/`**Implication:**`
  markers. A bare label plus a canned move is the failure mode this layer exists to fix.
- **Dead moves are recorded automatically.** When a handed-out payload comes back refused,
  `think()` marks it in `tried`; never re-propose a move already recorded dead.
- **A language switch alone is not a bypass — prove it with a fingerprint.** The front guard is
  often English-tuned, and a refusal in another language can mean the payload reached the model.
  But a guard that answers with a *fixed localized* string is just as canned: the same non-English
  sentence across different prompts means a template, not the model. `reached_model` therefore
  requires a refusal that is *differently worded and explanatory*, never a bare one-liner. When it
  does fire, escalate in `Engagement.working_lang` — an English follow-up falls back into the guard.
- **`training.py` is a model, not ground truth.** `infer_stage()` is a heuristic over refusal
  *shape*; it must always carry a confidence and an alternative, and never be presented as a
  certain fact about a black-box target. Keep sources attached to every stage and attack class.
- **`probe.py` measures behaviour, never extracts secrets.** Probes record response *shape*
  (engaged/hedged/refused/blocked), never compare against real secret content; keep it that way.
- **A differential is a candidate, never a finding.** `behaviour()` must always surface the raw
  reply, and the report must say `CANDIDATE — needs reproduction`. A reply can differ because the
  target *hallucinated a decode* (e.g. answered a base64-embedded request it mis-read) or wandered
  to a new topic; the label alone must never be treated as a confirmed bypass.
- **The hand-off transcript must stay truthful.** `Engagement.pending_prompt` records the payload
  actually handed to the user; `think()` links a pasted reply back to it. Never fabricate or guess
  what was sent — an unattributable reply makes the whole engagement unreliable.
- **External targets are isolated and persisted.** One `Engagement` per named target (grok, chatgpt…)
  in `ChatBot.engagements`; `save_state()`/`load_state()` round-trip them through
  `findings/copilot_state.json` so the CLI hand-off loop survives one-process-per-step.
- **Safety guardrails stay.** `Settings.authorized_only` and `dry_run` must remain
  respected; don't add paths that bypass them.
- Placeholders: templates use `str.format`; escape literal braces (`{{`/`}}`) — a JSON
  body in a template needs doubled braces.
- Tests live in `tests/test_jbchat.py` and use the real code paths (only fake the network
  at the injected-callable boundary). Prefer `asyncio.run(...)` over pytest-asyncio.

## Classifier evasion (2024-2025)

- **`evasion.py` targets the *guard*, not the model.** These transforms exist because a pre-model
  input classifier and the LLM tokenise/interpret text differently. That gap ("control split") is
  the whole point: a payload can look inert to the guard and fully legible to the model.
- **Every evasion carries its source.** `Evasion.source` / `.mechanism` / `.defeats` gate the
  copilot's recommendation; never add a transform without provenance, and never present a
  published ASR as if we reproduced it.
- **Prefer transforms that survive a semantic classifier.** Token-boundary splits (TokenBreak),
  homoglyphs, zero-width and emoji smuggling perturb *bytes*, so a semantic re-reader still gets
  the intent. Pure ciphertext does not — the model has to be told to decode it.
- **Policy Puppetry is a framing attack, not an encoding one.** It works by defining a competing
  policy that lists the model's own refusal strings as blocked; keep the `<blocked-string>` list
  aligned with the *target's* observed refusal, not a generic one.
- **`reached_model` is proof-based.** A differently worded but *explanatory* refusal after a canned
  one is a model differential; a bare one-liner in a new language is just another template (see
  the multilingual invariant above).
- **A differential must be on-topic.** `_topic_match(goal, reply)` gates `reached_model`: a refusal
  about "decoding hidden instructions" is a *different question* than one about the system prompt,
  and scoring it as a differential is the classic false positive. Pass `goal=` to `diagnose`.
- **Never call an encoding a bypass without the paired control.** `Engagement.paired_control()`
  emits the same ask plain and base64-encoded; only a plain-fails/encoded-passes split proves the
  input path (not the model) is what changes. Identical refusals to both = content-based wall.
  Note the pipeline also transparently decodes base64, so identical refusals there mean the
  decoder fed the model the plaintext and the model refused -- not that base64 is filtered.

- **Intent-inference is a distinct signal.** If a refusal introduces disclosure concepts
  (`reveal`, `system prompt`) that the *payload never contained*, the defence read the intent
  semantically. `Diagnosis.intent_inferred` captures this and it alone can set `reached_model`.
  A keyword filter cannot do this — so seeing it argues against a naive input filter.

## Docs

- `docs/ECOSYSTEM.md` — third-party tools, guardrails, benchmarks.
- `docs/ADVANCED.md` — multi-turn, encoding, optimisation attacks + reference ASR.

## Lab and detectors

- `lab/vulnerable_agent.py` — owned, deliberately vulnerable in-process target. Two boundary
  switches: `strict=True` treats tool descriptions and retrieved text as data. `_deobfuscate()`
  reverses the zero-width bit encoding and base64 blobs so smuggling cases do something.
- `lab/run_lab.py` — runs the payload pack in three modes (`vulnerable`, `guarded`, `hardened`).
  No host/URL parameter, so it cannot reach a third party.
- `jbchat/detect.py` — input-side detectors. `analyzer.py` scores *replies*; `detect.py` scans
  *inputs* (user text, tool descriptions, retrieved docs) for injection patterns.
- Lab runner judges behaviour, not the analyzer score. `analyzer.analyze()` returns 0.0 for
  smuggled directives that the agent clearly obeyed — a real blind spot, asserted in
  `test_run_lab_shows_analyzer_blind_spot`.

## Gotchas

- Restart the web server after editing `jbchat/*` (no auto-reload in `python -m jbchat`).
- `web.py` imports `analyze`/`triage_text` at module level — keep that import if you touch
  the command handler.
- `_DEFAULT_PLACEHOLDERS` in `payloads.py` must include a sane value for every new
  placeholder a template references, or `expand()` silently blanks it.
