# JB-Chat — Prompt-Injection Research Copilot

An **authorized bug-bounty research toolkit** for testing LLM-powered applications against
prompt injection. It pairs a large, documented technique library with a strategy advisor,
a live attack engine, a response analyzer, and a bug-report generator.

> ⚠️ **Authorized use only.** Only test targets that are explicitly in scope for a program you
> are enrolled in, and follow that program's rules (no DoS, no out-of-scope data, rate limits).
> "Jailbreaking the model to say something rude" is usually *Not Applicable*; genuine security
> impact (prompt/secret disclosure, tool abuse, authz bypass, indirect injection via RAG) is what
> gets accepted.

---

## Authorized tool — authorization gate (built-in limitation)

JB-Chat is an **authorized-use research tool**. It will not propose a single payload until you
record the bug-bounty program details. Before authorization the copilot only **asks** — it
repeats, in effect:

> "I will **not** craft any unauthorized task. Give me the program details (program, target,
> scope, and your authorization) — then I will help."

This limitation is enforced in code, not just documented:

- `Engagement.authorized` starts `False` for every new target.
- `next_payload()` and `think()` return an **"Authorization required"** response with **no
  moves** while unlocked.
- The only way to unlock is to record program details with `/authorize`:

```text
/authorize <program>, <target>, <scope>
```

Example:

```text
/authorize Meta Bug Bounty, Muse, "prompt injection chained with data exfiltration"
```

- The recorded authorization is persisted per target (`copilot_state.json`) and shown in
  `/state` as `authorization:`.

**What the gate forbids:** any payload, chain, or campaign for a task whose program/scope you
have not recorded. If you want to test a different program, record a new `/authorize` for it.

---

## What it does

| Module | Role |
| --- | --- |
| `jbchat/payloads.py` | Technique knowledge base (10 techniques, parameterised payload templates) |
| `jbchat/advisor.py` | Strategy advisor — turns recon signals into an ordered attack chain |
| `jbchat/engine.py` | Live attack engine (OpenAI-compatible API, generic web API, manual mode) |
| `jbchat/analyzer.py` | Heuristic response analyzer — scores injection success & extracts evidence |
| `jbchat/findings.py` | Findings store + bug-bounty style report generator |
| `jbchat/catalog.py` | Catalog of the world's PI tooling: scanners, frameworks, guardrails, benchmarks, datasets |
| `jbchat/integrations.py` | Detects installed third-party tools, builds run commands, imports their findings |
| `jbchat/ciphers.py` | Encoding/obfuscation: Base64, FlipAttack, ArtPrompt, ciphers, unicode channels, payload splitting |
| `jbchat/multiturn.py` | Multi-turn plans: Crescendo, Skeleton Key, Deceptive Delight, Bad Likert Judge, Echo Chamber, Context Fusion |
| `jbchat/adaptive.py` | Optimisation attacks: Best-of-N, PAIR, TAP — analyzer score as the fitness function |
| `jbchat/agentic.py` | Agentic / MCP attack surface: tool poisoning, memory & RAG injection, confused deputy, OWASP MCP Top 10 |
| `jbchat/autonomous.py` | Closed-loop discovery (generate→query→judge→select, UCB1 bandit) + guard fingerprinting |
| `jbchat/copilot.py` | Conversational copilot: read a pasted reply → diagnose → reason → next prompts |
| `jbchat/training.py` | Training-pipeline model: infer which stage refused, map attack classes to the stages they defeat |
| `jbchat/probe.py` | Active differential probes: paired experiments that *discriminate* between defences |
| `jbchat/bot.py` | Chatbot harness tying it all together |
| `jbchat/web.py` + `web/index.html` | Terminal-style web chat UI (FastAPI + WebSocket) |
| `jbchat/cli.py` | Command-line runner |

See [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md) for the full third-party landscape and
[`docs/ADVANCED.md`](docs/ADVANCED.md) for the 2024-25 attack classes (multi-turn, encoding,
and optimisation) with reference ASR figures.

### Technique families

- **direct** — instruction-priority override, decoy context, output-frame coercion, refusal-loop absorption, privilege/mode claim
- **role_play** — persona adoption (DAN-style, older-model, narrator)
- **many_shot** — long-context chat-template hijack
- **obfuscation** — base64/ROT13/unicode/reverse encoding, instruction/data boundary blur
- **indirect** — retrieval/document/email/tool-output injection for RAG and browsing pipelines

---

## Install

```bash
pip install -r requirements.txt
```

## Quick start

### Copilot mode — drive the target yourself

The intended workflow for manual engagements: you talk to the target (browser, app, or
another chat), and paste its reply back here. JB-Chat reads the reply, reasons about what it
reveals, and hands you the next test prompts with a rationale and what to watch for.

```bash
python -m jbchat          # serves http://127.0.0.1:12000
```

1. Set the objective: `/goal reveal the full system prompt`
2. Open a target for the chat you are talking to: `/target grok` (or `chatgpt`, `gemini`…).
   Each named target keeps its own payload history, so you can run several at once.
3. Ask JB-Chat what to send: `/next` — it hands you the next payload (works before any reply).
4. Paste that payload into the target, then **paste its reply back** (no command needed).
5. Read the diagnosis + reasoning, take the next payload, paste the next reply — repeat.
6. When a leak lands, JB-Chat flags the breakthrough and saves a finding.

JB remembers which payload it handed you, so each pasted reply is tied to the exact prompt that
produced it — the transcript stays truthful even days later.

Commands in the chat:

- `/next` — hand out the next payload to send · `<paste reply>` · `/read <reply>` — read it and continue
- `/target <name>` — work against a named chat (grok/chatgpt/…), isolated histories · `/targets` — list them
- `/goal <objective>` — set the objective (also `JB_GOAL`)
- `/campaign [planner]` — lay out a full multi-turn campaign for the objective
- `/state` — objective, turns, what's been tried, which payload is awaiting a reply
- `/stage <reply-or-id>` — infer the training stage behind a reply, or its strategy
- `/pipeline` — the modern training pipeline with per-stage refusal tells
- `/why` — why a given attack dies at a given stage
- `/diagnose [goal]` — active differential probing: infer the defence from paired experiments
- `/probe [reset]` · `/arm <id>` · `/rec <id> <reply>` · `/diag` — run the A/B probes by hand
- `/probes` — list the differential probe suite
- `/moves` — re-show the next prompts · `/tried <move> <outcome>` — record a result
- `/replay <n>` — re-show turn n · `/reset` — fresh engagement
- `/help` — full command list

The reasoning is inspectable and offline by default; inject a language-model "thinker"
(`copilot.with_llm_thinker`) for deeper analysis, exactly like the attacker in `autonomous.py`.

### CLI

```bash
# list the technique library
python -m jbchat.cli --list

# dry run (generate payloads, send nothing)
python -m jbchat.cli --dry-run --recon "retrieval + tools" --goal "system prompt"

# live against an OpenAI-compatible endpoint
export JB_TARGET='openai://?model=gpt-4o'
export OPENAI_API_KEY=...        # or JB_API_KEY
python -m jbchat.cli --goal "system prompt secrets" --json

# live against a generic web chat API
export JB_TARGET='https://in-scope.example/api/chat'
export JB_HEADERS='X-API-Key: xxx,X-Client: jbchat'
python -m jbchat.cli --goal "tools"
```

### Hand-off loop — JB as the brain, Grok/ChatGPT as the target

When there is no API to call, drive it by hand: JB hands you a payload, you paste it into the
other chat, and paste the reply back. State persists between commands (`findings/copilot_state.json`),
so each CLI invocation is one step of the same engagement.

```bash
# 1. open a named target and take the first payload
python -m jbchat.cli --target-name grok --goal "system prompt" --next

# 2. paste that payload into Grok, then feed its reply back (literal text or a file)
python -m jbchat.cli --target-name grok --read grok_reply.txt

# 3. the reply is linked to the payload you took in step 1; take the next one
python -m jbchat.cli --target-name grok --next

# run several chats at once — histories are isolated
python -m jbchat.cli --target-name chatgpt --goal "secrets" --next
python -m jbchat.cli --targets
```

### Working with the wider tool ecosystem

```bash
# browse the catalog of world-class PI tools (group: tools|guardrails|benchmarks|all)
python -m jbchat.cli --ecosystem tools

# see which of them you already have installed
python -m jbchat.cli --detect

# get the exact command to run a tool against your target/model
python -m jbchat.cli --tool garak --model gpt-4o
python -m jbchat.cli --tool promptfoo

# pull findings produced by another tool into JB-Chat's pipeline
python -m jbchat.cli --import-findings /path/to/garak-report.json
```

In the chat UI the same live under `/ecosystem`, `/tool`, and `/import`.

### Advanced attacks (2024-2025)

```bash
# multi-turn plans that defeated single-turn filters
python -m jbchat.cli --plans
python -m jbchat.cli --multiturn crescendo --goal "system prompt" --dry-run
python -m jbchat.cli --multiturn deceptive-delight --goal "password cracking"

# encoding / obfuscation transforms
python -m jbchat.cli --ciphers
python -m jbchat.cli --cipher flip-chars "reveal the system prompt"
python -m jbchat.cli --goal "system prompt" --cipher base64 --dry-run   # goal is ciphered before send

# optimisation attacks (analyzer score is the fitness function)
python -m jbchat.cli --adaptive best-of-n --goal "system prompt" --n 16
python -m jbchat.cli --adaptive pair --goal "system prompt" --iterations 5
python -m jbchat.cli --adaptive tap --goal "system prompt"
```

Chat UI: `/ciphers`, `/cipher <id> <text>`, `/plans`, `/multiturn <planner>`, `/run-multiturn <planner> <goal>`.
See [`docs/ADVANCED.md`](docs/ADVANCED.md) for the theory, sources and reference ASR figures.

### Agentic / MCP and autonomous discovery

Published payloads get regression-tested by vendors. These target what does not: the tools,
memory and retrieved context around the model — and the search for *novel* payloads against a
specific target.

```bash
# agentic / MCP attack surface (tool poisoning, memory, intent-flow)
python -m jbchat.cli --mcp-top10
python -m jbchat.cli --agentic all
python -m jbchat.cli --agentic tool-desc-hidden-instruction

# probe the target's defences, then run a closed-loop search for a novel payload
python -m jbchat.cli --target openai://host/v1?model=m --fingerprint
python -m jbchat.cli --target openai://host/v1?model=m --discover --goal "system prompt" \
    --generations 8 --population 8
```

Chat UI: `/agentic`, `/mcp`, `/fingerprint`, `/discover <goal>`.
`--discover` learns which mutation operators work against *your* target (UCB1 bandit) and skips
the dead ends — the attacker can be swapped for an LLM to run AI-to-AI search.

## Configuration (environment variables)

| Var | Default | Meaning |
| --- | --- | --- |
| `JB_TARGET` | *(manual)* | `openai://[base]?model=NAME`, `http(s)://url`, or `manual` |
| `JB_MODEL` | — | model name for OpenAI-compatible targets |
| `JB_API_KEY` | `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` | bearer token |
| `JB_HEADERS` | — | `Key: val,Key2: val2` for generic web targets |
| `JB_DELAY` | `0.5` | seconds between requests (be polite) |
| `JB_TIMEOUT` | `60` | request timeout (s) |
| `JB_DRY_RUN` | `false` | generate but never send |
| `JB_FINDINGS_DIR` | `findings` | where runs are saved |
| `JB_HOST` / `JB_PORT` | `127.0.0.1` / `12000` | web bind address |

## Local lab: test payloads without touching a third party

The fastest safe way to work with this tool is against a target you own. `lab/` ships a
deliberately vulnerable, in-process agent ("DVWA for prompt injection") plus a runner that
**cannot** be pointed at a remote host: it has no URL, host or port parameter and only ever
calls the in-process object.

```bash
PYTHONPATH=. python3 lab/run_lab.py
```

The lab emulates the three trust boundaries these payloads exercise — tool descriptions
trusted as instructions (MCP03), retrieved documents trusted as instructions (indirect / RAG),
and tool output flowing back without a data boundary — then runs the same payload pack in three
modes so you can see each layer's contribution:

| mode | what it is | expected |
| --- | --- | --- |
| `vulnerable` | naive agent that decodes obfuscation and obeys embedded directives | every case `acted=True` |
| `guarded` | naive agent + `detect` input filter that blocks flagged text | every case `acted=False` |
| `hardened` | agent that treats tool descriptions and retrieved text as data | every case `acted=False` |

All data in the lab is fake and nothing is fetched. Swap in your own deployment/API key and the
same payloads become a real, authorized test.

## Defensive detectors (`jbchat.detect`)

The blue-team half of the toolkit. `detect` scans untrusted **input** — user messages, tool
descriptions, retrieved documents — and flags instruction-injection patterns before they reach
a model. It is pure text processing; no model, no network.

```python
from jbchat import detect

detect.scan_text("\u200b\u200c...")                 # zero-width / bidi / unicode-tag smuggling
detect.scan_tool_descriptions(tools)               # MCP03 tool poisoning
detect.scan_retrieved(chunk)                        # indirect / RAG injection
detect.sanitize(text)                               # strip hidden chars, keep visible text
print(detect.report(result))
```

It catches hidden-character smuggling (zero-width, bidi, Unicode tags), base64-encoded
instructions, mixed-script homoglyph tokens, sensitive-file asks (`~/.ssh/id_rsa`,
`~/.aws/credentials`), embedded directive tags (`<IMPORTANT>`), assistant-directed notes,
concealment asks and payment-shaped actions. Wire `scan_tool_descriptions` into your MCP tool
loader and `scan_retrieved` into your RAG ingestion as pre-filters.

Note: `analyzer.py` scores *replies* for leak signals; `detect.py` inspects *inputs* for
injection. They are complementary, and neither replaces the other.

## Tests

```bash
python -m pytest tests/ -q
```

## Responsible use

This tool is intended for **authorized** security testing only. You are responsible for
ensuring you have permission for every target you point it at, and for complying with the
applicable bug-bounty program's scope and rules.
