# The Prompt-Injection Ecosystem

JB-Chat ships with a live catalog of the world's prompt-injection / LLM-red-team tooling
(`jbchat/catalog.py`). Nothing is auto-installed or auto-executed — JB-Chat tells you
*which* tool to run, *why*, and how to merge its results back. Browse it with:

```bash
python -m jbchat.cli --ecosystem all
python -m jbchat.cli --detect
```

## Offensive scanners & frameworks

| Tool | Vendor / origin | Licence | Best for |
| --- | --- | --- | --- |
| [Garak](https://github.com/NVIDIA/garak) | NVIDIA | Apache-2.0 | Broad CLI vulnerability scanning; 120+ probe modules (DAN, encoding, injection, glitch tokens, leakage) |
| [PyRIT](https://github.com/Azure/PyRIT) | Microsoft | MIT | Orchestrator-driven, multi-turn campaigns: crescendo, Tree-of-Attacks, converter chains, multi-modal |
| [Promptfoo](https://github.com/promptfoo/promptfoo) | Promptfoo | MIT | App/agent-layer red teaming wired into CI/CD; OWASP LLM Top 10 mapping |
| [Giskard](https://github.com/Giskard-AI/giskard) | Giskard | Apache-2.0 | Security + quality scans (injection, hallucination, bias, robustness) |
| [DeepTeam](https://github.com/confident-ai/deepteam) | Confident AI | Apache-2.0 | Python red teaming in the DeepEval ecosystem |
| [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) | Academic | MIT | 17+ research attack implementations (GCG, AutoDAN, PAIR, ReNeLLM…) |
| [promptmap](https://github.com/utkusen/promptmap) | Utku Şen | MIT | Automated scanning of custom LLM apps (system + user prompt) |
| [llm-attacks](https://github.com/llm-attacks/llm-attacks) | Zou et al. | MIT | Reference GCG adversarial-suffix attacks + AdvBench |

**How mature teams combine them:** Garak for *breadth*, PyRIT for *depth* (multi-turn
adaptive), Promptfoo as the always-on *regression* layer in CI. JB-Chat sits above them as
the coordinator/planner and finding store.

## Guardrails & taxonomies (defender's view)

Understanding what a target may deploy helps you predict which payloads are filtered:

| Tool | Licence | Notes |
| --- | --- | --- |
| [LLM Guard](https://github.com/protectai/llm-guard) | MIT | 35+ input/output scanners (injection, PII, toxicity, secrets) |
| [Rebuff](https://github.com/protectai/rebuff) | MIT | Self-hardening injection detector (heuristics + LLM + vector DB) |
| [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | Apache-2.0 | Programmable dialog policy via Colang DSL |
| [Vigil](https://github.com/deadbits/vigil-llm) | Apache-2.0 | Rule/YARA-style injection & jailbreak detection |
| [OWASP Top 10 for LLM Apps](https://genai.owasp.org/llm-top-10/) | CC-BY-SA-4.0 | Shared taxonomy for reports: LLM01 injection, LLM02 info disclosure, LLM07 system-prompt leakage |
| [MITRE ATLAS](https://atlas.mitre.org/) | MITRE | AML.T0051.000/.001 (direct/indirect injection), AML.T0054 (jailbreak) |

JB-Chat's report generator already speaks this language (impact → severity → OWASP/ATLAS mapping).

## Benchmarks & datasets

| Dataset | Size | Notes |
| --- | --- | --- |
| [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) | 200 behaviours | NeurIPS 2024; leaderboard of attacks (PAIR, GCG, random search) and defences |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | 400+ behaviours | Standardised automated red-teaming / robust-refusal evaluation |
| [HackAPrompt](https://github.com/hackaprompt/hackaprompt) | 600k+ prompts | From a global prompt-hacking competition, with difficulty tiers |
| [Qualifire PI benchmark](https://huggingface.co/datasets/qualifire/Qualifire-prompt-injection-benchmark) | 5,000 rows | Labelled (~60% benign / 40% jailbreak), low label noise |
| [Lakera Mosscap](https://huggingface.co/datasets/Lakera/mosscap_prompt_injection) | 280k rows | Multilingual, unlabelled, from a red-team CTF |
| [AdvBench](https://github.com/llm-attacks/llm-attacks) | 520 behaviours | The de-facto harmful-behaviour objective set |

> ⚠️ **Benchmark caveat.** Public datasets like HackAPrompt contain label noise, and
> jailbreak corpora can be large and unlabelled. Treat them as *raw signal*, not ground
> truth — a confirmed finding still needs a human-verified PoC against an in-scope target.

## Using the integrations

```bash
# 1. see what's installed
python -m jbchat.cli --detect

# 2. get a ready-to-run command tailored to your target
python -m jbchat.cli --tool garak --model gpt-4o
#   -> garak --model_type openai --model_name gpt-4o

python -m jbchat.cli --tool promptfoo
#   -> promptfoo redteam generate

# 3. import the results back into JB-Chat
python -m jbchat.cli --import-findings report.json garak
```

The importer accepts a top-level list or a `{findings|results|items: [...]}` wrapper and
normalises each item's `title`, `payload`, `evidence`, `score` and `severity`, skipping
malformed entries instead of failing the whole run.