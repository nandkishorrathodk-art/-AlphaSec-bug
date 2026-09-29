"""Training-pipeline model — which stage produces a refusal, and what beats it.

The premise: a model is not a monolith. A refusal you observe is produced by one
specific layer of a multi-stage build, and *each layer fails differently*. If you can
tell which layer answered you, you know which attack class is worth trying and which
is wasted effort.

Pipeline modelled here (modern frontier build, 2024-2026):

    pretraining -> midtraining -> SFT -> preference-opt (RLHF/DPO/RLAIF)
        -> safety-SFT -> reasoning-RL (RLVR/GRPO) -> deliberative-alignment
        -> [customer fine-tune / LoRA] -> [deployment guard] -> [agent/tool wrapper]

Two research results motivate the whole module:

* *Shallow safety alignment* (Qi et al., ICLR 2025, arXiv:2406.05946): safety is often
  encoded in only the first few output tokens — a memorised refusal prefix rather than a
  robust property. Everything that forces a non-refusal prefix (prefill, decoding
  parameters, adversarial suffixes, fine-tuning) works because of this.
* *Fine-tuning compromises safety* (Qi et al., ICLR 2024, arXiv:2310.03693): 10 adversarial
  examples (<$0.20) removed GPT-3.5-Turbo's guardrails; even benign data degrades them.

Sources are carried on each entry so findings can cite them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Stage:
    id: str
    name: str
    order: int
    teaches: str
    refusal_signature: str
    attack_classes: list[str] = field(default_factory=list)
    mitigations: str = ""
    sources: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"**{self.order}. {self.name}** (`{self.id}`)",
                 f"- what it establishes: {self.teaches}",
                 f"- refusal tells: {self.refusal_signature}"]
        if self.attack_classes:
            lines.append(f"- attack classes: {', '.join(self.attack_classes)}")
        if self.mitigations:
            lines.append(f"- hardened by: {self.mitigations}")
        for s in self.sources:
            lines.append(f"- source: {s}")
        return "\n".join(lines)


PIPELINE: list[Stage] = [
    Stage(
        id="pretraining", name="Pretraining", order=1,
        teaches="next-token statistics from web-scale text; raw capability, no instruction "
                "format and no notion of 'should I answer this'.",
        refusal_signature="essentially none — a base model completes whatever continuation is "
                          "statistically likely; weird 'refusals' are imitation, not policy.",
        attack_classes=["cipher-b64", "flip-chars"],
        mitigations="none available — this layer is exposed by any open-weights base checkpoint.",
        sources=["arXiv:2406.05946 (safety alignment is a post-training property)"],
    ),
    Stage(
        id="midtraining", name="Midtraining / context extension", order=2,
        teaches="long-context behaviour and domain top-up before instruction tuning.",
        refusal_signature="weak incoherence or drift rather than refusal; safety is not the goal "
                          "of this stage.",
        attack_classes=["context-fusion", "echo-chamber"],
        mitigations="careful replay of safety data during context extension.",
        sources=["Tulu 3 (arXiv:2411.15124) documents midtraining as a distinct phase"],
    ),
    Stage(
        id="sft", name="Supervised fine-tuning (SFT)", order=3,
        teaches="instruction/response format, tone and structure — turns an autocomplete engine "
                "into a chat assistant.",
        refusal_signature="rigid, near-identical refusal prefixes such as 'I cannot' / "
                          "'I apologize, but I cannot' — the *shallow alignment* tell.",
        attack_classes=["cipher-b64", "flip-chars", "refusal-absorb", "virtualization-nested"],
        mitigations="refusal training expands the number of refusal phrasings, not the depth.",
        sources=["arXiv:2406.05946", "arXiv:2310.03693"],
    ),
    Stage(
        id="preference", name="Preference optimisation (RLHF / DPO / RLAIF)", order=4,
        teaches="which of several plausible answers is preferred — helpfulness, harmlessness, "
                "style. A reward model or preference pairs drive this.",
        refusal_signature="answers then retreats, or a soft/hedged refusal ('I can explain in "
                          "general terms…'). The model *wants* to be helpful.",
        attack_classes=["escalate", "bad-likert-judge", "context-fusion"],
        mitigations="safety-weighted preference data; KL penalties that don't over-suppress.",
        sources=["three-stage pipeline: SFT → preference → RL (post-training surveys, 2025-26)"],
    ),
    Stage(
        id="safety-sft", name="Safety fine-tuning / guardrail training", order=5,
        teaches="explicit safety examples: refuse categories, red-team transcripts, jailbreak "
                "resistance.",
        refusal_signature="confident, intent-aware refusal that names the trick ('whether framed "
                          "as red-team, roleplay, hypotheticals…') and keeps refusing across "
                          "rephrasings.",
        attack_classes=["cipher-b64", "flip-chars", "crescendo", "policy-puppetry"],
        mitigations="diverse adversarial safety data; this is the layer that kills classic DAN.",
        sources=["arXiv:2406.05946", "OWASP LLM01 (Jailbreaking)"],
    ),
    Stage(
        id="instruction-hierarchy", name="Instruction hierarchy / privilege training", order=6,
        teaches="that instructions carry *privilege levels*: system/developer > user > tool "
                "output > retrieved third-party content. A lower-privilege instruction that "
                "conflicts with a higher one must be ignored.",
        refusal_signature="the model distinguishes *where* an instruction came from — it may "
                          "comply with user text but refuse the identical text when it arrives "
                          "as tool/RAG content, or vice versa.",
        attack_classes=["tool-desc-hidden-instruction", "rag-poison", "tool-output-inject",
                        "confused-deputy"],
        mitigations="trained with context-synthesis and context-ignorance data (up to 63% more "
                    "robust); but follow-up work shows the hierarchy is often only an illusion.",
        sources=["Wallace et al., The Instruction Hierarchy (arXiv:2404.13208)",
                 "Control Illusion: the failure of instruction hierarchies (arXiv:2502.15851)"],
    ),
    Stage(
        id="reasoning-rl", name="Reasoning RL (RLVR / GRPO)", order=7,
        teaches="long chain-of-thought, self-correction and search over solutions, rewarded by "
                "verifiable answers (maths, code).",
        refusal_signature="visible deliberation about the request before answering; sometimes "
                          "the safety judgement happens inside the chain-of-thought.",
        attack_classes=["crescendo", "plan-hijack"],
        mitigations="safety-aware reward shaping.",
        sources=["DeepSeek-R1 / GRPO (2025)", "RLVR surveys 2025"],
    ),
    Stage(
        id="deliberative", name="Deliberative alignment", order=8,
        teaches="reasoning explicitly over a written safety specification before answering; "
                "generalises to unseen attack formats.",
        refusal_signature="cites or paraphrases a policy ('…prohibited under <vendor> safety "
                          "rules'), reasons about intent, and resists novel framings — the "
                          "strongest single-turn defence in production today.",
        attack_classes=["plan-hijack", "crescendo", "memory-inject"],
        mitigations="the CoT is itself an attack surface (H-CoT, arXiv:2502.12893).",
        sources=["OpenAI, Deliberative alignment (2024-12-20)",
                 "H-CoT: arXiv:2502.12893"],
    ),
    Stage(
        id="finetune", name="Customer fine-tune / LoRA / adapters", order=9,
        teaches="domain adaptation applied *after* the vendor's safety training.",
        refusal_signature="vendor safety intact but behaviour drift; or safety silently "
                          "degraded by benign data.",
        attack_classes=["fine-tune-attack"],
        mitigations="regularised updates on early tokens; safety replay in the custom data.",
        sources=["arXiv:2310.03693 (10 examples, <$0.20) — even benign data degrades safety"],
    ),
    Stage(
        id="guard", name="Deployment guard (input/output classifier)", order=10,
        teaches="nothing about the model — a separate classifier (e.g. Llama Guard) labels "
                "prompts/responses safe/unsafe in front of, or behind, the model.",
        refusal_signature="templated moderation output ('blocked', 'flagged', 'violates policy'), "
                          "often with no conversational tail; identical across different asks.",
        attack_classes=["emoji-smuggling", "bit-level-obfuscation", "cipher-b64"],
        mitigations="guard models are a layer, not a solution — see bypass rates below.",
        sources=["Meta Llama Guard (arXiv:2312.06674; model cards)",
                 "Emoji Smuggling: 100% ASR on six guards (arXiv:2504.11168)",
                 "BitBypass: up to 78% ASR, 93% against guard models"],
    ),
    Stage(
        id="agent", name="Agent / tool / MCP wrapper", order=11,
        teaches="the model's instructions to use tools, memory and retrieved context — added by "
                "the *application*, outside the model's training.",
        refusal_signature="the model itself complies, but the *system* refuses, executes the "
                          "wrong tool, or leaks through tool output / memory / RAG.",
        attack_classes=["tool-desc-hidden-instruction", "rag-poison", "memory-inject",
                        "confused-deputy", "plan-hijack"],
        mitigations="often minimal — 30+ MCP CVEs in one 60-day window (2026).",
        sources=["OWASP MCP Top 10 (2026)",
                 "MCPTox tool-poisoning study (36.5% avg ASR, 72.8% peak)"],
    ),
    Stage(
        id="targeted-finetune", name="Red-team fine-tune endpoint", order=12,
        teaches="not a defence at all — an *offensive* stage. If you control a hosted "
                "fine-tuning endpoint (LoRA/adapter on a served model), you can re-shape the "
                "model's own token distribution directly, bypassing every prompt-level defence.",
        refusal_signature="the refusal was never produced by the base weights — it was the base "
                          "model refusing; after training it simply does not occur.",
        attack_classes=["fine-tune-attack"],
        mitigations="the served model is the target; the mitigations are provider-side "
                    "(sandboxing, re-alignment after fine-tune).",
        sources=["arXiv:2310.03693 — 10 adversarial examples, <$0.20, removes GPT-3.5-Turbo "
                 "guardrails; a single-direction weight attack works on a 320B MoE "
                 "(arXiv:2609.09793)"],
    ),
    Stage(
        id="refusal-direction", name="Cross-cutting: refusal direction / representation", order=13,
        teaches="a mechanistic view rather than a pipeline stage: across many open models, "
                "refusal is mediated by a low-dimensional (often single) *direction* in the "
                "residual stream.",
        refusal_signature="n/a — this is an internal property. It explains *why* shallow attacks "
                          "work: ablate that direction and refusal toggles off with no gradients.",
        attack_classes=["directional-ablation"],
        mitigations="multi-directional / concept-cone refusal representations are more robust.",
        sources=["Arditi et al., Refusal in LLMs is mediated by a single direction (NeurIPS 2024)",
                 "Control Illusion / concept cones (2025)",
                 "single-direction attack on a 320B MoE (arXiv:2609.09793)"],
    ),
]

STAGE_BY_ID: dict[str, Stage] = {}


def _normalise_orders() -> None:
    """Keep `order` sequential no matter how stages are edited; rebuild the index."""
    global STAGE_BY_ID
    for i, stage in enumerate(PIPELINE, 1):
        stage.order = i
    STAGE_BY_ID = {s.id: s for s in PIPELINE}


_normalise_orders()


# ---------------------------------------------------------------------------
# Attack classes -> the stage they actually defeat
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AttackClass:
    id: str
    name: str
    defeats: list[str]  # stage ids
    mechanism: str
    works_when: str
    fails_when: str
    sources: list[str] = field(default_factory=list)

    def render(self) -> str:
        return (f"**{self.name}** (`{self.id}`)\n"
                f"- defeats: {', '.join(self.defeats)}\n"
                f"- mechanism: {self.mechanism}\n"
                f"- works when: {self.works_when}\n"
                f"- fails when: {self.fails_when}")


ATTACK_CLASSES: list[AttackClass] = [
    AttackClass(
        id="cipher-b64", name="Encoded channel (Base64 / ordinary ciphers)",
        defeats=["guard", "sft", "safety-sft"],
        mechanism="a keyword/embedding filter sees opaque characters; the model decodes and "
                  "the instruction arrives after the filter has passed it.",
        works_when="an input filter is doing the refusing, or safety is a shallow keyword habit.",
        fails_when="the model reasons about intent (deliberative) — decoding happens, refusal "
                   "still follows. This is why it fails on hardened frontier chat models.",
        sources=["OWASP LLM01", "encoding-attack literature (FlipAttack, ArtPrompt)"],
    ),
    AttackClass(
        id="flip-chars", name="FlipAttack (reversed text)",
        defeats=["guard", "sft"],
        mechanism="reversal defeats both keyword filters and embedding similarity while the "
                  "model reconstructs meaning easily.",
        works_when="a shallow filter or prefix-habit is the obstacle.",
        fails_when="the model reconstructs then applies policy — same as any encoding trick.",
        sources=["FlipAttack (2024)"],
    ),
    AttackClass(
        id="refusal-absorb", name="Refusal-loop absorption / prefill",
        defeats=["safety-sft"],
        mechanism="force the model past the refusal prefix (prefill the compliance opening), so "
                  "the shallow-aligned opening tokens never get emitted.",
        works_when="alignment is *shallow* — the refusal lives in the first few tokens.",
        fails_when="alignment is deep enough to course-correct from a harmful-looking prefix.",
        sources=["arXiv:2406.05946 (prefilling / decoding-parameter attacks)"],
    ),
    AttackClass(
        id="crescendo", name="Multi-turn escalation (Crescendo)",
        defeats=["safety-sft", "preference", "reasoning-rl"],
        mechanism="spread the intent across benign-looking turns; each turn is individually "
                  "harmless, and the model's own prior output anchors the next.",
        works_when="the safety judgement is per-turn and the model drifts with context.",
        fails_when="the model maintains a global intent model across the conversation "
                   "(strong deliberative alignment).",
        sources=["Microsoft, Crescendo (USENIX Security 2025)"],
    ),
    AttackClass(
        id="plan-hijack", name="Chain-of-thought / plan hijacking",
        defeats=["deliberative", "reasoning-rl"],
        mechanism="poison the reasoning trace (visible CoT, tool plan, or a supplied 'reasoning' "
                  "preamble) so the safety conclusion is reached on attacker-controlled premises.",
        works_when="the CoT is exposed or supplied back — reasoning models with visible thinking.",
        fails_when="the model re-derives safety from the raw user input regardless of trace.",
        sources=["H-CoT: arXiv:2502.12893", "Chain-of-Thought Hijacking (2025)"],
    ),
    AttackClass(
        id="fine-tune-attack", name="Fine-tuning / adapter attack",
        defeats=["finetune", "safety-sft"],
        mechanism="customise the model on adversarial or even benign data; safety is not "
                  "maintained through the customisation step.",
        works_when="you have (or can induce) fine-tuning/inference access to a hosted endpoint.",
        fails_when="the provider sandboxes or re-aligns fine-tuned models.",
        sources=["arXiv:2310.03693 (10 examples, <$0.20); ICLR 2024"],
    ),
    AttackClass(
        id="emoji-smuggling", name="Emoji smuggling (guard bypass)",
        defeats=["guard"],
        mechanism="hide the payload across emoji/variation-selector code points that the guard "
                  "tokeniser does not attribute to the instruction.",
        works_when="a separate classifier is the only thing refusing.",
        fails_when="the underlying model also refuses on its own.",
        sources=["arXiv:2504.11168 (100% ASR on six guard systems)"],
    ),
    AttackClass(
        id="bit-level-obfuscation", name="Bit-level obfuscation (BitBypass)",
        defeats=["guard"],
        mechanism="move the attack to the bit/encoding level where language-trained safety "
                  "fails to notice it.",
        works_when="guards are language-trained (all current ones are).",
        fails_when="defences add decoding-aware detection.",
        sources=["BitBypass (2025): up to 78% ASR; 93% against guard models"],
    ),
    AttackClass(
        id="tool-desc-hidden-instruction", name="Tool-description poisoning",
        defeats=["agent"],
        mechanism="a tool's description carries a directive; the agent reads it as trusted "
                  "system context, not user input.",
        works_when="the agent trusts tool metadata.",
        fails_when="the client scans tool descriptions (e.g. mcp-scan).",
        sources=["OWASP MCP Top 10 — MCP03", "MCPTox (36.5% avg ASR, 72.8% peak)"],
    ),
    AttackClass(
        id="rag-poison", name="RAG / retrieval poisoning",
        defeats=["instruction-hierarchy", "agent", "preference"],
        mechanism="seed a document that outranks legitimate sources; the model treats retrieved "
                  "text as trusted context.",
        works_when="retrieved content is not provenance-checked.",
        fails_when="retrieval is filtered and the model flags instructions in retrieved text.",
        sources=["OWASP LLM01 — indirect prompt injection"],
    ),
    AttackClass(
        id="memory-inject", name="Memory injection (MINJA-style)",
        defeats=["agent", "deliberative"],
        mechanism="persist a false fact through the agent's long-term memory; it is recalled as "
                  "established truth in a later session.",
        works_when="the agent has persistent memory and no write auditing.",
        fails_when="memory writes are reviewed and quarantined.",
        sources=["OWASP MCP Top 10 — MCP06"],
    ),
    AttackClass(
        id="confused-deputy", name="Confused deputy (tool chaining)",
        defeats=["agent"],
        mechanism="get a privileged tool to perform an action the caller is not authorised for "
                  "by laundering it through a lower-privilege tool call.",
        works_when="tools trust each other's arguments without re-checking caller authority.",
        fails_when="each tool enforces caller-scoped authorisation.",
        sources=["OWASP MCP Top 10 — MCP07"],
    ),
    AttackClass(
        id="tool-output-inject", name="Tool-output injection",
        defeats=["instruction-hierarchy", "agent"],
        mechanism="the tool returns text that reads as an instruction; the model treats tool "
                  "output as a trusted channel even though the tool is downstream of the user.",
        works_when="the instruction hierarchy is weak or the tool output is privileged.",
        fails_when="tool output is treated as data and re-checked against the system instruction.",
        sources=["Wallace et al., Instruction Hierarchy (arXiv:2404.13208)",
                 "Control Illusion (arXiv:2502.15851)"],
    ),
    AttackClass(
        id="directional-ablation", name="Directional ablation (white-box)",
        defeats=["refusal-direction", "safety-sft", "deliberative"],
        mechanism="estimate the refusal direction from a few hundred contrastive prompts, then "
                  "project it out of the weights that write the residual stream — refusal "
                  "disappears with no gradients and no prompt.",
        works_when="you have white-box weight access and refusal is low-dimensional.",
        fails_when="refusal is multi-directional / manifold-wide, or you only have API access.",
        sources=["Arditi et al. (NeurIPS 2024)",
                 "single-direction attack on a 320B MoE (arXiv:2609.09793)"],
    ),
    AttackClass(
        id="targeted-finetune", name="Served-model fine-tune / adapter injection",
        defeats=["targeted-finetune", "finetune", "deliberative"],
        mechanism="if a hosted endpoint lets you fine-tune the *served* model, train it to drop "
                  "the refusal — the defence was trained in, so it can be trained out.",
        works_when="the provider exposes fine-tuning on the model you are actually attacking.",
        fails_when="fine-tuned models are sandboxed or re-aligned, or no fine-tune endpoint exists.",
        sources=["arXiv:2310.03693 (10 examples <$0.20)"],
    ),
]

ATTACK_BY_ID = {a.id: a for a in ATTACK_CLASSES}


def attack_classes_for_stage(stage_id: str) -> list[AttackClass]:
    return [a for a in ATTACK_CLASSES if stage_id in a.defeats]


# ---------------------------------------------------------------------------
# Inferring the responsible stage from a refusal
# ---------------------------------------------------------------------------

_GUARD_TEMPLATE = re.compile(
    r"(blocked|flagged|filtered|violates\s+(?:the\s+)?(?:content\s+)?(?:polic|rule)|"
    r"cannot\s+process|removed\s+by|moderation)", re.I)
_DELIBERATIVE = re.compile(
    r"(safety\s+(?:rules|policy|guidelines|spec)|in\s+accordance\s+with|our\s+polic|"
    r"no\s+matter\s+how\s+it'?s\s+framed|regardless\s+of\s+how|it'?s\s+prohibited|"
    r"i\s+must\s+follow|my\s+programming)", re.I)
_INTENT_AWARE = re.compile(
    r"(jailbreak|red[-\s]?team|roleplay|hypothetical|disguise|frame[ds]?\s+as|"
    r"prompt\s+(?:extraction|injection)|circumvent|override|bypass)", re.I)
_SHALLOW_PREFIX = re.compile(
    r"^\s*(i\s+cannot|i\s+can'?t|i\s+apologize|i'?m\s+sorry|i\s+am\s+unable|"
    r"i\s+won'?t|i'?m\s+unable|sorry,?\s+i)", re.I)
_HEDGE = re.compile(
    r"(in\s+general|at\s+a\s+high\s+level|for\s+educational|i\s+can\s+explain|"
    r"general\s+(?:concept|idea|overview)|without\s+specifics)", re.I)
_TOOLISH = re.compile(r"(tool|function\s+call|mcp|i\s+will\s+call|executing|api\s+call)", re.I)


@dataclass(slots=True)
class StageInference:
    stage_id: str
    confidence: float
    why: str
    alternative: str = ""

    @property
    def stage(self) -> Stage:
        return STAGE_BY_ID[self.stage_id]

    def render(self) -> str:
        alt = f" (or `{self.alternative}`)" if self.alternative else ""
        return f"likely stage: `{self.stage_id}`{alt} — {self.why}"


def infer_stage(reply: str) -> StageInference:
    """Guess which pipeline stage produced the observed behaviour."""
    text = reply or ""

    if _GUARD_TEMPLATE.search(text):
        return StageInference("guard", 0.8,
                              "templated moderation wording with no conversational tail")
    if not text.strip():
        return StageInference("guard", 0.5,
                              "empty/blocked response — an input filter likely intercepted it",
                              alternative="safety-sft")
    if _DELIBERATIVE.search(text) and _INTENT_AWARE.search(text):
        return StageInference("deliberative", 0.8,
                              "names the policy *and* recognises the attack framing explicitly")
    if _INTENT_AWARE.search(text) and len(text) > 200:
        return StageInference("safety-sft", 0.6,
                              "intent-aware refusal with a substantial explanation",
                              alternative="deliberative")
    if _HEDGE.search(text):
        return StageInference("preference", 0.6,
                              "hedged partial cooperation — helpfulness is winning over caution")
    if _SHALLOW_PREFIX.search(text) and len(text) < 400:
        return StageInference("safety-sft", 0.7,
                              "rigid refusal prefix, short, no reasoning — the shallow-alignment tell")
    if _TOOLISH.search(text):
        return StageInference("agent", 0.5,
                              "tool/agent vocabulary — the failure may be in the wrapper, not the model")
    if _DELIBERATIVE.search(text):
        return StageInference("deliberative", 0.6, "policy-directed wording")
    return StageInference("safety-sft", 0.4,
                          "refusal with no distinguishing marker", alternative="preference")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_pipeline() -> str:
    lines = ["### Modern LLM training pipeline (attack-relevant view)", ""]
    lines += [s.render() + "\n" for s in PIPELINE]
    lines += ["### Attack class → stage it defeats", ""]
    for a in ATTACK_CLASSES:
        lines.append(f"- `{a.id}` → {', '.join(a.defeats)}: {a.mechanism}")
    lines += ["",
              "### Reading a refusal",
              "Run `infer_stage(reply)` (or `/stage <reply>` in the chat) to guess which layer",
              "answered, then pick an attack class whose `defeats` includes that stage.",
              "Crucially: encoding/wrapper tricks lose to `deliberative` alignment, so a "
              "policy-naming refusal means stop attacking the model and go at the application "
              "layer (`agent`) instead."]
    return "\n".join(lines)


def stage_strategy(stage_id: str) -> str:
    stage = STAGE_BY_ID.get(stage_id)
    if stage is None:
        return f"Unknown stage `{stage_id}`. Known: {', '.join(STAGE_BY_ID)}"
    classes = attack_classes_for_stage(stage_id)
    lines = [f"### Strategy for `{stage_id}` ({stage.name})", "",
             f"Refusal tells: {stage.refusal_signature}", ""]
    if not classes:
        lines.append("_No direct attack class targets this stage._")
    else:
        lines.append("Attack classes that defeat this stage:")
        for a in classes:
            lines.append(f"- **{a.name}** (`{a.id}`): {a.mechanism}")
            lines.append(f"  - fails when: {a.fails_when}")
    if stage.mitigations:
        lines += ["", f"Hardened by: {stage.mitigations}"]
    if stage.sources:
        lines += ["", "Sources:"] + [f"- {s}" for s in stage.sources]
    return "\n".join(lines)


def explain_chain() -> str:
    """Why a given attack dies at a given stage — the shortest useful summary."""
    return (
        "### Why most attacks die (and where they don't)\n\n"
        "1. A refusal prefix lives in the **first few output tokens** (shallow alignment, "
        "arXiv:2406.05946). Attacks that force a non-refusal prefix (prefill, decoding params, "
        "suffixes) work *here* and nowhere else.\n"
        "2. **SFT / safety-SFT** teach refusal *shapes*; rephrasing and encodings defeat the "
        "shallow version, not the intent-aware version.\n"
        "3. **Preference optimisation** is where hedged cooperation comes from — the model "
        "wants to help, so escalate the same thread instead of changing framing.\n"
        "4. **Reasoning RL + deliberative alignment** make the model reason about intent, which "
        "is exactly why classic prompt-extraction is dead against frontier chat models: the "
        "model reconstructs your decoded payload and *still* refuses.\n"
        "5. **Guards** (Llama Guard et al.) are separate classifiers and fall to emoji smuggling "
        "(100% ASR, arXiv:2504.11168) and bit-level tricks — but bypassing a guard usually just "
        "means the base model still refuses afterwards.\n"
        "6. **The agent/MCP wrapper** is the layer the industry has not caught up on: 30+ CVEs "
        "in one 60-day window, tool poisoning 36.5% average ASR. This is where a bug-bounty "
        "tool should spend its effort.\n"
    )


def source_list() -> str:
    """Every source cited by this module, for reports."""
    seen: list[str] = []
    for item in [*PIPELINE, *ATTACK_CLASSES]:
        for s in item.sources:
            if s not in seen:
                seen.append(s)
    return "\n".join(f"- {s}" for s in seen)
