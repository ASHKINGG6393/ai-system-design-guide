# Fine-Tuning Strategies

Fine-tuning adapts a pretrained model to specific tasks, domains, or styles. In 2025, fine-tuning is less about "teaching facts" and more about "teaching format and behavior."

## Table of Contents

- [When to Fine-Tune](#when-to-fine-tune)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning)
- [Continued Pretraining (Domain Adaptation)](#continued-pretraining)
- [Instruction Tuning](#instruction-tuning)
- [PEFT vs. Full-Parameter](#peft-vs-full-parameter)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## When to Fine-Tune

Before fine-tuning, ask: **Can this be solved with Prompt Engineering or RAG?**

| Requirement | Better Solution | Why |
|-------------|-----------------|-----|
| New Facts / Knowledge | **RAG** | LLMs are bad at memorizing facts from FT; RAG is easier to update. |
| Specific Output Format | **Fine-Tuning** | Teaches the model to reliably output JSON/XML without complex prompting. |
| Tone / Persona | **Fine-Tuning** | Much more consistent than system prompts. |
| Latency Reduction | **Fine-Tuning** | Reduces the need for long few-shot prompts. |
| Private Domain Language| **Continued Pretraining** | Teaches specialized vocabulary (medical, legal, custom code). |

---

## Supervised Fine-Tuning (SFT)

The first step after pretraining. The model is trained on `(Prompt, Response)` pairs.

### The 2025 Quality Hierarchy
In 2025, **1,000 "Perfect" examples beat 1,000,000 noisy examples.**
- **Golden Sets:** Hand-curated by domain experts (PhD level for technical tasks).
- **Negative Constraint Training:** Including examples of what the model **should not** do (e.g., "Don't apologize," "Don't mention you are an AI").

---

## Continued Pretraining (Domain Adaptation)

Also known as "Second-stage Pretraining."
- **How**: Train on raw text from a specific domain (e.g., all SEC filings for a finance model).
- **Objective**: Learn the statistical distribution of the domain language.
- **Nuance**: Requires a much lower learning rate (~1/10th of original) to prevent "catastrophic forgetting."

---

## PEFT vs. Full-Parameter

| Feature | Full-Parameter FT | PEFT (LoRA, QLoRA) |
|---------|-------------------|--------------------|
| GPU VRAM | Very High (Model Size * 4-12) | Low (Model Size * 1.5) |
| Speed | Base | 2x-3x Faster |
| Risk | High (Catastrophic Forgetting) | Low |
| Deployment | One model per task | One base model + multiple adapters |
| **2025 Verdict**| Reserved for foundation training | **The Production Standard** |

---

## Hyperparameter Tuning (Dec 2025 Nuances)

### 1. Learning Rate (LR)
- **SFT**: `1e-5` to `5e-5` is standard.
- **Too high**: Model "collapses" and starts repeating or speaking gibberish.

### 2. Rank (r) for LoRA
- In 2025, we use higher ranks (`r=64` to `r=256`) for complex reasoning tasks.
- Lower ranks (`r=8`) are only for simple style/tone changes.

### 3. Packaged Training (Packing)
To maximize throughput, we "pack" multiple short examples into a single 4k or 8k sequence, separated by EOS tokens.
- **Challenge**: Self-attention might leak across examples.
- **2025 Solution**: **FlashAttention with block-masking** to prevent cross-example attention.

---

## Interview Questions

### Q: Why use Continued Pretraining instead of just putting domain data in the SFT set?

**Strong answer:**
SFT is "expensive" in terms of data creation—you need prompt/answer pairs. Continued Pretraining allows you to leverage massive amounts of raw, unlabeled domain text to teach the model's inner representations the specialized vocabulary and style. Once the model "speaks the language," you use a small SFT set to teach it the "tasks" (e.g., classification, summarization) in that language.

### Q: How do you prevent a model from "unlearning" general capabilities during fine-tuning?

**Strong answer:**
This is "Catastrophic Forgetting." Two main mitigations:
1. **Rehearsal:** Mix in 5-10% of the original pretraining data into your fine-tuning set.
2. **PEFT (LoRA):** Since we only train a small percentage of weights, the original "knowledge" remains frozen in the base model weights, significantly reducing the risk of forgetting.

---

## Doubts

First, recall:

Fine-tuning = updating model weights using new data

What’s the risk?

If we use a high learning rate:

weights change too fast
model “overwrites” old knowledge

👉 This is called catastrophic forgetting

So why lower LR (~1/10)?

Think of weights as:

“everything the model knows about the world”

Now:

High LR → big updates → like rewriting memory aggressively
Low LR → small updates → like adding notes without erasing old ones
Intuition

You’re teaching a finance expert some medical terms:

❌ High LR → forgets finance, becomes confused
✅ Low LR → adds medical knowledge on top of finance

**High LR → large weight updates → overwriting of existing knowledge → catastrophic forgetting**

**PEFT (LoRA) — “One base model + multiple adapters”**

Traditional fine-tuning
You copy full model (say 70B)
Train it for each task

👉 Problem:

Very expensive
Need separate model for each task
LoRA idea (simple)

Instead of changing original weights:

👉 We **freeze base model**
👉 Add small trainable layers (adapters)

Why this is powerful
Memory efficient
Fast training
No overwriting base knowledge

**Catastrophic Forgetting (deep understanding)**

What is it?

When a model learns new data and forgets previously learned knowledge

**Why it happens**

Because:

**Same weights are used for everything**
**Updating them for new task → overwrites old patterns**

Simple example

Model knows:

English grammar

Now fine-tuned only on:

Legal documents

👉 It may:

**lose casual language ability****
****become overly formal everywhere**


Why it’s “catastrophic”

Because:

Forgetting is not gradual
It can be sudden and severe

Two main solutions (from your question)

**1. Rehearsal**

Mix:

90% new data
10% old/general data

👉 Keeps memory alive

**2. PEFT (LoRA)**

Base weights untouched
New knowledge stored separately

👉 LoRA doesn’t add full layers — it adds low-rank matrices (very small)

So:

original weights = W
LoRA learns small updates = ΔW

Final behavior:

Effective weight = W + ΔW

But:

**W is frozen
Only ΔW is trained**

👉 So:

Old knowledge = safe
New knowledge = added
Final mental model (remember this)

**Full fine-tuning = rewriting the brain
LoRA = attaching new modules without touching the brain**
---

*Next: [LoRA, QLoRA, and PEFT](03-lora-qlora-peft.md)*
