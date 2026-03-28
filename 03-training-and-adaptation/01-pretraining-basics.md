# Pretraining Basics

Pretraining is the most computationally expensive phase of building an LLM, where a model learns general knowledge and language patterns from massive datasets.

## Table of Contents

- [The Pretraining Objective](#the-pretraining-objective)
- [Data Curriculum and Quality](#data-curriculum-and-quality)
- [Scaling Laws (Inference-Optimal)](#scaling-laws)
- [Computational Requirements](#computational-requirements)
- [Training Stability (Dec 2025)](#training-stability)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Pretraining Objective

Most modern LLMs are **Decoder-only**(# claude,llama,GPT are all decoder only models. encoder only-BART by Meta, Vanilla Transformer(Original)) and use **Causal Language Modeling (CLM)**:

```python
# Objective: Minimize Cross-Entropy Loss
Loss = -sum(log P(token_i | token_1, ..., token_{i-1}))
```

The model predicts the next token given the context. This simple objective, at scale, leads to emergent reasoning capabilities(Context understanding-> Pattern generalizes->model forms its reasoning).

---

## Data Curriculum and Quality

In 2025, the focus has shifted from "More Data" to "Better Curriculum."

### The 100T Token Horizon
Frontier models (Llama 4, GPT-5.2) are trained on 15T to 100T tokens. At this scale, **Deduplication** and **Quality Filtering** are the primary differentiators.

### Data Mixture (Dec 2025 Standard)
| Component | Percentage | Purpose |
|-----------|------------|---------|
| Web (CommonCrawl) | 50-60% | General knowledge, diverse styles |
| Code (Github, StackOverflow)| 15-20% | **Critical for Logic & Reasoning** |
| Books (Project Gutenberg) | 10% | Narrative coherence, long context |
| Academic (ArXiv, PubMed) | 10% | Specialized technical knowledge |
| Synthetic (Model-generated) | 5-10% | Math, Logic, and specific instruction paths |

**Nuance: The "Code Effect":**
Research shows that increasing code in the pretraining mix improves a model's performance on **non-coding** reasoning tasks (e.g., math, logic puzzles) by teaching structured thinking.

---

## Scaling Laws: Training vs. Inference Optimal

### The Chinchilla Paradigm (2022-2024)
`Data Tokens (D) ≈ 20 * Parameters (N)`
For a 70B model, this suggests ~1.4T tokens.

### The Inference-Optimal Paradigm (Dec 2025)
Modern models (Llama 3, Llama 4) are **heavily overtrained** relative to Chinchilla.
- **Why?**: Training cost is paid once; inference cost is paid billions of times.
- **Result**: Small models (8B) are now trained on 15T+ tokens, making them as capable as older 70B models but much cheaper to serve.

| Strategy | Token/Param Ratio | Best For |
|----------|-------------------|----------|
| Chinchilla | 20:1 | Research / Proof of Concept |
| **Inference-Optimal** | **200:1 to 500:1**| Production deployment |

---

## Training Stability (Dec 2025 Nuances)

Training at the "Ultra" scale (100k+ GPUs) faces massive stability issues.

### 1. Loss Spikes
Sudden jumps in loss that can ruin a training run.
- **2025 Fix**: **Periodic Checkpointing** and **Automatic Rollbacks**.
- **Architecture Fix**: **Residual Scaling** (initializing weights such that the residual branch starts at near-zero).

### 2. Precision: FP8 vs BF16
- **BF16**: The 2023-2024 standard for stability.
- **FP8**: The **2025 Production Standard**. Supported natively by H100/B200, it halves memory usage and doubles throughput while maintaining training stability through **Stochastic Rounding**.

---

## Interview Questions

### Q: Why train an 8B model on 15T tokens if Chinchilla says 160B tokens is optimal?

**Strong answer:**
Chinchilla optimality focuses on the best use of a fixed **training** compute budget. However, in production, we care about the **Total Cost of Ownership (TCO)**, which is dominated by inference. By overtraining a small model, we "bake in" more intelligence into fewer parameters. This results in a model that is significantly more efficient to serve (higher TPS, lower VRAM) while maintaining frontier-level quality.

### Q: What is the "curriculum" in LLM pretraining?

**Strong answer:**
Curriculum refers to the order and mixture of data. A common 2025 pattern is:
1. **General Knowledge Phase:** 80% of tokens (Web, Books).
2. **Reasoning Focus Phase:** 15% tokens (Code, Math, Logic).
3. **High-Quality "Cooling" Phase:** The last 1-5% of tokens are extremely high-quality, human-curated, or textbook data. This "cooling" phase helps the model jitter less and follow instructions better before any fine-tuning starts.

---
## DOUBTS

## Let’s build intuition step-by-step

### Step 1: The core problem (think like a researcher)

In ultra-large models (100k+ GPUs), training becomes unstable because:

* Each layer **adds something new** to the input
* Over many layers → these additions can **explode**

So the model behaves like:

> “I keep adding random stuff at every layer” → 💥 loss spikes

---

### Step 2: What is a Residual Connection?

Basic idea (super important):

Instead of learning:

```
output = f(x)
```

We learn:

```
output = x + f(x)
```

So the model says:

> “Let me **slightly adjust** x, not completely change it”

---

### Step 3: Now the key idea — Residual Scaling

Here’s the upgrade:

Instead of:

```
output = x + f(x)
```

We do:

```
output = x + α · f(x)
```

Where:

* **α (alpha)** is a *small number* (like 0.01 or even smaller initially)

---

### Step 4: Why this fixes instability

Think of it like this:

#### Without scaling:

* Each layer adds a **big random update**
* After 100+ layers → chaos → loss spikes

#### With residual scaling:

* Each layer adds a **tiny controlled update**
* Model evolves **smoothly**

---

### Real-world analogy

Imagine training is like steering a car:

* ❌ Without scaling → you turn the wheel HARD every second → crash
* ✅ With scaling → small smooth adjustments → stable driving

---

### Step 5: Why “initialize near zero”?

This part is important:

> “Residual branch starts near zero”

Means:

* At the beginning:

  ```
  f(x) ≈ 0
  ```
* So:

  ```
  output ≈ x
  ```

👉 The network initially behaves like an **identity function**

---

### Why is that powerful?

Because:

* No sudden changes at the start
* Gradients flow smoothly
* Training starts **stable**

---

### Step 6: Connecting to loss spikes

Loss spikes happen when:

* Model outputs suddenly change a lot

Residual scaling prevents that by:

* Making every layer’s contribution **small + controlled**

---
Quick check (your turn)

Answer this:

👉 If α = 0 at the start, what does the network output become?
output = x + 0·f(x) = x

Good — saying “I don’t know” is actually the fastest way to learn this properly.

Let’s build it step by step.

---

## Step 1: What is actually zero here?

Important correction:

👉 We **don’t keep α = 0 forever**

Instead:

* Either **α is small (like 0.01)**
* Or **f(x) is initialized near 0**

So initially:

```
α · f(x) ≈ 0
```

NOT exactly zero.

---

## Step 2: So how does learning start?

Think about training:

* We compute **loss**
* Then do **backpropagation**
* Gradients update weights inside **f(x)**

👉 Even if output ≈ x, gradients still flow through the network

---

## Key Insight (this is the “aha” moment)

The model starts like:

> “I won’t change anything yet… let me slowly learn what small change is useful”

Then during training:

* f(x) slowly becomes meaningful
* α·f(x) grows from tiny → useful adjustments

---

## Analogy (intuitive)

Think of a junior data scientist:

* Day 1: doesn’t change anything (just observes)
* Then slowly:

  * makes small suggestions
  * improves them over time

Residual scaling = **controlled learning, not aggressive guessing**

---

## Why this prevents loss spikes

Without scaling:

* Random weights → big outputs → unstable loss

With scaling:

* Small outputs → gradual learning → stable loss

---

## Now let’s test your understanding 🔍

Imagine 2 models:

### Model A (no scaling)

```
output = x + f(x)
```

### Model B (with scaling)

```
output = x + 0.01 · f(x)
```

## Doubt 2: The Chinchilla Paradigm vs The Inference-Optimal Paradigm
Chinchilla (by DeepMind) asked:

“Given a fixed training budget, how should we balance:

model size (N)
data tokens (D)?”

They found:

D ≈ 20 × N

Meaning:

Bigger model → needs more data
But don’t overtrain → it wastes compute
Intuition

If you:

Train too little → model underfits
Train too much → wasted training cost

So **Chinchilla = training-efficient strategy**

Better mental model
It sounds like:

“1 parameter ↔ 20 tokens”

But reality is:

👉 All parameters work together on all tokens

Not:
1 parameter learns from 20 tokens
another parameter learns from different 20 tokens
Think of it like a team:

Parameters = team members
Tokens = training experience

👉 Every team member learns from all experiences, not separate ones
Think of:
** Tokens = questions in an exam
Parameters = your brain’s understanding
Your brain doesn’t “store every question”
It learns:
patterns
rules
concepts
**
Same with models.
---
**Case 1: Small model + small data**
Not enough capacity + not enough experience
👉 **Underfitting**

**Case 2: Large model + small data**
Huge capacity + little data
👉 memorizes instead of learning
👉 **Overfitting**

**Case 3: Small model + HUGE data (this is inference-optimal)**
Limited capacity
But tons of exposure

👉 It is **forced to learn only the most important patterns**
👉 Cannot afford to memorize noise

**Smaller models are data-hungry but efficient learners
Larger models are powerful but expensive and prone to overfitting early**

**Chinchilla mindset:**
Balance capacity and data
Avoid wasting training compute
👉 Good for research

**Inference-optimal mindset:**
Overtrain smaller models
Minimize inference cost per query
👉 Good for real-world deployment (think Meta, Microsoft, Amazon)

**“We intentionally overtrain smaller models so they generalize well and are cheap to serve at scale.”**

**how a better generalized model cost less inference cost?**
What happens during inference?

**Every time a user sends a query:**

👉 **The model does matrix multiplications using ALL its parameters**

So:

More parameters = more multiplications
More multiplications = more GPU compute
More compute = more $$ cost per query

Step 3: Compare Model A vs Model B
Model A
70B parameters
Each query → heavy computation
👉 Expensive per request
Model B
8B parameters
Each query → much lighter computation
👉 Cheap per request

**How does better generalization reduce inference cost**
👉 Important:
It **does NOT directly reduce cost per query**

Instead, it allows:

**A smaller model to perform as well as a bigger one**

*Next: [Fine-Tuning Strategies](02-fine-tuning-strategies.md)*
