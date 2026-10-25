# Hallucination Detection in AI-Generated Text: A Step-by-Step Guide

This guide walks through the whole project, one step at a time. For each step it covers:
**the problem → the idea → the code → how to run it → what you should see → questions to prepare for your viva.**

---

## 0. Background: what is a hallucination?

A **hallucination** is text from an AI model that sounds fluent and confident but is **false or unsupported**.

| Type | Meaning | Example |
|---|---|---|
| **Factual (extrinsic)** | Contradicts world knowledge | "Einstein won the Nobel Prize in **1931**" (it was 1921) |
| **Faithfulness (intrinsic)** | Contradicts the source it was given | A summary says "three offices in **Asia**" when the article says "two in Europe" |
| **Fabrication** | Invents entities | Citing a paper that doesn't exist |

**Why do LLMs hallucinate?** They are trained to predict the *most plausible next word*, not the *true* one.
When the model doesn't know something, a plausible-sounding guess is still the most likely continuation.

### Our approach: a hybrid, explainable detector

No single detection method works everywhere, so we combine three independent signals:

```
                    ┌────────────────────────────────────────────────┐
  AI answer ──►  STEP 1: split into claims                           │
                    │                                                │
         for each claim:                                             │
                    ├──► STEP 2: retrieve evidence (Wikipedia/doc)   │
                    │        └──► STEP 3: NLI — supported or         │
                    │                       contradicted?  ──────┐   │
                    ├──► STEP 4: self-consistency — do re-sampled │   │
                    │            answers agree?  ────────────────┤   │
                    └──► STEP 5: token confidence — is the model │   │
                                 itself unsure?  ────────────────┤   │
                                                                  ▼   │
                         STEP 6: logistic regression → score 0..1    │
                                 + reason for every verdict          │
                    STEP 7: train & evaluate on HaluEval              │
                    STEP 8: web app with colour highlighting          │
                    └────────────────────────────────────────────────┘
```

| Signal | Needs external knowledge? | Needs model internals? | Catches |
|---|---|---|---|
| Evidence + NLI (Steps 2–3) | Yes | No | Facts that contradict a source |
| Self-consistency (Step 4) | No | No (black-box) | Facts the model is "guessing" |
| Token confidence (Step 5) | No | Yes (white-box) | Surprising words such as wrong names or dates |

---

## Setup

You need Python **3.10 or newer**.

```bash
cd hallucination-detector
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads three free models (~1.4 GB total) from Hugging Face:

| Model | Size | Used for |
|---|---|---|
| `all-MiniLM-L6-v2` | 90 MB | Semantic similarity (Step 2) |
| `nli-deberta-v3-xsmall` | 280 MB | Entailment/contradiction (Steps 3, 4) |
| `Qwen2.5-0.5B-Instruct` | 1 GB | Sampling (Step 4) and token probabilities (Step 5) |

### Project structure

```
hallucination-detector/
├── halludetect/
│   ├── config.py              all settings in one place
│   ├── models.py              loads each model once
│   ├── step1_claims.py        answer → claims
│   ├── step2_retrieval.py     claim → evidence passages
│   ├── step3_nli.py           (evidence, claim) → supported / contradicted
│   ├── step4_consistency.py   re-sample + compare
│   ├── step5_confidence.py    token log-probabilities
│   ├── step6_combine.py       features → score + explanation
│   └── detector.py            full pipeline
├── train_and_evaluate.py      STEP 7
├── app.py                     STEP 8
└── requirements.txt
```

Every step file has a demo at the bottom. **Run them in order**; each one shows exactly what that step adds.

---

## Step 1: Claim extraction  (`step1_claims.py`)

**Problem.** If you score a whole paragraph at once, one false sentence gets averaged away by the true ones.

**Idea.** Split the answer into small claims and check each one on its own. This is the idea behind *FActScore*.

**How it works:**
1. `split_sentences()`: splits on `. ! ?` without breaking on "Dr.", "U.S.", "3.14" or initials like "J. K."
2. `is_checkable()`: drops questions, opinions and filler ("Hope this helps!")
3. `decontextualize()`: "He developed relativity" → "**Albert Einstein** developed relativity". Without this, the evidence search wouldn't know who "he" is.
4. `make_statement()`: a one-word answer like "Delhi" means nothing to an NLI model, so it becomes *"The answer to the question '…' is Delhi."*

```bash
python -m halludetect.step1_claims
```
```
Claim 2
  as written : He developed the theory of relativity.
  to verify  : Albert Einstein developed the theory of relativity.
Claim 3
  as written : His Nobel Prize in Physics was awarded in 1931 ...
  to verify  : Albert Einstein's Nobel Prize in Physics was awarded in 1931 ...
```
"Hope this helps!" was dropped automatically.

**Viva questions**
- *Why sentence-level instead of paragraph-level?* Granularity: one wrong fact shouldn't hide among correct ones, and the user sees exactly which sentence is wrong.
- *Limitation?* Rule-based splitting can't break "X was born in A and died in B" into two facts. FActScore uses an LLM for this. That's a possible future improvement.

---

## Step 2: Evidence retrieval  (`step2_retrieval.py`)

**Problem.** To judge a claim we need trustworthy text. The LLM itself is not trustworthy.

**Idea.** Search Wikipedia (or use a source document you provide), cut the text into passages of 3 sentences, and keep the 3 passages **most similar in meaning** to the claim.

**Key concept: embeddings and cosine similarity.** MiniLM maps any text to a 384-dimensional vector. Texts with similar meaning point in similar directions:

```
cos(θ) = (a · b) / (|a| |b|)       1 = same meaning, 0 = unrelated
```

Passages overlap by one sentence, so a fact that falls across a boundary still appears whole in some passage.

```bash
python -m halludetect.step2_retrieval
```
```
Evidence 1  (relevance 0.79, Wikipedia: List of awards and honors received by Albert Einstein)
   In 1922 Albert Einstein was awarded the 1921 Nobel Prize in Physics, "for his services ...
```

**Viva questions**
- *Why not keyword search?* "born" vs "birth" or "died" vs "passed away" share no keywords but mean the same thing.
- *Why is context optional?* To check a summary against its article, or a RAG chatbot against its documents (faithfulness hallucination), you pass the source as context instead of using Wikipedia.

---

## Step 3: Verification with NLI  (`step3_nli.py`)

**Problem.** Similarity isn't truth: "won in 1921" and "won in 1931" are nearly identical as embeddings.

**Idea.** A **Natural Language Inference** model reads `(premise = evidence, hypothesis = claim)` *together* and outputs
P(entailment), P(neutral) and P(contradiction). It was trained on ~1M labelled pairs (SNLI and MultiNLI).

```bash
python -m halludetect.step3_nli
```
```
Supported     entail=1.00 contra=0.00  | Einstein's Nobel Prize in Physics was awarded in 1921.
Contradicted  entail=0.00 contra=1.00  | Einstein's Nobel Prize in Physics was awarded in 1931.
Unverifiable  entail=0.00 contra=0.00  | Einstein played the violin.
```

**An interesting finding to put in your report.** Without a fix, the model labelled *"Einstein played the violin"* as **Contradicted (1.00)**. The passage was about his Nobel Prize and says nothing about violins. NLI models often call *unrelated* sentences contradictions, because that's how the SNLI dataset was annotated. Our fix: a passage may only contradict a claim when its relevance from Step 2 is at least 0.45 (`MIN_RELEVANCE_TO_CONTRADICT`). The unrelated pair scored 0.35 and the real contradiction scored 0.71.

**Viva questions**
- *Bi-encoder vs cross-encoder?* A bi-encoder (Step 2) embeds each text separately. It's fast, so it suits searching. A cross-encoder (Step 3) reads both texts together. It's slower but compares details word by word, so it suits verifying.
- *Why take the max over passages?* One clear supporting passage is enough. Others may simply be about something else.

---

## Step 4: Self-consistency  (`step4_consistency.py`)

**Problem.** What if no evidence exists or retrieval fails?

**Idea (SelfCheckGPT).** If a model *knows* a fact, it repeats it every time. If it's *guessing*, each attempt differs. So we sample the question 3 more times at temperature 1.0 and use NLI to check whether each sample supports the claim.

```bash
python -m halludetect.step4_consistency
```
```
  sample 1: The capital of Australia is Canberra. It serves as ...
  sample 2: The capital of Australia is Canberra.
  sample 3: The capital of Australia is Canberra.
support=0.99 conflict=0.00 | The capital of Australia is Canberra.
support=0.00 conflict=1.00 | The capital of Australia is Sydney.
```

**What is temperature?** The model's probabilities are divided by T before sampling. T→0 always picks the top word. T=1 samples from the true distribution, which exposes what the model is unsure about.

**Limitation (state this honestly in your report).** Ideally the samples come from the *same* model that wrote the answer. We use a small free local model, so its samples are noisier than GPT-4's would be. To improve this, replace `generate_samples()` with API calls to the original model.

---

## Step 5: Token confidence  (`step5_confidence.py`)

**Problem.** Steps 3–4 look at text from the outside. Can we look inside the model?

**Idea.** An LLM gives every possible next token a probability. We feed in the claim and read how likely the model found each of its tokens:

- `mean_logprob`: average log-probability (closer to 0 means more confident)
- `min_logprob`: the single most surprising token. Hallucinations often hide in one wrong word.
- `confidence = exp(mean_logprob)`, which is the inverse of perplexity

```bash
python -m halludetect.step5_confidence
```
```
confidence=0.967 mean_lp=-0.03 min_lp=-0.14 weakest='The'  | The capital of France is Paris.
confidence=0.395 mean_lp=-0.93 min_lp=-6.08 weakest='Lyon' | The capital of France is Lyon.
```
It pinpointed **"Lyon"** as the suspicious word, and the app shows this to the user.

**Viva questions**
- *Why log-probabilities?* Probabilities of long sequences become tiny numbers (underflow). Logs turn products into sums.
- *Can low probability mean "true but rare"?* Yes, and that's why this signal is combined with the others rather than used alone.

---

## Step 6: Combining signals  (`step6_combine.py`)

**Idea.** **Logistic regression** over the six features:

```
z = w₁·entail + w₂·contradict + w₃·support + w₄·conflict + w₅·mean_lp + w₆·min_lp + b
score = σ(z) = 1 / (1 + e^(−z))         → probability of hallucination
```

**Why logistic regression and not a neural network?**
1. **Explainability.** Each feature's contribution `wᵢ·xᵢ` is visible, so the app can show *why* a claim was flagged.
2. Only six features and a few hundred examples, so a bigger model would overfit.
3. The learned weights are themselves a result: they show which signal matters most.

Features are standardized first (mean 0, std 1) so the weights can be compared with each other.

Verdict bands: `< 0.35` likely correct · `0.35–0.65` uncertain · `> 0.65` likely hallucinated.

**Evidence rule.** After the learned score, a *decisive* evidence verdict (NLI ≥ 0.95) caps the score at 0.30 (supported) or raises it to 0.70 (contradicted). Step 7 explains why and what it costs. Switch it off with `EVIDENCE_OVERRIDE = False` in `config.py`.

```bash
python -m halludetect.step6_combine
python -m halludetect.detector        # the full pipeline (Steps 1–6)
```
```
OVERALL: Likely hallucinated (score 0.91)
[Likely correct      0.19] Albert Einstein was a German-born theoretical physicist.
[Likely correct      0.30] He developed the theory of relativity.
[Likely hallucinated 0.91] His Nobel Prize in Physics was awarded in 1931 for the photoelectric effect.
     evidence: Contradicted (entail 0.00, contra 1.00)
```
The overall answer score is the **worst** claim's score: one false fact makes the answer unreliable.

---

## Step 7: Training and evaluation  (`train_and_evaluate.py`)

**Dataset: HaluEval QA** (10,000 items). Each item has a knowledge snippet, a question, a **correct** answer and a **hallucinated** answer that ChatGPT wrote on purpose to be plausible.

```bash
python train_and_evaluate.py --n-items 200                 # ~5–10 min on a laptop
python train_and_evaluate.py --n-items 200 --consistency   # adds Step 4 (slower)
```

**Important methodology points (examiners like these):**
1. **Split by question, not by answer.** Both answers to a question go into the same split. Otherwise the model has already "seen" the question during training: that's data leakage.
2. **Every signal is evaluated on its own first**, then combined. This is an ablation study: it shows what each part contributes.
3. **Dataset bias.** In HaluEval, correct answers are short ("Delhi") and hallucinated ones are full sentences. The script reports answer length as a *diagnostic only*. It scores high, which proves the dataset has this shortcut, and we deliberately never use it as a feature.

**Metrics:**
- **Precision**: of the answers we flagged, how many really were hallucinations
- **Recall**: of the real hallucinations, how many we caught
- **F1**: the balance of the two
- **AUROC**: the chance a random hallucinated answer scores higher than a random correct one. 0.5 is a coin flip and 1.0 is perfect. It doesn't depend on a threshold, so it's the fairest comparison.

### Results

Final run: **400 HaluEval items (800 answers), 560 train / 240 test**, without self-consistency (`python train_and_evaluate.py --n-items 400`).

| Signal (alone) | Test AUROC |
|---|---|
| Evidence supports it | 0.650 |
| Evidence contradicts it | 0.557 |
| Model's least-confident word | 0.618 |
| Model's average confidence | 0.738 |
| **Combined (logistic regression)** | **0.779** |

| Combined detector | Learned only | + evidence rule |
|---|---|---|
| Accuracy | 0.696 | 0.679 |
| Precision | 0.724 | 0.713 |
| Recall | 0.633 | 0.600 |
| F1 | 0.676 | 0.652 |
| AUROC | 0.779 | 0.759 |

**The story behind these numbers (good material for the report):**
1. *First attempt:* evidence AUROC was about 0.55 (near chance), and confidence looked strong for the wrong reason. Short answers like "Afrojack" were turned into an awkward template sentence that the NLI model couldn't handle, and a lone name always looks "surprising" to the LLM. So confidence was secretly measuring **length**.
2. *Fix:* short answers are rewritten into natural sentences by the local LLM (Step 1, `question_to_statement`). Every signal now points the right way, and the length shortcut dropped from 0.96 to 0.70.
3. **The combination beats every single signal**, which shows the signals complement each other.
4. **Small test sets mislead.** An earlier 150-item run reported AUROC 0.895. With 400 items it settled at 0.779. Always report the larger evaluation.
5. **Why is evidence weak on HaluEval?** Looking at the errors showed two causes:
   - HaluEval's "hallucinated" answers are often **true sentences that don't answer the question** (Q: *Which band…?* A: *"Heart is American."*). NLI checks the sentence against the evidence, not against the question, so it correctly says "supported". Even confident NLI verdicts (≥ 0.95) agree with HaluEval labels only about 70% of the time.
   - **The helper LLM in Step 1 sometimes hallucinates while rewriting**: "Queen Mab originated in Shakespeare's *The Tempest*" (it's *Romeo and Juliet*). The evidence then rightly contradicts a correct answer. The detector's own components can hallucinate too.
6. **The evidence rule.** Because of point 5, the learned weights under-trust evidence. In the app this made a Wikipedia-supported claim ("Einstein developed relativity", entailment 0.99) score 0.49, "uncertain". For open-domain fact checking a decisive NLI verdict is the strongest signal we have, so `config.EVIDENCE_OVERRIDE` caps the score at 0.30 when entailment ≥ 0.95 and raises it to 0.70 when contradiction ≥ 0.95. This costs 0.02 AUROC on HaluEval (table above) but gives correct verdicts on the demo cases. It's a deliberate, documented trade-off between benchmark score and the intended use.

---

## Step 8: Web app  (`app.py`)

```bash
streamlit run app.py
```

- Paste a question and an answer, or click an example, or let the local model generate an answer
- Each sentence is highlighted 🟩 🟨 🟥
- Expand a claim to see the evidence passage and its source, the agreement across samples, the word the model was least sure about, and a bar chart of **what pushed the score**
- Optional source document, for checking summaries or RAG output against their source

---

## Limitations and future work (for your report's final chapter)

| Limitation | Possible improvement |
|---|---|
| Rule-based claim splitting | Use an LLM to extract atomic facts (FActScore) |
| The Step 1 rewrite LLM sometimes adds invented details | Use a bigger model, or reject rewrites that add words not in the question or answer |
| NLI ignores whether the claim answers the question | Add a "does this answer the question?" check (QA relevance model) |
| Small NLI model (xsmall) | Use `nli-deberta-v3-base` or `large` with a GPU |
| Self-consistency uses a different model than the one that wrote the answer | Sample from the original model via API |
| Wikipedia only | Add web search, PubMed (medical), or a domain database |
| HaluEval QA only | Also evaluate on TruthfulQA, FEVER, RAGTruth |
| Entailment ≠ truth: Wikipedia can be wrong or outdated | Use multiple sources and show agreement |
| Token confidence only works with open models | Use API logprobs where available |

## Ideas to extend the project

1. **Ablation table**: run with each signal switched off and compare AUROC.
2. **Bigger models**: compare NLI xsmall, small and base, and plot accuracy against speed.
3. **Semantic entropy** (Farquhar et al., *Nature* 2024): cluster samples by meaning and compute entropy over the clusters.
4. **Fake citation checker**: verify DOIs and paper titles against the CrossRef API.
5. **Hidden-state probe**: train a classifier on the LLM's internal activations.

## References

1. Manakul, Liusie & Gales (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs.* EMNLP.
2. Li et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs.* EMNLP.
3. Min et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP.
4. Farquhar et al. (2024). *Detecting hallucinations in large language models using semantic entropy.* Nature.
5. Huang et al. (2023). *A Survey on Hallucination in Large Language Models.* arXiv:2311.05232.
6. Williams, Nangia & Bowman (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MultiNLI).* NAACL.
7. Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
