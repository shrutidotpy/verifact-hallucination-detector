"""
STEP 3 — Verifying claims with Natural Language Inference (NLI)
===============================================================

Problem:  We have a claim and some relevant evidence.  Similarity alone is
not enough: "Einstein won the Nobel in 1921" and "...in 1931" are almost
identical in embedding space, but one is true and one is false.

Idea:     An NLI model is trained (on millions of labelled sentence pairs,
datasets SNLI & MultiNLI) to answer one question:

      Given the PREMISE (evidence), is the HYPOTHESIS (claim) ...
          entailed      -> evidence says the claim is true
          contradicted  -> evidence says the claim is false
          neutral       -> evidence doesn't say either way

It's a "cross-encoder": premise and hypothesis are read TOGETHER, so the model
can compare details word by word (1921 vs 1931).

Decision rule for a claim checked against several passages:
  * entailment    = the strongest support from any passage
  * contradiction = the strongest contradiction from any RELEVANT passage
                    (NLI models wrongly call unrelated sentences "contradiction",
                    so passages with low relevance from Step 2 can't contradict)
  * Supported / Contradicted / Unverifiable based on thresholds in config.py

Run the demo:   python -m halludetect.step3_nli
"""
from dataclasses import dataclass

import torch

from . import config
from .models import get_device, load_nli
from .step2_retrieval import Passage, rank_passages


@dataclass
class EvidenceResult:
    entailment: float          # 0..1, max over passages
    contradiction: float       # 0..1, max over passages
    label: str                 # "Supported" / "Contradicted" / "Unverifiable"
    best_passage: Passage | None


def nli_probs(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """For each (premise, hypothesis) pair return
    {'entailment': p, 'neutral': p, 'contradiction': p} summing to 1."""
    if not pairs:
        return []
    tokenizer, model = load_nli()
    premises, hypotheses = zip(*pairs)
    inputs = tokenizer(list(premises), list(hypotheses), padding=True,
                       truncation=True, max_length=512, return_tensors="pt").to(get_device())
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).cpu()

    # The model config tells us which output index means which label.
    id2label = {i: name.lower() for i, name in model.config.id2label.items()}
    return [{id2label[i]: float(row[i]) for i in range(row.shape[0])} for row in probs]


def verify_claim(claim: str, passages: list[Passage]) -> EvidenceResult:
    if not passages:
        return EvidenceResult(0.0, 0.0, "Unverifiable", None)

    scores = nli_probs([(p.text, claim) for p in passages])
    for s, p in zip(scores, passages):
        if p.relevance < config.MIN_RELEVANCE_TO_CONTRADICT:
            s["contradiction"] = 0.0
    best_entail = max(range(len(scores)), key=lambda i: scores[i]["entailment"])
    best_contra = max(range(len(scores)), key=lambda i: scores[i]["contradiction"])
    entail = scores[best_entail]["entailment"]
    contra = scores[best_contra]["contradiction"]

    if contra >= config.CONTRADICT_THRESHOLD and contra > entail:
        return EvidenceResult(entail, contra, "Contradicted", passages[best_contra])
    if entail >= config.SUPPORT_THRESHOLD:
        return EvidenceResult(entail, contra, "Supported", passages[best_entail])
    return EvidenceResult(entail, contra, "Unverifiable", passages[best_entail])


if __name__ == "__main__":
    evidence = Passage(
        "Einstein received the 1921 Nobel Prize in Physics for his services to "
        "theoretical physics, and especially for his discovery of the law of the "
        "photoelectric effect.", "example")
    for claim in [
        "Einstein's Nobel Prize in Physics was awarded in 1921.",   # true
        "Einstein's Nobel Prize in Physics was awarded in 1931.",   # false
        "Einstein played the violin.",                              # not in evidence
    ]:
        # In the real pipeline Step 2 fills in relevance; here we compute it ourselves.
        evidence = rank_passages(claim, [evidence], top_k=1)[0]
        r = verify_claim(claim, [evidence])
        print(f"{r.label:13s} entail={r.entailment:.2f} contra={r.contradiction:.2f}  | {claim}")
