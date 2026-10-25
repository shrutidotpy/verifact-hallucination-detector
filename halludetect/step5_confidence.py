"""
STEP 5 — Model confidence from token probabilities (white-box signal)
=====================================================================

Problem:  Steps 3 and 4 look at the text from the outside.  Can we look
INSIDE the language model?

Idea:     An LLM writes text one token at a time.  For every position it
assigns a probability to every possible next token.  If we feed the claim in,
we can read off how probable the model found each of the claim's tokens:

      "Einstein won the Nobel Prize in 1921"
       p(1921 | ...) = 0.62   -> model is fairly sure
       p(1931 | ...) = 0.03   -> model finds this surprising

We work with log-probabilities (log p, always <= 0; closer to 0 = more sure):
  * mean_logprob – average surprise across the whole claim
  * min_logprob  – the single most surprising token.  Hallucinations often
                   hide in ONE wrong token (a date, a name), so this is
                   frequently the more useful feature.
  * confidence   – exp(mean_logprob), a 0..1 number that is easier to read
                   (the inverse of "perplexity").

Low probability does not prove a claim false (rare true facts are also
"surprising"), which is exactly why we COMBINE this with Steps 3 & 4.

Run the demo:   python -m halludetect.step5_confidence
"""
import math
from dataclasses import dataclass

import torch

from .models import get_device, load_llm

DEFAULT_QUESTION = "Write a short factual answer."


@dataclass
class ConfidenceResult:
    mean_logprob: float
    min_logprob: float
    confidence: float
    weakest_token: str      # the token the model found least likely (great for explanations)


def claim_confidence(claim: str, question: str | None = None,
                     preceding_text: str = "") -> ConfidenceResult:
    """Probability of `claim` given the question and the answer text before it."""
    tokenizer, model = load_llm()

    messages = [{"role": "user", "content": question or DEFAULT_QUESTION}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if preceding_text:
        prefix += preceding_text.strip()
        claim = " " + claim          # keep the space so tokens match natural text

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    claim_ids = tokenizer(claim, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prefix_ids + claim_ids], device=get_device())

    with torch.no_grad():
        logits = model(input_ids).logits[0]

    # logits at position i predict token i+1, so shift by one.
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    token_logprobs = log_probs[torch.arange(len(targets)), targets][-len(claim_ids):].cpu()

    weakest = int(token_logprobs.argmin())
    mean_lp = float(token_logprobs.mean())
    return ConfidenceResult(
        mean_logprob=mean_lp,
        min_logprob=float(token_logprobs[weakest]),
        confidence=math.exp(mean_lp),
        weakest_token=tokenizer.decode([claim_ids[weakest]]).strip(),
    )


if __name__ == "__main__":
    question = "What is the capital of France?"
    for claim in ["The capital of France is Paris.", "The capital of France is Lyon."]:
        r = claim_confidence(claim, question)
        print(f"confidence={r.confidence:.3f} mean_lp={r.mean_logprob:.2f} "
              f"min_lp={r.min_logprob:.2f} weakest='{r.weakest_token}' | {claim}")
