"""
STEP 4 — Self-consistency (the SelfCheckGPT idea)
=================================================

Problem:  Retrieval only works if the evidence exists and we found it.
What if Wikipedia has nothing on the topic?

Idea:     (Manakul et al., 2023, "SelfCheckGPT")
  * When a model really KNOWS a fact, it states it the same way every time.
  * When it is making things up, each new attempt invents something different.
So: ask the model the same question several more times with randomness
switched on (temperature > 0), then check whether those extra samples agree
with each claim.  Agreement is checked with the same NLI model from Step 3:

      premise    = one sampled answer
      hypothesis = the claim
      support    = average entailment over all samples   (high = consistent)
      conflict   = average contradiction over samples    (high = suspicious)

No external knowledge needed — this is a "zero-resource, black-box" method.

Limitation to mention in your report:  ideally the samples come from the SAME
model that wrote the answer.  We use a small local model (Qwen2.5-0.5B) so the
project is free; it knows less than ChatGPT, so its samples are noisier.  If you
have API access, swap generate_samples() for calls to that model.

Run the demo:   python -m halludetect.step4_consistency
"""
from dataclasses import dataclass, field

import torch

from . import config
from .models import get_device, load_llm
from .step3_nli import nli_probs

SYSTEM_PROMPT = "You are a helpful assistant. Answer factually in 1-3 sentences."


@dataclass
class ConsistencyResult:
    support: float                    # mean entailment over samples
    conflict: float                   # mean contradiction over samples
    samples: list[str] = field(default_factory=list)


def generate_samples(question: str, n: int = config.NUM_SAMPLES,
                     temperature: float = config.SAMPLE_TEMPERATURE,
                     max_new_tokens: int = config.SAMPLE_MAX_TOKENS) -> list[str]:
    tokenizer, model = load_llm()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(get_device())
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=0.95,
                                max_new_tokens=max_new_tokens, num_return_sequences=n,
                                pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[:, inputs["input_ids"].shape[1]:]   # drop the prompt part
    return [text.strip() for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def consistency_score(claim: str, samples: list[str]) -> ConsistencyResult:
    if not samples:
        return ConsistencyResult(0.0, 0.0, [])
    scores = nli_probs([(sample, claim) for sample in samples])
    support = sum(s["entailment"] for s in scores) / len(scores)
    conflict = sum(s["contradiction"] for s in scores) / len(scores)
    return ConsistencyResult(support, conflict, samples)


if __name__ == "__main__":
    question = "What is the capital of Australia?"
    samples = generate_samples(question)
    print("QUESTION:", question)
    for i, s in enumerate(samples, 1):
        print(f"  sample {i}: {s}")
    print()
    for claim in ["The capital of Australia is Canberra.", "The capital of Australia is Sydney."]:
        r = consistency_score(claim, samples)
        print(f"support={r.support:.2f} conflict={r.conflict:.2f} | {claim}")
