"""
STEP 1 — Claim extraction
=========================

Problem:  An AI answer is usually a paragraph.  Some sentences are true, some
may be invented.  If we score the whole paragraph at once, one false sentence
gets "averaged away" by the true ones.

Idea:     Break the answer into small, independently checkable CLAIMS and
verify each one separately.  (This is the core idea of FActScore, Min et al.
2023, which uses an LLM to split text into "atomic facts".  We use simple,
transparent rules instead so every decision is explainable.)

What this file does:
  1. split_sentences()   – split text into sentences (careful with "Dr." etc.)
  2. is_checkable()      – drop sentences that state no fact ("Hope this helps!")
  3. decontextualize()   – make each sentence understandable on its own
                           ("He was born in 1879." -> "Albert Einstein was born in 1879.")
  4. make_statement()    – turn a very short answer ("Delhi") into a full
                           sentence the NLI model can understand

Run the demo:   python -m halludetect.step1_claims
"""
import re
from dataclasses import dataclass


@dataclass
class Claim:
    text: str        # the sentence exactly as it appears in the answer (for highlighting)
    statement: str   # the self-contained version we actually verify


# --- 1. Sentence splitting --------------------------------------------------

# A full stop after these is NOT the end of a sentence.
_ABBREVIATIONS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Jr.", "Sr.",
                  "vs.", "e.g.", "i.e.", "etc.", "approx.", "U.S.", "U.K."]
_DOT = "<DOT>"


def split_sentences(text: str) -> list[str]:
    protected = text.strip()
    # Hide dots that don't end a sentence, split, then put them back.
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _DOT))
    protected = re.sub(r"(\d)\.(\d)", rf"\1{_DOT}\2", protected)       # 3.14
    # Some datasets glue sentences together: "century.First ..." -> add a space.
    protected = re.sub(r"([a-z0-9)])\.([A-Z])", r"\1. \2", protected)
    # Split after . ! ? followed by whitespace, but not after an initial like "J."
    parts = re.split(r"(?<!\b[A-Z]\.)(?<=[.!?])\s+", protected)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


# --- 2. Is there a fact to check? -------------------------------------------

_NON_FACTUAL_STARTS = (
    "i think", "i believe", "in my opinion", "as an ai", "i'm not sure",
    "i am not sure", "hope this helps", "let me know", "feel free",
    "great question", "sure,", "sure!", "certainly", "of course",
)


def is_checkable(sentence: str) -> bool:
    s = sentence.lower().strip()
    if s.endswith("?"):                      # questions assert nothing
        return False
    if s.startswith(_NON_FACTUAL_STARTS):    # opinions / filler
        return False
    return len(s.split()) >= 3               # "Yes." / "Thanks!" carry no fact alone


# --- 3. Decontextualization -------------------------------------------------

_SUBJECT_PRONOUNS = {"he", "she", "it", "they"}
_POSSESSIVE_PRONOUNS = {"his", "her", "its", "their"}
_NOT_A_SUBJECT = {"the", "a", "an", "this", "that", "these", "those", "in", "on", "at",
                  "after", "before", "during", "when", "there", "however", "yes", "no"}


def find_main_subject(sentence: str) -> str | None:
    """Heuristic: the first run of Capitalized Words is usually the topic.
    'Albert Einstein was a physicist.' -> 'Albert Einstein'"""
    match = re.match(r"^((?:The\s)?(?:[A-Z][\w'’\-]*\s?)+)", sentence)
    if not match:
        return None
    subject = match.group(1).strip()
    # "The company reported ..." matches just "The" — that's not a name.
    if subject.lower() in _SUBJECT_PRONOUNS | _POSSESSIVE_PRONOUNS | _NOT_A_SUBJECT:
        return None
    return subject


def decontextualize(sentence: str, subject: str | None) -> str:
    """Replace a leading pronoun with the main subject of the answer."""
    if not subject:
        return sentence
    first, _, rest = sentence.partition(" ")
    if first.lower() in _SUBJECT_PRONOUNS:
        return f"{subject} {rest}"
    if first.lower() in _POSSESSIVE_PRONOUNS:
        return f"{subject}'s {rest}"
    return sentence


# --- 4. Short answers -> full statements ------------------------------------

_QA2D_PROMPT = """Combine the question and the answer into one complete, standalone factual sentence.

Question: Which magazine was started first, Arthur's Magazine or First for Women?
Answer: Arthur's Magazine
Sentence: Arthur's Magazine was started before First for Women.

Question: Are Yukio Mishima and Roberto Bolaño both Chilean?
Answer: no
Sentence: Yukio Mishima and Roberto Bolaño are not both Chilean.

Question: {question}
Answer: {answer}
Sentence:"""


def question_to_statement(question: str, answer: str) -> str:
    """Ask the local LLM to merge question + short answer into one sentence
    (called "QA-to-declarative" conversion).  The two worked examples in the
    prompt show the model what we want — this is "few-shot prompting"."""
    import torch

    from .models import get_device, load_llm

    tokenizer, model = load_llm()
    messages = [{"role": "user", "content": _QA2D_PROMPT.format(question=question.strip(), answer=answer.strip())}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(get_device())
    with torch.no_grad():
        # Greedy decoding; the None values switch off the model's default sampling
        # settings, which would otherwise print harmless but noisy warnings.
        output = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                                temperature=None, top_p=None, top_k=None,
                                pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n")[0]


def make_statement(text: str, question: str | None, use_llm: bool = True) -> str:
    """'Delhi' alone means nothing to an NLI model, so short answers are merged
    with their question into a natural sentence:
        Q: Where is the head office of The Oberoi Group?   A: Delhi
        -> "The head office of The Oberoi Group is in Delhi."
    Falls back to a simple template if the LLM is unavailable."""
    if question and len(text.split()) <= 5:
        if use_llm:
            try:
                statement = question_to_statement(question, text)
                if statement:
                    return statement
            except Exception as error:   # model not downloaded, out of memory, ...
                print(f"[claims] LLM rewrite failed, using template: {error}")
        return f'The answer to the question "{question.strip()}" is {text.strip().rstrip(".")}.'
    return text


# --- Putting it together ----------------------------------------------------

def extract_claims(answer: str, question: str | None = None, use_llm: bool = True) -> list[Claim]:
    answer = answer.strip()
    if not answer:
        return []

    # A very short answer is one claim on its own.
    if len(answer.split()) <= 5:
        return [Claim(text=answer, statement=make_statement(answer, question, use_llm))]

    sentences = split_sentences(answer)
    subject = find_main_subject(sentences[0]) if sentences else None

    claims = []
    for sentence in sentences:
        if not is_checkable(sentence):
            continue
        statement = decontextualize(sentence, subject)
        claims.append(Claim(text=sentence, statement=statement))
    return claims


if __name__ == "__main__":
    demo_answer = (
        "Albert Einstein was a German-born theoretical physicist. "
        "He developed the theory of relativity. "
        "His Nobel Prize in Physics was awarded in 1931 for the photoelectric effect. "
        "Hope this helps!"
    )
    print("ANSWER:\n ", demo_answer, "\n")
    for i, claim in enumerate(extract_claims(demo_answer), 1):
        print(f"Claim {i}")
        print("  as written :", claim.text)
        print("  to verify  :", claim.statement)

    print("\nShort answer example (template, no model needed):")
    question = "Where is the head office of The Oberoi Group?"
    for claim in extract_claims("Delhi", question, use_llm=False):
        print("  to verify  :", claim.statement)
    print("Short answer example (LLM rewrite, downloads ~1 GB model on first run):")
    for claim in extract_claims("Delhi", question):
        print("  to verify  :", claim.statement)
