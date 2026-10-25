"""
Central settings for the whole project.

Keeping every model name and threshold in one file means you can run
experiments ("what if I use a bigger NLI model?") by changing one line.
"""

# ---------------------------------------------------------------------------
# Models (all free, all run locally, all small enough for a laptop)
# ---------------------------------------------------------------------------

# Turns sentences into vectors so we can measure "how similar" two texts are.
# Used in Step 2 to pick the most relevant evidence passages.  (~90 MB)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Natural Language Inference model: given (premise, hypothesis) it predicts
# entailment / neutral / contradiction.  The heart of Steps 3 and 4.  (~280 MB)
# Bigger, more accurate options: "cross-encoder/nli-deberta-v3-small",
# "cross-encoder/nli-deberta-v3-base".
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"

# A small chat LLM.  Used in Step 4 (generate extra samples) and Step 5
# (read token probabilities).  (~1 GB)
# Bigger option if you have a GPU: "Qwen/Qwen2.5-1.5B-Instruct".
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# Step 2: retrieval
# ---------------------------------------------------------------------------
WIKI_PAGES_PER_QUERY = 2       # how many Wikipedia pages to read per search
SENTENCES_PER_PASSAGE = 3      # evidence is split into chunks of N sentences
TOP_K_PASSAGES = 3             # how many best chunks we send to the NLI model

# ---------------------------------------------------------------------------
# Step 3: NLI verdict thresholds
# ---------------------------------------------------------------------------
SUPPORT_THRESHOLD = 0.5        # entailment prob above this  -> "Supported"
CONTRADICT_THRESHOLD = 0.5     # contradiction prob above this -> "Contradicted"
# NLI models tend to call an UNRELATED sentence a "contradiction"
# ("Einstein played the violin" vs a passage about his Nobel Prize -> contradiction!).
# So a passage may only contradict a claim if it is actually about the same thing.
MIN_RELEVANCE_TO_CONTRADICT = 0.45

# ---------------------------------------------------------------------------
# Step 4: self-consistency
# ---------------------------------------------------------------------------
NUM_SAMPLES = 3                # extra answers to sample from the LLM
SAMPLE_TEMPERATURE = 1.0       # high temperature = more diverse samples
SAMPLE_MAX_TOKENS = 80

# ---------------------------------------------------------------------------
# Step 6: final verdict bands on the 0..1 hallucination score
# ---------------------------------------------------------------------------
LOW_RISK = 0.35                # below -> "Likely correct"
HIGH_RISK = 0.65               # above -> "Likely hallucinated"

COMBINER_PATH = "models/combiner.pkl"

# Evidence rule: when the NLI verdict is decisive, it overrides the learned score.
#   entailment    >= 0.95  -> score capped at 0.30 (always "Likely correct")
#   contradiction >= 0.95  -> score raised to 0.70 (always "Likely hallucinated")
# Why: the weights are learned on HaluEval, whose "hallucinations" are often TRUE
# sentences that just don't answer the question — so there evidence looks weak.
# For checking facts against Wikipedia, a decisive NLI verdict is the strongest
# signal we have.  Costs ~0.02 AUROC on HaluEval (train_and_evaluate.py reports both).
EVIDENCE_OVERRIDE = True
EVIDENCE_DECISIVE = 0.95
SUPPORTED_CAP = 0.30
CONTRADICTED_FLOOR = 0.70
