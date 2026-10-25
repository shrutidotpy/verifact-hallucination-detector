"""
Loads the three neural models once and reuses them.

`lru_cache(maxsize=1)` means: the first call downloads/loads the model, every
later call returns the same object instantly.  Models are loaded lazily, so if
you only run Step 1 nothing heavy is loaded at all.
"""
from functools import lru_cache

import torch

from . import config


def get_device() -> str:
    """Pick the fastest hardware available: NVIDIA GPU > Apple GPU > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL, device=get_device())


@lru_cache(maxsize=1)
def load_nli():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL)
    model.to(get_device()).eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL)
    model.to(get_device()).eval()
    return tokenizer, model
