"""
STEP 2 — Evidence retrieval
===========================

Problem:  To decide whether a claim is true we need something trustworthy to
compare it against.  The LLM itself is NOT trustworthy — that's the point.

Idea:     Retrieval-Augmented Verification.  Fetch real text (Wikipedia, or a
source document the user provides), cut it into small passages, and keep only
the passages most relevant to the claim.

How relevance is measured (semantic search):
  * An embedding model turns every text into a vector of 384 numbers.
  * Texts with similar MEANING get vectors pointing in a similar direction,
    even if they use different words ("born" vs "birth").
  * Cosine similarity between claim vector and passage vector = relevance.

Two evidence sources:
  * context given  -> use it (e.g. checking a summary against its article,
                      or a RAG chatbot against its retrieved documents)
  * no context     -> search Wikipedia live through its public API

Run the demo:   python -m halludetect.step2_retrieval
"""
import time
from dataclasses import dataclass
from functools import lru_cache

import requests

from . import config
from .models import load_embedder
from .step1_claims import split_sentences

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks every client to identify itself with a User-Agent.
HEADERS = {"User-Agent": "HallucinationDetectorStudentProject/1.0 (educational use)"}


@dataclass
class Passage:
    text: str
    source: str            # e.g. "Wikipedia: Albert Einstein" or "Provided context"
    relevance: float = 0.0  # cosine similarity to the claim, filled in by rank_passages()


# --- Wikipedia access -------------------------------------------------------

def _wiki_get(params: dict, retries: int = 3) -> dict:
    """Call the Wikipedia API.  If it says "429 Too Many Requests", wait and
    retry (1s, 2s, 4s) — the polite way to handle rate limits."""
    for attempt in range(retries + 1):
        response = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        if response.status_code == 429 and attempt < retries:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()


# lru_cache remembers results so the same page is never downloaded twice.
# Errors are raised (not returned) inside the cached functions so that a
# temporary failure is NOT remembered forever.
@lru_cache(maxsize=256)
def _search(query: str, n_pages: int) -> tuple[str, ...]:
    data = _wiki_get({"action": "query", "list": "search", "srsearch": query,
                      "srlimit": n_pages, "format": "json"})
    return tuple(hit["title"] for hit in data["query"]["search"])


@lru_cache(maxsize=256)
def _page_text(title: str) -> str:
    data = _wiki_get({"action": "query", "prop": "extracts", "explaintext": 1,
                      "titles": title, "format": "json", "redirects": 1})
    return next(iter(data["query"]["pages"].values())).get("extract", "")


def search_wikipedia(query: str, n_pages: int = config.WIKI_PAGES_PER_QUERY) -> tuple[str, ...]:
    """Return the titles of the best-matching Wikipedia pages."""
    try:
        return _search(query, n_pages)
    except (requests.RequestException, KeyError) as error:
        print(f"[retrieval] Wikipedia search failed: {error}")
        return ()


def fetch_page_text(title: str, max_chars: int = 20000) -> str:
    """Download the plain text of one Wikipedia page."""
    try:
        return _page_text(title)[:max_chars]
    except (requests.RequestException, KeyError, StopIteration) as error:
        print(f"[retrieval] Could not fetch '{title}': {error}")
        return ""


# --- Passages ---------------------------------------------------------------

def make_passages(text: str, source: str,
                  size: int = config.SENTENCES_PER_PASSAGE) -> list[Passage]:
    """Cut text into overlapping windows of `size` sentences.
    Overlap (stride = size - 1) makes sure a fact split across a boundary
    still appears whole in at least one passage."""
    # Wikipedia section headings look like "== Early life ==" — drop them.
    lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("==")]
    sentences = split_sentences(" ".join(lines))
    if len(sentences) <= size:
        return [Passage(" ".join(sentences), source)] if sentences else []
    stride = max(1, size - 1)
    return [Passage(" ".join(sentences[i:i + size]), source)
            for i in range(0, len(sentences) - size + 1, stride)]


def rank_passages(claim: str, passages: list[Passage], top_k: int) -> list[Passage]:
    """Keep the top_k passages whose meaning is closest to the claim."""
    if not passages:
        return []
    embedder = load_embedder()
    # normalize_embeddings=True makes the dot product equal to cosine similarity.
    claim_vec = embedder.encode([claim], normalize_embeddings=True)
    passage_vecs = embedder.encode([p.text for p in passages], normalize_embeddings=True)
    similarities = (passage_vecs @ claim_vec.T).ravel()
    for passage, sim in zip(passages, similarities):
        passage.relevance = float(sim)
    return sorted(passages, key=lambda p: p.relevance, reverse=True)[:top_k]


def retrieve_evidence(claim: str, question: str | None = None, context: str | None = None,
                      top_k: int = config.TOP_K_PASSAGES) -> list[Passage]:
    if context:
        passages = make_passages(context, "Provided context")
    else:
        # Search with the claim, and with the question too — together they find
        # the right pages more reliably than either alone.
        queries = [claim] + ([question] if question else [])
        titles = dict.fromkeys(t for q in queries for t in search_wikipedia(q))  # unique, ordered
        passages = [p for t in titles for p in make_passages(fetch_page_text(t), f"Wikipedia: {t}")]
    return rank_passages(claim, passages, top_k)


if __name__ == "__main__":
    claim = "Albert Einstein's Nobel Prize in Physics was awarded in 1931 for the photoelectric effect."
    print("CLAIM:", claim, "\n")
    for i, p in enumerate(retrieve_evidence(claim), 1):
        print(f"Evidence {i}  (relevance {p.relevance:.2f}, {p.source})")
        print("  ", p.text[:300], "\n")
