"""
The full pipeline: Steps 1-6 wired together.

    answer ──► Step 1 claims ──► for each claim:
                                   Step 2 retrieve evidence ─► Step 3 NLI ─┐
                                   Step 4 self-consistency ───────────────┤──► Step 6 score + verdict
                                   Step 5 token confidence ───────────────┘

Usage:
    from halludetect.detector import HallucinationDetector
    report = HallucinationDetector().check(answer="...", question="...")

Run the demo:   python -m halludetect.detector
"""
from dataclasses import dataclass, field

from .step1_claims import Claim, extract_claims
from .step2_retrieval import Passage, retrieve_evidence
from .step3_nli import EvidenceResult, verify_claim
from .step4_consistency import ConsistencyResult, consistency_score, generate_samples
from .step5_confidence import ConfidenceResult, claim_confidence
from .step6_combine import Combiner, verdict


@dataclass
class ClaimReport:
    claim: Claim
    score: float
    verdict: str
    features: dict
    contributions: dict
    passages: list[Passage] = field(default_factory=list)
    evidence: EvidenceResult | None = None
    consistency: ConsistencyResult | None = None
    confidence: ConfidenceResult | None = None


@dataclass
class AnswerReport:
    claims: list[ClaimReport]
    overall_score: float      # = the worst claim: one false fact makes the answer unreliable
    verdict: str
    samples: list[str]


def compute_features(claim: Claim, question: str | None, context: str | None, preceding_text: str,
                     samples: list[str] | None, use_retrieval: bool, use_confidence: bool):
    """Run Steps 2-5 for one claim.  Shared by the detector and the training script."""
    features, parts = {}, {}

    if use_retrieval:
        passages = retrieve_evidence(claim.statement, question=question, context=context)
        evidence = verify_claim(claim.statement, passages)
        features["evidence_entail"] = evidence.entailment
        features["evidence_contradict"] = evidence.contradiction
        parts.update(passages=passages, evidence=evidence)

    if samples:
        consistency = consistency_score(claim.statement, samples)
        features["consistency_support"] = consistency.support
        features["consistency_conflict"] = consistency.conflict
        parts["consistency"] = consistency

    if use_confidence:
        # Short answers: score the full statement, not "Delhi" — a lone name always
        # looks "surprising", which would make confidence a disguised length measure.
        # Normal sentences: score the text as written; the rewritten version
        # ("He ..." -> "Albert Einstein ...") reads unnaturally after the preceding text.
        text = claim.statement if len(claim.text.split()) <= 5 else claim.text
        confidence = claim_confidence(text, question, preceding_text)
        features["mean_logprob"] = confidence.mean_logprob
        features["min_logprob"] = confidence.min_logprob
        parts["confidence"] = confidence

    return features, parts


class HallucinationDetector:
    def __init__(self, use_retrieval=True, use_consistency=True, use_confidence=True, n_samples=3):
        self.use_retrieval = use_retrieval
        self.use_consistency = use_consistency
        self.use_confidence = use_confidence
        self.n_samples = n_samples
        self.combiner = Combiner()

    def check(self, answer: str, question: str | None = None, context: str | None = None) -> AnswerReport:
        claims = extract_claims(answer, question)

        # Samples depend only on the question, so generate them once for all claims.
        samples = []
        if self.use_consistency and question:
            samples = generate_samples(question, n=self.n_samples)

        reports, preceding = [], ""
        for claim in claims:
            features, parts = compute_features(claim, question, context, preceding, samples,
                                               self.use_retrieval, self.use_confidence)
            score, contributions = self.combiner.predict(features)
            reports.append(ClaimReport(claim, score, verdict(score), features, contributions, **parts))
            preceding += " " + claim.text

        overall = max((r.score for r in reports), default=0.0)
        return AnswerReport(reports, overall, verdict(overall), samples)


if __name__ == "__main__":
    question = "Tell me about Albert Einstein."
    answer = ("Albert Einstein was a German-born theoretical physicist. "
              "He developed the theory of relativity. "
              "His Nobel Prize in Physics was awarded in 1931 for the photoelectric effect.")
    report = HallucinationDetector().check(answer, question)
    print(f"OVERALL: {report.verdict} (score {report.overall_score:.2f})\n")
    for r in report.claims:
        print(f"[{r.verdict:19s} {r.score:.2f}] {r.claim.text}")
        if r.evidence:
            print(f"     evidence: {r.evidence.label} (entail {r.evidence.entailment:.2f}, "
                  f"contra {r.evidence.contradiction:.2f})")
            if r.evidence.best_passage:
                print(f"     source  : {r.evidence.best_passage.source}")
