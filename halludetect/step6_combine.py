"""
STEP 6 — Combining the signals into one explainable score
=========================================================

Problem:  We now have up to six numbers per claim:

    evidence_entail, evidence_contradict      (Step 3: retrieval + NLI)
    consistency_support, consistency_conflict (Step 4: self-consistency)
    mean_logprob, min_logprob                 (Step 5: model confidence)

Each signal is right in some cases and wrong in others.  How do we merge them?

Idea:     Logistic regression — deliberately simple and interpretable.

    z     = w1*x1 + w2*x2 + ... + b
    score = 1 / (1 + e^-z)          (the sigmoid squashes z into 0..1)

  * score near 1  -> likely hallucinated, near 0 -> likely correct
  * The weights are LEARNED from labelled examples (HaluEval) by
    train_and_evaluate.py.
  * Because the model is linear, each feature's contribution w_i * x_i can be
    shown to the user: "this claim was flagged mainly because the evidence
    contradicts it".  That is what makes the detector *explainable*.
  * Features are standardized first (mean 0, std 1) so the weights are
    comparable with each other.

Until you train a model, a hand-written heuristic with the same shape is used.

Run the demo:   python -m halludetect.step6_combine
"""
import math
import os
import pickle

from . import config

ALL_FEATURES = [
    "evidence_entail", "evidence_contradict",
    "consistency_support", "consistency_conflict",
    "mean_logprob", "min_logprob",
]

# Plain-English names for the UI.
FEATURE_LABELS = {
    "evidence_entail": "Evidence supports it",
    "evidence_contradict": "Evidence contradicts it",
    "consistency_support": "Other samples agree",
    "consistency_conflict": "Other samples disagree",
    "mean_logprob": "Model's average confidence",
    "min_logprob": "Model's least-confident word",
}


def verdict(score: float) -> str:
    if score < config.LOW_RISK:
        return "Likely correct"
    if score > config.HIGH_RISK:
        return "Likely hallucinated"
    return "Uncertain"


class Combiner:
    def __init__(self, path: str = config.COMBINER_PATH):
        self.path = path
        self.trained = None              # {"features": [...], "pipeline": sklearn Pipeline}
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.trained = pickle.load(f)
            except Exception as error:   # e.g. saved with a different scikit-learn version
                print(f"[combiner] Could not load {path} ({error}). Using heuristic weights; "
                      "run `python train_and_evaluate.py` to retrain.")

    @property
    def is_trained(self) -> bool:
        return self.trained is not None

    # ---- training ----------------------------------------------------------
    def fit(self, rows: list[dict], labels: list[int], features: list[str]):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = [[row[f] for f in features] for row in rows]
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
        pipeline.fit(X, labels)
        self.trained = {"features": features, "pipeline": pipeline}

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.trained, f)

    # ---- prediction --------------------------------------------------------
    def predict(self, feats: dict, evidence_override: bool = config.EVIDENCE_OVERRIDE) -> tuple[float, dict[str, float]]:
        """Returns (hallucination score 0..1, contribution of each feature).
        Positive contribution pushes towards 'hallucinated'."""
        if self.is_trained and all(f in feats for f in self.trained["features"]):
            score, contributions = self._predict_trained(feats)
        else:
            score, contributions = self._predict_heuristic(feats)
        if evidence_override:
            score = self._apply_evidence_rule(score, feats)
        return score, contributions

    @staticmethod
    def _apply_evidence_rule(score: float, feats: dict) -> float:
        """A decisive evidence verdict wins over the other signals (see config.py)."""
        entail = feats.get("evidence_entail", 0.0)
        contra = feats.get("evidence_contradict", 0.0)
        if entail >= config.EVIDENCE_DECISIVE and entail > contra:
            return min(score, config.SUPPORTED_CAP)
        if contra >= config.EVIDENCE_DECISIVE and contra > entail:
            return max(score, config.CONTRADICTED_FLOOR)
        return score

    def _predict_trained(self, feats):
        features = self.trained["features"]
        scaler, logreg = self.trained["pipeline"].steps[0][1], self.trained["pipeline"].steps[1][1]
        x = [[feats[f] for f in features]]
        z_scaled = scaler.transform(x)[0]
        contributions = {f: float(w * v) for f, w, v in zip(features, logreg.coef_[0], z_scaled)}
        score = float(self.trained["pipeline"].predict_proba(x)[0, 1])
        return score, contributions

    @staticmethod
    def _predict_heuristic(feats):
        """Hand-set weights: evidence matters most, then consistency, then confidence."""
        c = {}
        if "evidence_entail" in feats:
            c["evidence_entail"] = -2.5 * feats["evidence_entail"]
            c["evidence_contradict"] = 3.0 * feats["evidence_contradict"]
        if "consistency_support" in feats:
            c["consistency_support"] = -1.0 * feats["consistency_support"]
            c["consistency_conflict"] = 1.5 * feats["consistency_conflict"]
        if "min_logprob" in feats:
            # min_logprob is about -1 for confident text and -8 or lower for surprising text
            c["min_logprob"] = 0.15 * (-feats["min_logprob"] - 4)
        z = 0.3 + sum(c.values())
        return 1 / (1 + math.exp(-z)), c


if __name__ == "__main__":
    combiner = Combiner()
    print("Using trained model" if combiner.is_trained else "Using heuristic (run train_and_evaluate.py to learn weights)")
    examples = {
        "supported by evidence": dict(evidence_entail=0.95, evidence_contradict=0.02, min_logprob=-2.0, mean_logprob=-0.6,
                                      consistency_support=0.7, consistency_conflict=0.1),
        "contradicted by evidence": dict(evidence_entail=0.03, evidence_contradict=0.92, min_logprob=-6.5, mean_logprob=-1.4,
                                         consistency_support=0.1, consistency_conflict=0.6),
        "no evidence either way": dict(evidence_entail=0.2, evidence_contradict=0.1, min_logprob=-4.0, mean_logprob=-1.0,
                                       consistency_support=0.3, consistency_conflict=0.3),
    }
    for name, feats in examples.items():
        score, contrib = combiner.predict(feats)
        top = max(contrib, key=lambda k: abs(contrib[k]))
        print(f"{name:26s} score={score:.2f} -> {verdict(score):20s} main reason: {FEATURE_LABELS[top]}")
