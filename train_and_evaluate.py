"""
STEP 7 — Training the combiner and evaluating on a benchmark
============================================================

Dataset: HaluEval QA (Li et al., 2023).  Each item has
    knowledge            – a Wikipedia snippet
    question
    right_answer         – correct
    hallucinated_answer  – plausible but wrong (written by ChatGPT on purpose)
So every item gives us two labelled examples: (right, label 0), (hallucinated, label 1).

Pipeline:
  1. download the data (6 MB, once)
  2. sample N items, compute features for both answers  (cached to data/)
  3. split train/test BY ITEM — both answers to the same question go to the same
     side, otherwise the model could "cheat" by having seen the question already
  4. report every single signal on its own, then the combined model
  5. save the trained combiner to models/combiner.pkl (used by the app)

Metrics explained:
  * Accuracy  – % of answers labelled correctly
  * Precision – of answers we flagged as hallucinated, % that really were
  * Recall    – of real hallucinations, % we caught
  * F1        – harmonic mean of precision and recall
  * AUROC     – probability that a random hallucinated answer gets a higher
                score than a random correct one.  0.5 = coin flip, 1.0 = perfect.
                Doesn't depend on a threshold, so it's the fairest comparison.

Usage:
    python train_and_evaluate.py                    # 200 items, NLI + confidence (~5-10 min)
    python train_and_evaluate.py --n-items 500
    python train_and_evaluate.py --consistency      # also self-consistency (much slower)
"""
import argparse
import json
import os
import random

import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from halludetect.detector import compute_features
from halludetect.step1_claims import Claim, make_statement
from halludetect.step4_consistency import generate_samples
from halludetect.step6_combine import ALL_FEATURES, FEATURE_LABELS, Combiner

DATA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
DATA_PATH = "data/halueval_qa.jsonl"


def load_halueval() -> list[dict]:
    if not os.path.exists(DATA_PATH):
        print("Downloading HaluEval QA ...")
        os.makedirs("data", exist_ok=True)
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        with open(DATA_PATH, "w") as f:
            f.write(response.text)
    with open(DATA_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_examples(items: list[dict], use_consistency: bool, cache_path: str) -> list[dict]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = {row["key"]: row for row in map(json.loads, f)}

    examples = []
    with open(cache_path, "a") as cache_file:
        for n, (idx, item) in enumerate(items, 1):
            samples = None
            for answer_field, label in [("right_answer", 0), ("hallucinated_answer", 1)]:
                key = f"{idx}-{label}"
                if key not in cache:
                    if use_consistency and samples is None:
                        samples = generate_samples(item["question"])
                    answer = item[answer_field]
                    # Score the answer as ONE claim (HaluEval answers are a single fact).
                    claim = Claim(text=answer, statement=make_statement(answer, item["question"]))
                    features, _ = compute_features(claim, item["question"], item["knowledge"], "",
                                                   samples, use_retrieval=True, use_confidence=True)
                    cache[key] = {"key": key, "item": idx, "label": label, "statement": claim.statement,
                                  "n_words": len(answer.split()),
                                  "n_words_statement": len(claim.statement.split()), **features}
                    cache_file.write(json.dumps(cache[key]) + "\n")
                    cache_file.flush()
                examples.append(cache[key])
            if n % 20 == 0:
                print(f"  features computed for {n}/{len(items)} items")
    return examples


def auroc_with_direction(values, labels):
    auc = roc_auc_score(labels, values)
    return (auc, "higher = hallucinated") if auc >= 0.5 else (1 - auc, "lower = hallucinated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--consistency", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_halueval()
    rng = random.Random(args.seed)
    items = rng.sample(list(enumerate(data)), args.n_items)

    cache_path = f"data/features{'_cons' if args.consistency else ''}.jsonl"
    print(f"Computing features for {len(items)} items ({2 * len(items)} answers) ...")
    examples = build_examples(items, args.consistency, cache_path)

    # --- split by item --------------------------------------------------------
    item_ids = sorted({e["item"] for e in examples})
    rng.shuffle(item_ids)
    test_ids = set(item_ids[:int(len(item_ids) * args.test_frac)])
    train = [e for e in examples if e["item"] not in test_ids]
    test = [e for e in examples if e["item"] in test_ids]
    y_test = [e["label"] for e in test]
    print(f"\nTrain: {len(train)} answers   Test: {len(test)} answers\n")

    features = [f for f in ALL_FEATURES if f in examples[0]]

    # --- each signal alone ----------------------------------------------------
    print("=" * 70)
    print("Each signal on its own (test set AUROC)")
    print("=" * 70)
    results = {"single_features": {}}
    for f in features:
        auc, direction = auroc_with_direction([e[f] for e in test], y_test)
        results["single_features"][f] = auc
        print(f"  {FEATURE_LABELS[f]:32s} AUROC {auc:.3f}   ({direction})")
    for field, name in [("n_words", "raw answer length"), ("n_words_statement", "statement length")]:
        auc, direction = auroc_with_direction([e[field] for e in test], y_test)
        results["single_features"][f"diagnostic_{field}"] = auc
        print(f"  {'[diagnostic] ' + name:32s} AUROC {auc:.3f}   ({direction})")
    print("  ^ Length is a dataset artifact (right answers are short), NOT a real signal.\n"
          "    Converting answers to full statements (Step 1) should shrink this shortcut.\n"
          "    Reported only so you can discuss dataset bias; never used as a feature.")

    # --- combined model ---------------------------------------------------------
    combiner = Combiner()
    combiner.fit(train, [e["label"] for e in train], features)

    def evaluate(evidence_override: bool) -> dict:
        scores = [combiner.predict(e, evidence_override)[0] for e in test]
        preds = [int(s >= 0.5) for s in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
        return {"accuracy": accuracy_score(y_test, preds), "precision": precision,
                "recall": recall, "f1": f1, "auroc": roc_auc_score(y_test, scores)}

    results["combined"] = evaluate(evidence_override=False)
    results["combined_with_evidence_rule"] = evaluate(evidence_override=True)

    print("\n" + "=" * 70)
    print("Combined detector (logistic regression)")
    print("=" * 70)
    print(f"  {'':10s} {'learned only':>13s} {'+ evidence rule':>16s}")
    for name in results["combined"]:
        print(f"  {name:10s} {results['combined'][name]:13.3f} "
              f"{results['combined_with_evidence_rule'][name]:16.3f}")
    print("  The evidence rule (config.py) trades a little HaluEval accuracy for\n"
          "  decisive verdicts when Wikipedia clearly supports/contradicts a claim.")

    logreg = combiner.trained["pipeline"].steps[1][1]
    print("\nLearned weights (positive = pushes towards 'hallucinated'):")
    for f, w in sorted(zip(features, logreg.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {FEATURE_LABELS[f]:32s} {w:+.2f}")

    combiner.save()
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model to {combiner.path} and metrics to results/metrics.json")


if __name__ == "__main__":
    main()
