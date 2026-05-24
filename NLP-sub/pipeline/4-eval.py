"""
Step 4 — Per-sample AUROC for every metric.

For each sample we collapse the per-token score tensor into a single scalar
(max-pool by default: a response is as hallucinated as its most suspicious
token) and produce a sample-level binary label (1 if the response has any
hallucination span). AUROC, 95% bootstrap CI, Spearman ρ, F1-span, and ECE
are then computed over one score/label pair per sample.

Usage:
  python pipeline/4-eval.py [--scores-dir DIR] [--aggregate max|mean]
                             [--threshold T] [--n-boot N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import auroc, bootstrap_ci, ece, f1, spearman

METRIC_NAMES = [
    "attention_entropy",   # Baseline 1 (Track B rubric E1)
    "logit_confidence",    # Baseline 2
    "cosine_drift",
    "mahalanobis",
    "logit_lens",
    "pca_deviation",
    "cie_top3",            # Top-3 layers surrogate (replace once E3 provides CIE)
    "composite",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", type=Path, default=Path("outputs/scores"))
    p.add_argument("--aggregate", choices=["max", "mean"], default="max",
                   help="Per-sample token aggregation. max = any-token-hallucinated (standard).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Score threshold for F1/ECE. Defaults to per-metric mean.")
    p.add_argument("--n-boot", type=int, default=1000)
    return p.parse_args()


def _aggregate(tokens: torch.Tensor, mode: str) -> float:
    if mode == "max":
        return float(tokens.max().item())
    return float(tokens.mean().item())


def _metric_score(sample: dict, metric: str, aggregate: str) -> float | None:
    if metric == "composite":
        if "composite_sample_score" in sample:
            return float(sample["composite_sample_score"])
        if "composite" in sample:
            return _aggregate(sample["composite"].float(), aggregate)
        return None
    if metric not in sample:
        return None
    return _aggregate(sample[metric].float(), aggregate)


def main() -> None:
    args = parse_args()
    paths = sorted(args.scores_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No score files in {args.scores_dir}. Run pipeline/3-score.py first.")

    # Pool into per-sample (label, score) pairs for each metric.
    labels: list[int] = []
    pooled: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for path in paths:
        s = torch.load(path, map_location="cpu", weights_only=False)
        labels.append(int(s.get("has_hallucination", int(s["token_labels"].any()))))
        for m in METRIC_NAMES:
            score = _metric_score(sample=s, metric=m, aggregate=args.aggregate)
            if score is not None:
                pooled[m].append(score)

    n_pos = sum(labels)
    print(f"\n{len(labels)} samples | {n_pos} hallucinated ({100*n_pos/len(labels):.1f}%) "
          f"| aggregate = {args.aggregate}\n")

    header = f"{'Metric':<20} {'AUROC':>7}  {'95% CI':^15}  {'Spearman':>9}  {'F1':>7}  {'ECE':>7}"
    print(header)
    print("-" * len(header))

    for metric in METRIC_NAMES:
        scores = pooled[metric]
        if len(scores) != len(labels):
            print(f"{metric:<20}  (missing)")
            continue
        try:
            auc = auroc(labels, scores)
            lo, hi = bootstrap_ci(labels, scores, n_boot=args.n_boot)
        except ValueError:
            print(f"{metric:<20}  (single class — need more samples)")
            continue

        rho = spearman(labels, scores)
        threshold = args.threshold if args.threshold is not None else sum(scores) / len(scores)
        f1_val = f1(labels, scores, threshold)
        cal = ece(labels, scores)
        print(f"{metric:<20} {auc:>7.4f}  [{lo:.4f}, {hi:.4f}]  {rho:>9.4f}  {f1_val:>7.4f}  {cal:>7.4f}")


if __name__ == "__main__":
    main()
