"""
Evaluation metrics for per-sample (or per-token) hallucination detection.

All functions accept:
  labels  list[int]   — binary labels (1 = hallucinated, 0 = faithful)
  scores  list[float] — real-valued anomaly score (higher = more suspicious)

Metrics:
  auroc        — primary ranking metric; threshold-free measure of separation
  bootstrap_ci — percentile confidence interval for AUROC
  spearman     — rank correlation between scores and labels
  f1_span      — span-level F1 (token-level runs; predicted span is a TP if it
                 overlaps any ground-truth span). Used for per-token evaluation.
  f1           — plain binary F1 after thresholding scores. Used for per-sample
                 evaluation where there are no spans.
  ece          — calibration: do score magnitudes match true hallucination rates?
"""

from __future__ import annotations

import math
import random


def auroc(labels: list[int], scores: list[float]) -> float:
    """
    AUROC via the Wilcoxon-Mann-Whitney U statistic.

    Equivalent to sklearn roc_auc_score but has no external dependency.
    The U-statistic counts concordant (positive ranked above negative) pairs;
    dividing by n_pos * n_neg normalises to [0, 1].
    Average ranks handle ties so identical scores don't inflate the result.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"AUROC needs both classes; got pos={n_pos}, neg={n_neg}")

    ordered = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        # Extend the tie group as far as it goes
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(lbl for _, lbl in ordered[i:j])
        i = j

    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_ci(
    labels: list[int],
    scores: list[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for AUROC.

    Resamples (label, score) pairs with replacement n_boot times and returns
    the (alpha/2, 1-alpha/2) quantiles of the resulting AUROC distribution.
    Draws that happen to contain only one class are skipped rather than
    crashing, since they carry no useful information.
    """
    rng = random.Random(seed)
    n = len(labels)
    pairs = list(zip(labels, scores))
    boot_aurocs: list[float] = []

    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        bl, bs = zip(*sample)
        try:
            boot_aurocs.append(auroc(list(bl), list(bs)))
        except ValueError:
            pass  # skip draws with only one class

    boot_aurocs.sort()
    lo = boot_aurocs[int(alpha / 2 * len(boot_aurocs))]
    hi = boot_aurocs[int((1 - alpha / 2) * len(boot_aurocs))]
    return lo, hi


def spearman(labels: list[int], scores: list[float]) -> float:
    """
    Spearman rank correlation between anomaly scores and binary labels.

    Measures whether the ranking of scores agrees with the ranking of labels.
    +1 means perfect agreement (high scores always on hallucinated tokens),
    0 means no agreement, -1 means perfect inversion.
    """
    n = len(labels)

    def _ranks(vals: list) -> list[float]:
        # Assign average ranks to tied values
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i + 1
            while j < n and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in range(i, j):
                r[order[k]] = avg
            i = j
        return r

    rl, rs = _ranks(labels), _ranks(scores)
    mean_l = sum(rl) / n
    mean_s = sum(rs) / n
    num = sum((rl[i] - mean_l) * (rs[i] - mean_s) for i in range(n))
    den = math.sqrt(
        sum((rl[i] - mean_l) ** 2 for i in range(n)) *
        sum((rs[i] - mean_s) ** 2 for i in range(n))
    )
    return num / den if den > 0 else 0.0


def f1_span(labels: list[int], scores: list[float], threshold: float) -> float:
    """
    Span-level F1 after thresholding scores into binary predictions.

    Consecutive predicted-positive tokens form a predicted span. A predicted
    span is a true positive if it overlaps any ground-truth span. This is more
    lenient than token-level F1: a detector that fires near the right location
    gets credit even if the exact token boundaries don't align.
    """
    def _spans(seq: list[int]) -> list[tuple[int, int]]:
        spans, i = [], 0
        while i < len(seq):
            if seq[i]:
                j = i + 1
                while j < len(seq) and seq[j]:
                    j += 1
                spans.append((i, j))
                i = j
            else:
                i += 1
        return spans

    preds = [1 if s >= threshold else 0 for s in scores]
    pred_spans = _spans(preds)
    true_spans = _spans(labels)

    if not pred_spans and not true_spans:
        return 1.0
    if not pred_spans or not true_spans:
        return 0.0

    # A predicted span is a TP if it overlaps at least one ground-truth span
    tp = sum(
        1 for ps, pe in pred_spans
        if any(ps < te and pe > ts for ts, te in true_spans)
    )
    precision = tp / len(pred_spans)
    recall    = tp / len(true_spans)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1(labels: list[int], scores: list[float], threshold: float) -> float:
    """
    Plain binary F1 after thresholding scores.

    Used at the per-sample level where tokens do not form contiguous spans.
    Predictions are 1 if score >= threshold else 0, then TP/FP/FN are counted
    element-wise against labels.
    """
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def ece(labels: list[int], scores: list[float], n_bins: int = 10) -> float:
    """
    Expected Calibration Error.

    Scores are min-max normalised to [0, 1] so they can be interpreted as
    hallucination probability estimates. Within each equal-width probability
    bin, ECE measures the gap between the mean predicted probability and the
    true fraction of hallucinated tokens, weighted by bin size.
    A perfectly calibrated detector has ECE = 0.
    """
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return 0.0
    probs = [(s - lo) / (hi - lo) for s in scores]
    bin_size = 1.0 / n_bins
    error = 0.0
    for b in range(n_bins):
        lo_b = b * bin_size
        hi_b = (b + 1) * bin_size
        idx = [i for i, p in enumerate(probs) if lo_b <= p < hi_b]
        if not idx:
            continue
        avg_conf = sum(probs[i] for i in idx) / len(idx)
        avg_acc  = sum(labels[i] for i in idx) / len(idx)
        # Weight each bin's error by the fraction of tokens it contains
        error += len(idx) / len(labels) * abs(avg_conf - avg_acc)
    return error
