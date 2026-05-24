"""
E4 — Temporal precedence of hallucination signals.

For each hallucinated response, align on the first hallucinated token (onset,
t=0) and read per-token metric values at relative offsets t-3, t-2, t-1, t, t+1.
A metric that "sees hallucination early" rises at negative offsets. We test
this with a one-sided Mann-Whitney U: H1 = metric(t-k) > metric(t) for k∈{3,2,1}
and H1 = metric(t+1) < metric(t) for the post-onset control.

Inputs : outputs/scores/*.pt  (per-token tensors, produced by pipeline/3-score.py)
Outputs: outputs/e4/temporal.csv   — metric × offset × mean × MWU p-value
         outputs/e4/temporal.json  — same numbers, machine-readable
         outputs/e4/temporal.png   — line plot, one line per metric
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))


METRIC_NAMES: list[str] = [
    "cosine_drift",
    "logit_lens",
    "mahalanobis",
    "pca_deviation",
    "attention_entropy",
    "logit_confidence",
]

OFFSETS: list[int] = [-3, -2, -1, 0, 1]
SCORES_DIR: Path = Path("outputs/scores")
OUT_DIR: Path = Path("outputs/e4")


def _find_onset(token_labels: torch.Tensor) -> int:
    """Index of the first hallucinated token, or -1 if none."""
    positives = (token_labels == 1).nonzero(as_tuple=False)
    if positives.numel() == 0:
        return -1
    return int(positives[0, 0].item())


def _collect_offset_values(
    paths: list[Path],
    metrics: list[str],
    offsets: list[int],
) -> dict[str, dict[int, list[float]]]:
    """Return values[metric][offset] = list of per-sample metric values."""
    values: dict[str, dict[int, list[float]]] = {
        m: {o: [] for o in offsets} for m in metrics
    }
    kept = 0
    total_hal = 0
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if int(sample.get("has_hallucination", 0)) != 1:
            continue
        total_hal += 1
        labels = sample["token_labels"]
        onset = _find_onset(labels)
        n = labels.shape[0]
        min_off = min(offsets)
        max_off = max(offsets)
        if onset < -min_off or onset + max_off >= n:
            continue
        kept += 1
        for metric in metrics:
            if metric not in sample:
                continue
            series = sample[metric].float()
            for off in offsets:
                values[metric][off].append(float(series[onset + off].item()))
    print(f"[e4] hallucinated samples: {total_hal}, usable after edge filter: {kept}")
    return values


def _mwu_p_greater(x: list[float], y: list[float]) -> float:
    """One-sided Mann-Whitney U: H1 = x > y (stochastically)."""
    if len(x) < 5 or len(y) < 5:
        return float("nan")
    stat = mannwhitneyu(x, y, alternative="greater")
    return float(stat.pvalue)


def _mwu_p_less(x: list[float], y: list[float]) -> float:
    if len(x) < 5 or len(y) < 5:
        return float("nan")
    stat = mannwhitneyu(x, y, alternative="less")
    return float(stat.pvalue)


def _compute_rows(
    values: dict[str, dict[int, list[float]]],
    metrics: list[str],
    offsets: list[int],
) -> list[dict]:
    rows: list[dict] = []
    for metric in metrics:
        per_off = values[metric]
        onset_vals = per_off[0]
        for off in offsets:
            vals = per_off[off]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            if off < 0:
                p = _mwu_p_greater(vals, onset_vals)
            elif off == 0:
                p = float("nan")
            else:
                p = _mwu_p_less(vals, onset_vals)
            rows.append({
                "metric": metric,
                "offset": off,
                "n": len(vals),
                "mean": mean,
                "p_value_vs_onset": p,
            })
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "offset", "n", "mean", "p_value_vs_onset"])
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2))


def _plot(
    values: dict[str, dict[int, list[float]]],
    metrics: list[str],
    offsets: list[int],
    path: Path,
) -> None:
    """One line per metric; y is z-scored within the metric so scales are comparable."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric in metrics:
        means: list[float] = []
        for off in offsets:
            vals = values[metric][off]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        series = torch.tensor(means)
        mu = series.mean()
        sd = series.std().clamp_min(1e-8)
        z = ((series - mu) / sd).tolist()
        ax.plot(offsets, z, marker="o", label=metric)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("offset from first hallucinated token (t=0)")
    ax.set_ylabel("z-scored mean metric value")
    ax.set_title("E4 — temporal precedence of hallucination signals")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_table(rows: list[dict]) -> None:
    header = f"{'metric':<20} {'offset':>6} {'n':>5} {'mean':>12} {'p_vs_onset':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        p_str = "        nan" if row["p_value_vs_onset"] != row["p_value_vs_onset"] else f"{row['p_value_vs_onset']:>12.3e}"
        print(f"{row['metric']:<20} {row['offset']:>6d} {row['n']:>5d} {row['mean']:>12.5f} {p_str}")


def main() -> None:
    paths = sorted(SCORES_DIR.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No score files in {SCORES_DIR}. Run pipeline/3-score.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = _collect_offset_values(paths=paths, metrics=METRIC_NAMES, offsets=OFFSETS)
    rows = _compute_rows(values=values, metrics=METRIC_NAMES, offsets=OFFSETS)
    _print_table(rows)
    _write_csv(rows, OUT_DIR / "temporal.csv")
    _write_json(rows, OUT_DIR / "temporal.json")
    _plot(values=values, metrics=METRIC_NAMES, offsets=OFFSETS, path=OUT_DIR / "temporal.png")
    print(f"\n[e4] wrote {OUT_DIR}/temporal.{{csv,json,png}}")


if __name__ == "__main__":
    main()
