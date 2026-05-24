"""
E2 layer-profile plot — per-sample AUROC at each saved layer for every
hidden-state metric.

Shows how discriminative power varies with depth so the reviewer can see which
layer range carries the hallucination signal. The three layers picked as
"CIE top-3" in pipeline/2-fit.py are highlighted on the x-axis.

Usage:
  python pipeline/plot.py [--artifacts-dir DIR] [--stats FILE] [--output FILE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import auroc
from src.inference import load as load_artifact
from src.metrics import mahalanobis_per_layer, unpack_symmetric


def _cosine_drift_per_layer(hidden: torch.Tensor) -> torch.Tensor:
    """(n_layers, n_tokens) intra-answer drift without mean-pooling across layers."""
    n_layers, n_tokens, _ = hidden.shape
    drift = torch.zeros(n_layers, n_tokens)
    if n_tokens > 1:
        sim = F.cosine_similarity(hidden[:, 1:], hidden[:, :-1], dim=-1, eps=1e-8)
        drift[:, 1:] = 1.0 - sim
    return drift


def _pca_per_layer(hidden: torch.Tensor, stats: dict) -> torch.Tensor:
    diff  = hidden.float() - stats["mean"].float().unsqueeze(1)
    comps = stats["components"].float()
    proj  = torch.einsum("ltd,lkd->ltk", diff, comps)
    recon = torch.einsum("ltk,lkd->ltd", proj, comps)
    return (diff - recon).norm(dim=-1)


def _attention_entropy_per_layer(ae: torch.Tensor) -> torch.Tensor:
    return ae.float()


def _logit_lens_per_layer(kl: torch.Tensor) -> torch.Tensor:
    return kl.float()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", type=Path, default=Path("outputs/artifacts/test"))
    p.add_argument("--stats",          type=Path, default=Path("outputs/stats.pt"))
    p.add_argument("--output",         type=Path, default=Path("outputs/layer_profile.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.artifacts_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No artifacts in {args.artifacts_dir}")

    stats = torch.load(args.stats, map_location="cpu", weights_only=False)
    mahal_stats = stats["mahalanobis"]
    if "inv_cov_tri" in mahal_stats:
        mahal_stats = {
            "mean":    mahal_stats["mean"],
            "inv_cov": unpack_symmetric(mahal_stats["inv_cov_tri"], dim=mahal_stats["mean"].shape[-1]),
        }
    pca_stats       = stats["pca"]
    cie_top3_layers = stats.get("cie_top3_layers", [])
    metric_layers = stats.get("metric_layers", {})

    metric_scores: dict[str, list[list[float]]] = {
        "cosine_drift":      [],
        "mahalanobis":       [],
        "pca_deviation":     [],
        "logit_lens":        [],
        "attention_entropy": [],
    }
    labels: list[int] = []
    n_layers: int | None = None

    for path in paths:
        a = load_artifact(path)
        h = a["hidden_states"]
        if n_layers is None:
            n_layers = h.shape[0]
            for name in metric_scores:
                metric_scores[name] = [[] for _ in range(n_layers)]
        labels.append(int(a.get("has_hallucination", int(a["token_labels"].any()))))

        per_layer = {
            "cosine_drift":  _cosine_drift_per_layer(h),
            "mahalanobis":   mahalanobis_per_layer(h, mahal_stats),
            "pca_deviation": _pca_per_layer(h, pca_stats),
        }
        if "logit_lens_per_layer" in a:
            per_layer["logit_lens"] = _logit_lens_per_layer(a["logit_lens_per_layer"])
        if "attention_entropy_per_layer" in a:
            per_layer["attention_entropy"] = _attention_entropy_per_layer(a["attention_entropy_per_layer"])

        for name, tensor in per_layer.items():
            for l in range(n_layers):
                metric_scores[name][l].append(float(tensor[l].max()))

    layer_aurocs: dict[str, list[float]] = {}
    for name, by_layer in metric_scores.items():
        if any(len(by_layer[l]) == 0 for l in range(n_layers)):
            continue
        layer_aurocs[name] = [auroc(labels, by_layer[l]) for l in range(n_layers)]

    plt.figure(figsize=(9, 5.5))
    line_by_metric: dict[str, object] = {}
    for name, vals in layer_aurocs.items():
        line, = plt.plot(range(n_layers), vals, marker="o", label=name.replace("_", " "))
        line_by_metric[name] = line
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random")
    for name, selected in metric_layers.items():
        if name not in layer_aurocs:
            continue
        indices = [int(idx) for idx in selected if 0 <= int(idx) < n_layers]
        if not indices:
            continue
        values = [layer_aurocs[name][idx] for idx in indices]
        color = line_by_metric[name].get_color()
        plt.scatter(indices, values, color=color, marker="D", s=55)
    plt.xlabel("Saved layer index (0 = shallowest of the saved slice)")
    plt.ylabel("Per-sample AUROC")
    plt.title(f"E2 layer profile — {len(labels)} test samples (selected layers highlighted)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot → {args.output}")
    for name, vals in layer_aurocs.items():
        print(f"  {name:<18}: {[round(v, 4) for v in vals]}")
    if metric_layers:
        print(f"  Selected layers by metric: {metric_layers}")
    if cie_top3_layers:
        print(f"  CIE top-3 layers: {cie_top3_layers}")


if __name__ == "__main__":
    main()
