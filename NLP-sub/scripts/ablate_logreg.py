"""
Train a logistic-regression combiner over the frozen strict-composite features.

This is an ablation only. It does not modify the existing pipeline or stats.pt.
It reuses the same sample-level feature construction as the strict composite,
fits a supervised logistic regression on the train split, tunes regularization
on the validation split, and reports test AUROC against the existing composite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import auroc  # noqa: E402
from src.inference import load as load_artifact  # noqa: E402
from src.metrics import mahalanobis_per_layer, unpack_symmetric  # noqa: E402


FEATURE_NAMES: tuple[str, ...] = (
    "cosine_drift",
    "mahalanobis",
    "logit_lens",
    "pca_deviation",
)
C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
NUMERIC_EPS: float = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/artifacts"))
    parser.add_argument("--scores-test-dir", type=Path, default=Path("outputs/scores_test"))
    parser.add_argument("--stats", type=Path, default=Path("outputs/stats.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation_logreg.json"))
    return parser.parse_args()


def _finite(tensor: torch.Tensor) -> torch.Tensor:
    as_float = torch.as_tensor(tensor).float()
    if torch.isfinite(as_float).all():
        return as_float
    return torch.nan_to_num(as_float, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine_drift_per_layer(hidden: torch.Tensor) -> torch.Tensor:
    state = _finite(hidden)
    n_layers, n_tokens, _ = state.shape
    drift = torch.zeros(n_layers, n_tokens)
    if n_tokens > 1:
        similarity = F.cosine_similarity(state[:, 1:], state[:, :-1], dim=-1, eps=1e-8)
        drift[:, 1:] = 1.0 - similarity
    return drift


def _pca_deviation_per_layer(hidden: torch.Tensor, pca_stats: dict[str, Any], n_components: int) -> torch.Tensor:
    state = _finite(hidden)
    mean = _finite(pca_stats["mean"]).unsqueeze(1)
    full_components = _finite(pca_stats["components"])
    top_k = min(n_components, full_components.shape[1])
    components = full_components[:, :top_k, :]
    diff = state - mean
    projected = torch.einsum("ltd,lkd->ltk", diff, components)
    reconstructed = torch.einsum("ltk,lkd->ltd", projected, components)
    return (diff - reconstructed).norm(dim=-1)


def _pool_tokens(tokens: torch.Tensor, pooling: str) -> float:
    if pooling == "max":
        return float(tokens.max().item())
    return float(tokens.mean().item())


def _sample_feature(
    per_layer_tokens: torch.Tensor,
    layers: list[int],
    pooling: str,
    layer_weights: list[float],
) -> float:
    selected = per_layer_tokens[layers]
    weights = torch.tensor(layer_weights, dtype=selected.dtype)
    if weights.numel() != selected.shape[0]:
        raise ValueError(
            f"Layer-weight size mismatch: got {weights.numel()} weights for {selected.shape[0]} layers."
        )
    weight_sum = float(weights.sum().item())
    if weight_sum <= 0.0:
        reduced = selected.mean(dim=0)
    else:
        normalized = weights / weight_sum
        reduced = (selected * normalized.unsqueeze(1)).sum(dim=0)
    return _pool_tokens(reduced, pooling)


def _prepare_stats(raw_stats: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int]:
    mahal_stats = raw_stats["mahalanobis"]
    if "inv_cov_tri" in mahal_stats:
        mahal_stats = {
            "mean": mahal_stats["mean"],
            "inv_cov": unpack_symmetric(mahal_stats["inv_cov_tri"], dim=mahal_stats["mean"].shape[-1]),
        }
    pca_stats = raw_stats["pca"]
    pca_components_selected = int(raw_stats["pca_components_selected"])
    return mahal_stats, pca_stats, pca_components_selected


def _artifact_features(
    artifact: dict[str, Any],
    raw_stats: dict[str, Any],
    mahal_stats: dict[str, Any],
    pca_stats: dict[str, Any],
    pca_components_selected: int,
) -> dict[str, float]:
    hidden = artifact["hidden_states"]
    cosine_per_layer = _cosine_drift_per_layer(hidden)
    mahal_per_layer = mahalanobis_per_layer(hidden, mahal_stats)
    pca_per_layer = _pca_deviation_per_layer(hidden, pca_stats, n_components=pca_components_selected)
    per_layer: dict[str, torch.Tensor] = {
        "cosine_drift": cosine_per_layer,
        "mahalanobis": mahal_per_layer,
        "pca_deviation": pca_per_layer,
        "logit_lens": _finite(artifact["logit_lens_per_layer"]),
    }
    metric_layers: dict[str, list[int]] = raw_stats["metric_layers"]
    metric_pooling: dict[str, str] = raw_stats["metric_pooling"]
    metric_layer_weights: dict[str, list[float]] = raw_stats["metric_layer_weights"]
    return {
        name: _sample_feature(
            per_layer_tokens=per_layer[name],
            layers=[int(index) for index in metric_layers[name]],
            pooling=metric_pooling[name],
            layer_weights=[float(value) for value in metric_layer_weights[name]],
        )
        for name in FEATURE_NAMES
    }


def _split_matrix(
    split: str,
    artifacts_dir: Path,
    raw_stats: dict[str, Any],
    mahal_stats: dict[str, Any],
    pca_stats: dict[str, Any],
    pca_components_selected: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    split_dir = artifacts_dir / split
    paths = sorted(split_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No artifacts in {split_dir}")
    print(f"[ablation] building features for {split}: {len(paths)} artifacts")
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    for path in tqdm(paths, desc=f"ablation {split}", unit="sample", dynamic_ncols=True, file=sys.stdout):
        artifact = load_artifact(path)
        features = _artifact_features(
            artifact=artifact,
            raw_stats=raw_stats,
            mahal_stats=mahal_stats,
            pca_stats=pca_stats,
            pca_components_selected=pca_components_selected,
        )
        feature_rows.append([features[name] for name in FEATURE_NAMES])
        labels.append(int(artifact.get("has_hallucination", int(artifact["token_labels"].any()))))
        sample_ids.append(str(artifact["sample_id"]))
    return np.asarray(feature_rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), sample_ids


def _robust_scale(
    train_x: np.ndarray,
    other_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.median(train_x, axis=0)
    q1 = np.quantile(train_x, 0.25, axis=0)
    q3 = np.quantile(train_x, 0.75, axis=0)
    iqr = np.maximum(q3 - q1, NUMERIC_EPS)
    return (train_x - med) / iqr, (other_x - med) / iqr, med


def _robust_apply(x: np.ndarray, med: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    q1 = np.quantile(train_x, 0.25, axis=0)
    q3 = np.quantile(train_x, 0.75, axis=0)
    iqr = np.maximum(q3 - q1, NUMERIC_EPS)
    return (x - med) / iqr


def _fit_best_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> tuple[LogisticRegression, float, float]:
    best_model: LogisticRegression | None = None
    best_c = 0.0
    best_auc = -1.0
    for c in C_GRID:
        model = LogisticRegression(
            C=c,
            solver="liblinear",
            penalty="l2",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(train_x, train_y)
        val_scores = model.decision_function(val_x)
        val_auc = float(auroc(val_y.tolist(), val_scores.tolist()))
        if val_auc > best_auc:
            best_model = model
            best_c = c
            best_auc = val_auc
    if best_model is None:
        raise ValueError("Failed to fit any logistic-regression candidate.")
    return best_model, best_c, best_auc


def _existing_test_composite(scores_test_dir: Path) -> tuple[list[int], list[float]]:
    paths = sorted(scores_test_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No score files in {scores_test_dir}")
    labels: list[int] = []
    scores: list[float] = []
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        labels.append(int(sample.get("has_hallucination", int(sample["token_labels"].any()))))
        scores.append(float(sample["composite_sample_score"]))
    return labels, scores


def main() -> None:
    args = parse_args()
    if not args.stats.exists():
        raise FileNotFoundError(f"Stats not found: {args.stats}")
    raw_stats = torch.load(args.stats, map_location="cpu", weights_only=False)
    mahal_stats, pca_stats, pca_components_selected = _prepare_stats(raw_stats)

    train_x_raw, train_y, _ = _split_matrix(
        split="train",
        artifacts_dir=args.artifacts_dir,
        raw_stats=raw_stats,
        mahal_stats=mahal_stats,
        pca_stats=pca_stats,
        pca_components_selected=pca_components_selected,
    )
    val_x_raw, val_y, _ = _split_matrix(
        split="val",
        artifacts_dir=args.artifacts_dir,
        raw_stats=raw_stats,
        mahal_stats=mahal_stats,
        pca_stats=pca_stats,
        pca_components_selected=pca_components_selected,
    )
    test_x_raw, test_y, _ = _split_matrix(
        split="test",
        artifacts_dir=args.artifacts_dir,
        raw_stats=raw_stats,
        mahal_stats=mahal_stats,
        pca_stats=pca_stats,
        pca_components_selected=pca_components_selected,
    )

    train_x, val_x, med = _robust_scale(train_x=train_x_raw, other_x=val_x_raw)
    test_x = _robust_apply(x=test_x_raw, med=med, train_x=train_x_raw)

    model, best_c, best_val_auc = _fit_best_model(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
    )
    print(f"[ablation] fitted logistic regression candidates over grid {list(C_GRID)}")

    test_scores = model.decision_function(test_x)
    test_auc = float(auroc(test_y.tolist(), test_scores.tolist()))

    composite_labels, composite_scores = _existing_test_composite(args.scores_test_dir)
    composite_auc = float(auroc(composite_labels, composite_scores))

    result = {
        "feature_names": list(FEATURE_NAMES),
        "regularization_grid": list(C_GRID),
        "selected_c": best_c,
        "val_auroc": best_val_auc,
        "test_auroc_logreg": test_auc,
        "test_auroc_existing_composite": composite_auc,
        "delta_vs_existing": test_auc - composite_auc,
        "coefficients": {
            name: float(model.coef_[0][idx])
            for idx, name in enumerate(FEATURE_NAMES)
        },
        "intercept": float(model.intercept_[0]),
        "n_train": int(train_x.shape[0]),
        "n_val": int(val_x.shape[0]),
        "n_test": int(test_x.shape[0]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(f"[ablation] selected C = {best_c}")
    print(f"[ablation] val AUROC = {best_val_auc:.4f}")
    print(f"[ablation] test AUROC (logreg) = {test_auc:.4f}")
    print(f"[ablation] test AUROC (existing composite) = {composite_auc:.4f}")
    print(f"[ablation] delta = {test_auc - composite_auc:+.4f}")
    print(f"[ablation] wrote {args.output}")


if __name__ == "__main__":
    main()
