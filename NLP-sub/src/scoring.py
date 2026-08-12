"""
Reusable frozen scoring helpers for single-sample and batch scoring.

This module centralizes the strict-composite scoring logic so the offline batch
pipeline and the live demo both use the same formulas and compatibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .metrics import (
    attention_entropy,
    cie_top3,
    logit_confidence,
    logit_lens,
    mahalanobis,
    mahalanobis_per_layer,
    pca_deviation,
    unpack_symmetric,
)


@dataclass(frozen=True)
class FrozenScoringConfig:
    stats_path: Path
    mahal_stats: dict[str, torch.Tensor]
    pca_stats: dict[str, torch.Tensor]
    cie_top3_layers: list[int]
    metric_pooling: dict[str, str]
    metric_layers: dict[str, list[int]]
    metric_layer_weights: dict[str, list[float]]
    pca_components_selected: int
    feature_names: list[str]
    train_median: list[float]
    train_iqr: list[float]
    signs: list[float]
    weights: list[float]
    expected_n_layers: int
    expected_hidden_dim: int


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


def _pca_deviation_per_layer(hidden: torch.Tensor, pca_stats: dict, n_components: int) -> torch.Tensor:
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
        normalized = weights / weights.sum()
        reduced = (selected * normalized.unsqueeze(1)).sum(dim=0)
    return _pool_tokens(reduced, pooling)


def _composite_score(
    sample_features: dict[str, float],
    feature_names: list[str],
    train_median: list[float],
    train_iqr: list[float],
    signs: list[float],
    weights: list[float],
) -> float:
    total = 0.0
    for i, name in enumerate(feature_names):
        scale = train_iqr[i] if train_iqr[i] > 1e-8 else 1.0
        z = (sample_features[name] - train_median[i]) / scale
        total += weights[i] * signs[i] * z
    return float(total)


def load_frozen_scoring_config(stats_path: Path) -> FrozenScoringConfig:
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    required = (
        "metric_pooling",
        "metric_layers",
        "metric_layer_weights",
        "pca_components_selected",
        "composite_rule",
        "composite_features",
        "composite_train_median",
        "composite_train_iqr",
        "composite_signs",
        "composite_weights",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise KeyError(
            f"Missing strict-composite keys in {stats_path}: {missing}. "
            "Re-run pipeline/2-fit.py with the weighted-zscore strict implementation."
        )
    if stats["composite_rule"] != "weighted_zscore":
        raise ValueError(
            f"Unknown composite_rule {stats['composite_rule']!r}. "
            "Re-run pipeline/2-fit.py."
        )

    mahal_stats = stats["mahalanobis"]
    if "inv_cov_tri" in mahal_stats:
        mahal_stats = {
            "mean": _finite(mahal_stats["mean"]),
            "inv_cov": unpack_symmetric(mahal_stats["inv_cov_tri"], dim=mahal_stats["mean"].shape[-1]),
        }
    else:
        mahal_stats = {
            "mean": _finite(mahal_stats["mean"]),
            "inv_cov": _finite(mahal_stats["inv_cov"]),
        }

    expected_n_layers = int(mahal_stats["mean"].shape[0])
    expected_hidden_dim = int(mahal_stats["mean"].shape[-1])

    return FrozenScoringConfig(
        stats_path=stats_path,
        mahal_stats=mahal_stats,
        pca_stats=stats["pca"],
        cie_top3_layers=[int(index) for index in stats["cie_top3_layers"]],
        metric_pooling={str(k): str(v) for k, v in stats["metric_pooling"].items()},
        metric_layers={str(k): [int(index) for index in v] for k, v in stats["metric_layers"].items()},
        metric_layer_weights={
            str(k): [float(value) for value in v]
            for k, v in stats["metric_layer_weights"].items()
        },
        pca_components_selected=int(stats["pca_components_selected"]),
        feature_names=[str(name) for name in stats["composite_features"]],
        train_median=[float(x) for x in stats["composite_train_median"]],
        train_iqr=[float(x) for x in stats["composite_train_iqr"]],
        signs=[float(x) for x in stats["composite_signs"]],
        weights=[float(x) for x in stats["composite_weights"]],
        expected_n_layers=expected_n_layers,
        expected_hidden_dim=expected_hidden_dim,
    )


def validate_artifact_matches_config(artifact: dict, config: FrozenScoringConfig) -> None:
    hidden = artifact["hidden_states"]
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden_states with 3 dims, got shape {tuple(hidden.shape)}.")

    n_layers, _, hidden_dim = hidden.shape
    if n_layers != config.expected_n_layers:
        raise ValueError(
            f"Artifact layer count {n_layers} does not match frozen stats layer count "
            f"{config.expected_n_layers} from {config.stats_path}."
        )
    if hidden_dim != config.expected_hidden_dim:
        raise ValueError(
            f"Artifact hidden size {hidden_dim} does not match frozen stats hidden size "
            f"{config.expected_hidden_dim} from {config.stats_path}."
        )
    if "logit_lens" in config.feature_names and "logit_lens_per_layer" not in artifact:
        raise KeyError("Artifact is missing logit_lens_per_layer required by the frozen composite.")


def score_artifact(artifact: dict, config: FrozenScoringConfig) -> dict:
    validate_artifact_matches_config(artifact, config)

    hidden = artifact["hidden_states"]
    cosine_per_layer = _cosine_drift_per_layer(hidden)
    mahal_per_layer = mahalanobis_per_layer(hidden, config.mahal_stats)
    pca_per_layer_selected = _pca_deviation_per_layer(
        hidden,
        config.pca_stats,
        n_components=config.pca_components_selected,
    )

    scores: dict[str, torch.Tensor] = {
        "cosine_drift": cosine_per_layer.mean(dim=0),
        "mahalanobis": mahalanobis(hidden, config.mahal_stats),
        "pca_deviation": pca_deviation(hidden, config.pca_stats),
        "cie_top3": cie_top3(hidden, config.mahal_stats, config.cie_top3_layers),
    }
    if "logit_lens_per_layer" in artifact:
        scores["logit_lens"] = logit_lens(artifact["logit_lens_per_layer"])
    if "attention_entropy_per_layer" in artifact:
        scores["attention_entropy"] = attention_entropy(artifact["attention_entropy_per_layer"])
    if "logit_confidence" in artifact:
        scores["logit_confidence"] = logit_confidence(artifact["logit_confidence"])

    per_layer_for_features: dict[str, torch.Tensor] = {
        "cosine_drift": cosine_per_layer,
        "mahalanobis": mahal_per_layer,
        "pca_deviation": pca_per_layer_selected,
    }
    if "logit_lens_per_layer" in artifact:
        per_layer_for_features["logit_lens"] = _finite(artifact["logit_lens_per_layer"])

    sample_features: dict[str, float] = {}
    for metric_name in config.feature_names:
        if metric_name not in per_layer_for_features:
            raise KeyError(f"Feature metric {metric_name} missing from artifact {artifact['sample_id']}.")
        sample_features[metric_name] = _sample_feature(
            per_layer_tokens=per_layer_for_features[metric_name],
            layers=config.metric_layers[metric_name],
            pooling=config.metric_pooling[metric_name],
            layer_weights=config.metric_layer_weights[metric_name],
        )

    composite_sample_score = _composite_score(
        sample_features=sample_features,
        feature_names=config.feature_names,
        train_median=config.train_median,
        train_iqr=config.train_iqr,
        signs=config.signs,
        weights=config.weights,
    )

    return {
        "sample_id": artifact["sample_id"],
        "has_hallucination": int(artifact.get("has_hallucination", int(artifact["token_labels"].any()))),
        "token_labels": artifact["token_labels"],
        **scores,
        "sample_features": sample_features,
        "composite_sample_score": composite_sample_score,
    }


def aggregate_token_scores(score_payload: dict, mode: str) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for name, value in score_payload.items():
        if name in {"sample_id", "has_hallucination", "token_labels", "sample_features", "composite_sample_score"}:
            continue
        if isinstance(value, torch.Tensor):
            aggregated[name] = _pool_tokens(value.float(), mode)
    aggregated["composite"] = float(score_payload["composite_sample_score"])
    return aggregated
