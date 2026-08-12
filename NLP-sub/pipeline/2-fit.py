"""
Step 2 — Fit train stats and freeze a strict non-supervised E1 composite.

This step:
  - Fits Mahalanobis and PCA statistics on TRAIN only (faithful tokens only).
  - Picks CIE top-3 layers on VAL (auxiliary metric only, not in composite).
  - Tunes representation-metric sample features on VAL:
      * pooling: max or mean
      * layer subset: single, contiguous-2, contiguous-4, full saved slice
      * PCA rank: k in {4, 8, 16} by slicing one fitted PCA basis
  - Sweeps Mahalanobis regularization λ in {1e-4, 1e-3, 1e-2} and selects the
    λ with the best validation composite AUROC (tie-break: smaller λ).
  - Builds a non-supervised weighted z-score composite over the four
    representation features. Standardization stats are fit on TRAIN using
    median/IQR, layer weights are derived on VAL within selected layer subsets,
    and metric signs/weights are derived from per-metric VAL AUROC.

Output: outputs/stats.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import auroc
from src.inference import load as load_artifact
from src.metrics import (
    attention_entropy,
    fit_mahalanobis,
    fit_pca,
    logit_confidence,
    logit_lens,
    mahalanobis_per_layer,
    pack_symmetric,
)

REP_METRICS = ("cosine_drift", "mahalanobis", "logit_lens", "pca_deviation")
BASELINE_METRICS = ("attention_entropy", "logit_confidence")
POOLINGS = ("max", "mean")
PCA_CANDIDATES = (4, 8, 16)
MAHAL_REG_CANDIDATES = (1e-4, 1e-3, 1e-2)
NUMERIC_EPS = 1e-8
# Sharpness of the val-AUROC weighting curve. α=1 gives weights proportional
# to each metric's val excess AUROC over chance; α=0.5 flattens; α=2 sharpens
# toward the strongest metric. Defensible as a single transparent constant.
COMPOSITE_WEIGHT_ALPHA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stats.pt"))
    parser.add_argument("--pca-components", type=int, default=16)
    return parser.parse_args()


def _finite(tensor: torch.Tensor) -> torch.Tensor:
    as_float = torch.as_tensor(tensor).float()
    if torch.isfinite(as_float).all():
        return as_float
    return torch.nan_to_num(as_float, nan=0.0, posinf=0.0, neginf=0.0)


def _check_split_leakage(artifacts_dir: Path) -> None:
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        split_dir = artifacts_dir / split
        split_ids[split] = {path.stem for path in split_dir.glob("*.pt")}
    overlap_tv = split_ids["train"] & split_ids["val"]
    overlap_tt = split_ids["train"] & split_ids["test"]
    overlap_vt = split_ids["val"] & split_ids["test"]
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            "Split leakage detected: "
            f"train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )
    print(
        "[leakage-check] "
        f"train={len(split_ids['train'])}  val={len(split_ids['val'])}  test={len(split_ids['test'])}  "
        "— all sample_ids disjoint."
    )


def _iter_artifacts(paths: list[Path], desc: str):
    for path in tqdm(paths, desc=desc, unit="artifact", dynamic_ncols=True, file=sys.stdout):
        yield load_artifact(path)


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


def _layer_candidates(n_layers: int) -> list[list[int]]:
    candidates: list[list[int]] = []
    candidates.extend([[index] for index in range(n_layers)])
    if n_layers >= 2:
        candidates.extend([list(range(start, start + 2)) for start in range(n_layers - 1)])
    if n_layers >= 4:
        candidates.extend([list(range(start, start + 4)) for start in range(n_layers - 3)])
    candidates.append(list(range(n_layers)))

    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for layers in candidates:
        key = tuple(layers)
        if key not in seen:
            seen.add(key)
            unique.append(layers)
    return unique


def _sample_scores_from_config(
    per_sample_per_layer: list[torch.Tensor],
    layers: list[int],
    pooling: str,
    layer_weights: list[float] | None = None,
) -> list[float]:
    scores: list[float] = []
    for per_layer in per_sample_per_layer:
        selected = per_layer[layers]
        if layer_weights is None:
            reduced = selected.mean(dim=0)
        else:
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
        scores.append(_pool_tokens(reduced, pooling))
    return scores


def _pick_cie_top3_layers(
    mahal_per_sample: list[torch.Tensor],
    token_labels_per_sample: list[torch.Tensor],
) -> list[int]:
    scores = torch.cat(mahal_per_sample, dim=1)
    labels = torch.cat(token_labels_per_sample, dim=0).numpy()
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("Validation tokens are single-class; cannot rank CIE layers.")

    n_layers = scores.shape[0]
    per_layer_aurocs = [float(roc_auc_score(labels, scores[layer].numpy())) for layer in range(n_layers)]
    ranked = sorted(range(n_layers), key=lambda layer: per_layer_aurocs[layer], reverse=True)
    print(f"[cie-top3] per-layer AUROC on val tokens: {[round(value, 4) for value in per_layer_aurocs]}")
    print(f"[cie-top3] selected layers: {ranked[:3]}")
    return ranked[:3]


def _tune_representation_metric(
    labels: list[int],
    metric_name: str,
    per_sample_per_layer_by_variant: dict[int, list[torch.Tensor]],
    layer_candidates: list[list[int]],
) -> tuple[dict, list[float]]:
    best_config: dict | None = None
    best_scores: list[float] = []
    best_auc = float("-inf")

    for variant_key, per_sample_per_layer in per_sample_per_layer_by_variant.items():
        for pooling in POOLINGS:
            for layers in layer_candidates:
                scores = _sample_scores_from_config(per_sample_per_layer, layers, pooling)
                try:
                    auc_value = float(auroc(labels, scores))
                except ValueError:
                    continue
                if auc_value > best_auc:
                    best_auc = auc_value
                    best_scores = scores
                    best_config = {
                        "metric": metric_name,
                        "pooling": pooling,
                        "layers": layers,
                        "auroc": auc_value,
                        "variant": variant_key,
                    }

    if best_config is None:
        raise ValueError(f"Could not tune metric {metric_name}: no valid AUROC configuration.")
    return best_config, best_scores


def _derive_layer_weights(
    labels: list[int],
    per_sample_per_layer: list[torch.Tensor],
    layers: list[int],
    pooling: str,
) -> tuple[list[float], list[float], list[float]]:
    per_layer_aurocs: list[float] = []
    for layer in layers:
        layer_scores = _sample_scores_from_config(
            per_sample_per_layer=per_sample_per_layer,
            layers=[layer],
            pooling=pooling,
        )
        per_layer_aurocs.append(float(auroc(labels, layer_scores)))

    raw = [max(0.0, auc_value - 0.5) for auc_value in per_layer_aurocs]
    total_raw = sum(raw)
    if total_raw <= 0.0:
        layer_weights = [1.0 / len(layers)] * len(layers)
    else:
        layer_weights = [value / total_raw for value in raw]

    weighted_scores = _sample_scores_from_config(
        per_sample_per_layer=per_sample_per_layer,
        layers=layers,
        pooling=pooling,
        layer_weights=layer_weights,
    )
    return layer_weights, per_layer_aurocs, weighted_scores


def _train_feature_robust_stats(
    train_paths: list[Path],
    mahal_stats: dict,
    pca_stats: dict,
    feature_names: list[str],
    metric_pooling: dict[str, str],
    metric_layers: dict[str, list[int]],
    metric_layer_weights: dict[str, list[float]],
    pca_components_selected: int,
) -> tuple[list[float], list[float]]:
    """
    Streaming train pass: compute the four per-metric sample features under
    their frozen val-tuned configs, then return per-metric (median, IQR) over
    train samples for robust standardization.

    No artifacts are held in memory beyond the current one — protects against
    the OOM that the eager-list version of 2-fit hit on the 17k corpus.
    """
    n = len(feature_names)
    values_by_feature: list[list[float]] = [[] for _ in range(n)]
    count = 0

    for artifact in _iter_artifacts(train_paths, "train-composite"):
        hidden = artifact["hidden_states"]
        per_layer_by_metric: dict[str, torch.Tensor] = {
            "cosine_drift":  _cosine_drift_per_layer(hidden),
            "mahalanobis":   mahalanobis_per_layer(hidden, mahal_stats),
            "pca_deviation": _pca_deviation_per_layer(hidden, pca_stats, n_components=pca_components_selected),
        }
        if "logit_lens_per_layer" in artifact:
            per_layer_by_metric["logit_lens"] = _finite(artifact["logit_lens_per_layer"])

        for i, name in enumerate(feature_names):
            if name not in per_layer_by_metric:
                # Should be impossible: feature_names is built from the val
                # pass which already enforced presence. Keep the guard so a
                # corrupted train artifact fails loudly rather than silently
                # biasing the standardization stats.
                raise KeyError(f"Train artifact missing per-layer tensor for {name}.")
            value = _sample_scores_from_config(
                per_sample_per_layer=[per_layer_by_metric[name]],
                layers=metric_layers[name],
                pooling=metric_pooling[name],
                layer_weights=metric_layer_weights[name],
            )[0]
            values_by_feature[i].append(value)
        count += 1

    if count == 0:
        raise ValueError("No train artifacts available for composite standardization.")

    median: list[float] = []
    iqr: list[float] = []
    for values in values_by_feature:
        tensor = torch.tensor(values, dtype=torch.float32)
        q1 = float(torch.quantile(tensor, 0.25).item())
        q2 = float(torch.quantile(tensor, 0.50).item())
        q3 = float(torch.quantile(tensor, 0.75).item())
        median.append(q2)
        iqr.append(max(q3 - q1, NUMERIC_EPS))
    return median, iqr


def _build_weighted_zscore_composite(
    train_paths: list[Path],
    mahal_stats: dict,
    pca_stats: dict,
    metric_pooling: dict[str, str],
    metric_layers: dict[str, list[int]],
    metric_layer_weights: dict[str, list[float]],
    pca_components_selected: int,
    val_metric_aurocs: dict[str, float],
    val_features: dict[str, list[float]],
    val_labels: list[int],
) -> dict:
    """
    Frozen non-supervised composite: weight every representation feature by
    its val excess AUROC over chance, sign-correct inverted-direction features,
    standardize each with train median/IQR. Recipe is fully deterministic given
    the val AUROC table — no CV, no learned parameters.
    """
    feature_names = list(REP_METRICS)

    train_median, train_iqr = _train_feature_robust_stats(
        train_paths=train_paths,
        mahal_stats=mahal_stats,
        pca_stats=pca_stats,
        feature_names=feature_names,
        metric_pooling=metric_pooling,
        metric_layers=metric_layers,
        metric_layer_weights=metric_layer_weights,
        pca_components_selected=pca_components_selected,
    )

    signs = [1.0 if val_metric_aurocs[name] >= 0.5 else -1.0 for name in feature_names]

    raw = [max(0.0, abs(val_metric_aurocs[name] - 0.5)) ** COMPOSITE_WEIGHT_ALPHA for name in feature_names]
    total_raw = sum(raw)
    if total_raw <= 0.0:
        # Every metric collapsed to chance on val. Fall back to a uniform
        # prior so the composite is still defined; downstream eval will
        # show it sitting at ~0.5 and you'll know to fix the upstream
        # signal rather than tweak the weighting rule.
        weights = [1.0 / len(feature_names)] * len(feature_names)
    else:
        weights = [r / total_raw for r in raw]

    composite_val_scores: list[float] = []
    for sample_idx in range(len(val_labels)):
        total = 0.0
        for i, name in enumerate(feature_names):
            z = (val_features[name][sample_idx] - train_median[i]) / max(train_iqr[i], NUMERIC_EPS)
            total += weights[i] * signs[i] * z
        composite_val_scores.append(total)
    composite_val_auroc = float(auroc(val_labels, composite_val_scores))

    return {
        "composite_rule":       "weighted_zscore",
        "composite_features":   feature_names,
        "composite_train_median": train_median,
        "composite_train_iqr":  train_iqr,
        "composite_signs":      signs,
        "composite_weights":    weights,
        "composite_val_auroc":  composite_val_auroc,
    }


def _evaluate_mahal_candidate(
    train_paths: list[Path],
    val_paths: list[Path],
    pca_stats: dict,
    regularization: float,
) -> dict:
    print(f"  [candidate] fitting Mahalanobis on train faithful tokens (lambda={regularization:g}) …")
    mahal_stats = fit_mahalanobis(_iter_artifacts(train_paths, f"train-mahal-{regularization:g}"), regularization=regularization)

    print("  collecting validation tensors for strict composite tuning …")
    val_labels: list[int] = []
    baseline_scores: dict[str, dict[str, list[float]]] = {
        name: {pooling: [] for pooling in POOLINGS}
        for name in BASELINE_METRICS
    }
    rep_tensors: dict[str, list[torch.Tensor]] = {
        "cosine_drift": [],
        "mahalanobis": [],
        "logit_lens": [],
    }
    pca_tensors: dict[int, list[torch.Tensor]] = {k: [] for k in PCA_CANDIDATES}
    val_token_labels: list[torch.Tensor] = []
    for artifact in _iter_artifacts(val_paths, f"val-metrics-{regularization:g}"):
        hidden = artifact["hidden_states"]
        val_labels.append(int(artifact.get("has_hallucination", int(artifact["token_labels"].any()))))
        val_token_labels.append(artifact["token_labels"].int())

        rep_tensors["cosine_drift"].append(_cosine_drift_per_layer(hidden))
        rep_tensors["mahalanobis"].append(mahalanobis_per_layer(hidden, mahal_stats))
        if "logit_lens_per_layer" not in artifact:
            raise ValueError("Strict composite requires logit_lens_per_layer in val artifacts.")
        rep_tensors["logit_lens"].append(_finite(artifact["logit_lens_per_layer"]))

        for k in PCA_CANDIDATES:
            pca_tensors[k].append(_pca_deviation_per_layer(hidden, pca_stats, n_components=k))

        if "attention_entropy_per_layer" in artifact:
            ae = attention_entropy(artifact["attention_entropy_per_layer"])
            baseline_scores["attention_entropy"]["max"].append(float(ae.max().item()))
            baseline_scores["attention_entropy"]["mean"].append(float(ae.mean().item()))
        if "logit_confidence" in artifact:
            lc = logit_confidence(artifact["logit_confidence"])
            baseline_scores["logit_confidence"]["max"].append(float(lc.max().item()))
            baseline_scores["logit_confidence"]["mean"].append(float(lc.mean().item()))

    print("  selecting CIE top-3 layers on val (auxiliary) …")
    cie_top3_layers = _pick_cie_top3_layers(
        mahal_per_sample=rep_tensors["mahalanobis"],
        token_labels_per_sample=val_token_labels,
    )

    if not val_labels:
        raise ValueError("No validation artifacts available for tuning.")

    n_layers = rep_tensors["cosine_drift"][0].shape[0]
    layer_candidates = _layer_candidates(n_layers)

    print("  tuning representation metrics for sample-level AUROC …")
    metric_layers: dict[str, list[int]] = {}
    metric_pooling: dict[str, str] = {}
    metric_layer_weights: dict[str, list[float]] = {}
    metric_layer_aurocs: dict[str, list[float]] = {}
    metric_val_auroc: dict[str, float] = {}
    selected_scores: dict[str, list[float]] = {}
    pca_components_selected = max(PCA_CANDIDATES)

    for metric_name in ("cosine_drift", "mahalanobis", "logit_lens"):
        config, _ = _tune_representation_metric(
            labels=val_labels,
            metric_name=metric_name,
            per_sample_per_layer_by_variant={0: rep_tensors[metric_name]},
            layer_candidates=layer_candidates,
        )
        selected_layers = list(config["layers"])
        selected_pooling = str(config["pooling"])
        layer_weights, per_layer_aucs, weighted_scores = _derive_layer_weights(
            labels=val_labels,
            per_sample_per_layer=rep_tensors[metric_name],
            layers=selected_layers,
            pooling=selected_pooling,
        )
        metric_layers[metric_name] = selected_layers
        metric_pooling[metric_name] = selected_pooling
        metric_layer_weights[metric_name] = layer_weights
        metric_layer_aurocs[metric_name] = per_layer_aucs
        metric_val_auroc[metric_name] = float(auroc(val_labels, weighted_scores))
        selected_scores[metric_name] = weighted_scores

    pca_config, _ = _tune_representation_metric(
        labels=val_labels,
        metric_name="pca_deviation",
        per_sample_per_layer_by_variant={k: pca_tensors[k] for k in PCA_CANDIDATES},
        layer_candidates=layer_candidates,
    )
    pca_components_selected = int(pca_config["variant"])
    selected_layers = list(pca_config["layers"])
    selected_pooling = str(pca_config["pooling"])
    pca_selected_tensors = pca_tensors[pca_components_selected]
    layer_weights, per_layer_aucs, weighted_scores = _derive_layer_weights(
        labels=val_labels,
        per_sample_per_layer=pca_selected_tensors,
        layers=selected_layers,
        pooling=selected_pooling,
    )
    metric_layers["pca_deviation"] = selected_layers
    metric_pooling["pca_deviation"] = selected_pooling
    metric_layer_weights["pca_deviation"] = layer_weights
    metric_layer_aurocs["pca_deviation"] = per_layer_aucs
    metric_val_auroc["pca_deviation"] = float(auroc(val_labels, weighted_scores))
    selected_scores["pca_deviation"] = weighted_scores

    print("  building weighted-zscore composite (non-supervised, frozen) …")
    composite = _build_weighted_zscore_composite(
        train_paths=train_paths,
        mahal_stats=mahal_stats,
        pca_stats=pca_stats,
        metric_pooling=metric_pooling,
        metric_layers=metric_layers,
        metric_layer_weights=metric_layer_weights,
        pca_components_selected=pca_components_selected,
        val_metric_aurocs=metric_val_auroc,
        val_features=selected_scores,
        val_labels=val_labels,
    )

    baseline_val_aurocs: dict[str, dict[str, float]] = {}
    for baseline in BASELINE_METRICS:
        per_pool: dict[str, float] = {}
        for pooling in POOLINGS:
            scores = baseline_scores[baseline][pooling]
            if len(scores) != len(val_labels):
                continue
            per_pool[pooling] = float(auroc(val_labels, scores))
        baseline_val_aurocs[baseline] = per_pool

    return {
        "regularization": regularization,
        "mahalanobis": mahal_stats,
        "pca_components_selected": pca_components_selected,
        "cie_top3_layers": cie_top3_layers,
        "metric_layers": metric_layers,
        "metric_pooling": metric_pooling,
        "metric_layer_weights": metric_layer_weights,
        "metric_layer_aurocs": metric_layer_aurocs,
        "strict_metric_val_auroc": metric_val_auroc,
        "baseline_val_auroc": baseline_val_aurocs,
        "composite": composite,
    }


def main() -> None:
    args = parse_args()
    _check_split_leakage(args.artifacts_dir)

    train_paths = sorted((args.artifacts_dir / "train").glob("*.pt"))
    val_paths = sorted((args.artifacts_dir / "val").glob("*.pt"))
    if not train_paths:
        raise FileNotFoundError(f"No artifacts found in {args.artifacts_dir / 'train'}")
    if not val_paths:
        raise FileNotFoundError(f"No artifacts found in {args.artifacts_dir / 'val'}")

    print(f"Loading {len(train_paths)} train and {len(val_paths)} val artifacts …")

    max_components = max(PCA_CANDIDATES)
    if args.pca_components < max_components:
        raise ValueError(
            f"--pca-components must be >= {max_components} for strict tuning; got {args.pca_components}."
        )

    print(f"  fitting PCA basis on train faithful tokens (k={args.pca_components}) …")
    pca_stats = fit_pca(_iter_artifacts(train_paths, "train-pca"), n_components=args.pca_components)

    best_result: dict | None = None
    for regularization in MAHAL_REG_CANDIDATES:
        print(f"\n=== Mahalanobis regularization candidate: {regularization:g} ===")
        candidate = _evaluate_mahal_candidate(
            train_paths=train_paths,
            val_paths=val_paths,
            pca_stats=pca_stats,
            regularization=regularization,
        )
        candidate_auc = float(candidate["composite"]["composite_val_auroc"])
        print(f"[candidate {regularization:g}] composite val AUROC: {candidate_auc:.4f}")
        if best_result is None:
            best_result = candidate
            continue
        best_auc = float(best_result["composite"]["composite_val_auroc"])
        if candidate_auc > best_auc + 1e-12:
            best_result = candidate
            continue
        if abs(candidate_auc - best_auc) <= 1e-12 and regularization < float(best_result["regularization"]):
            best_result = candidate

    if best_result is None:
        raise ValueError("No valid Mahalanobis regularization candidate produced a composite result.")

    selected_regularization = float(best_result["regularization"])
    mahal_stats = best_result["mahalanobis"]
    cie_top3_layers = best_result["cie_top3_layers"]
    metric_layers = best_result["metric_layers"]
    metric_pooling = best_result["metric_pooling"]
    metric_layer_weights = best_result["metric_layer_weights"]
    metric_layer_aurocs = best_result["metric_layer_aurocs"]
    pca_components_selected = int(best_result["pca_components_selected"])
    metric_val_auroc = best_result["strict_metric_val_auroc"]
    baseline_val_aurocs = best_result["baseline_val_auroc"]
    composite = best_result["composite"]

    feature_names = composite["composite_features"]
    weights_by_name = {name: round(composite["composite_weights"][i], 4) for i, name in enumerate(feature_names)}
    signs_by_name = {name: int(composite["composite_signs"][i]) for i, name in enumerate(feature_names)}

    print(f"\n=== Selected Mahalanobis regularization: {selected_regularization:g} ===")
    print(f"[strict] selected metric pooling: {metric_pooling}")
    print(f"[strict] selected metric layers: {metric_layers}")
    print(f"[strict] selected metric layer weights: { {k: [round(v, 4) for v in vals] for k, vals in metric_layer_weights.items()} }")
    print(f"[strict] selected metric per-layer AUROC (subset): { {k: [round(v, 4) for v in vals] for k, vals in metric_layer_aurocs.items()} }")
    print(f"[strict] selected PCA components: {pca_components_selected}")
    print(f"[strict] selected Mahalanobis regularization: {selected_regularization:g}")
    print(f"[strict] per-metric val AUROC: { {k: round(v, 4) for k, v in metric_val_auroc.items()} }")
    print(f"[strict] composite weights: {weights_by_name}")
    print(f"[strict] composite signs: {signs_by_name}")
    print(f"[strict] composite val AUROC (weighted zscore): {composite['composite_val_auroc']:.4f}")
    print(
        "[strict] baseline val AUROC (max/mean): "
        f"{ {k: {pk: round(pv, 4) for pk, pv in v.items()} for k, v in baseline_val_aurocs.items()} }"
    )

    best_baseline_by_pool = {
        name: (max(per_pool.items(), key=lambda kv: kv[1]) if per_pool else ("-", float("nan")))
        for name, per_pool in baseline_val_aurocs.items()
    }
    comp_val = composite["composite_val_auroc"]
    beats_both = all(comp_val >= best_baseline_by_pool[name][1] for name in BASELINE_METRICS)
    print("[strict] composite vs baselines (best pool per baseline):")
    print(f"   composite = {comp_val:.4f}")
    for name in BASELINE_METRICS:
        pool, auc_value = best_baseline_by_pool[name]
        print(f"   {name}_best = {auc_value:.4f} (pool={pool})")
    print(f"   beats both: {beats_both}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mahal_packed = {
        "mean": mahal_stats["mean"],
        "inv_cov_tri": pack_symmetric(mahal_stats["inv_cov"]),
    }
    torch.save(
        {
            "mahalanobis": mahal_packed,
            "pca": pca_stats,
            "cie_top3_layers": cie_top3_layers,
            "metric_pooling": metric_pooling,
            "metric_layers": metric_layers,
            "metric_layer_weights": metric_layer_weights,
            "pca_components_selected": pca_components_selected,
            "mahalanobis_regularization": selected_regularization,
            "composite_rule":       composite["composite_rule"],
            "composite_features":   composite["composite_features"],
            "composite_train_median": composite["composite_train_median"],
            "composite_train_iqr":  composite["composite_train_iqr"],
            "composite_signs":      composite["composite_signs"],
            "composite_weights":    composite["composite_weights"],
            "composite_val_auroc":  composite["composite_val_auroc"],
            "metric_layer_val_auroc": metric_layer_aurocs,
            "strict_metric_val_auroc": metric_val_auroc,
            "baseline_val_auroc": baseline_val_aurocs,
        },
        args.output,
    )
    print(f"Saved strict stats → {args.output}")


if __name__ == "__main__":
    main()
