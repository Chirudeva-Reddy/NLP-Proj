"""
E5 — Cross-domain zero-shot transfer to HaluEval-QA.

Downloads the official HaluEval QA set, runs inference on faithful and
hallucinated answers, scores them with the frozen RAGTruth stats.pt, and
compares AUROC against an existing RAGTruth test score directory.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import auroc
from src.halueval import ensure_qa_dataset, load_qa
from src.inference import InferenceRunner, load as load_artifact, save
from src.metrics import (
    attention_entropy,
    cie_top3,
    logit_confidence,
    logit_lens,
    mahalanobis,
    mahalanobis_per_layer,
    pca_deviation,
    unpack_symmetric,
)


METRIC_NAMES = [
    "attention_entropy",
    "logit_confidence",
    "cosine_drift",
    "mahalanobis",
    "logit_lens",
    "pca_deviation",
    "cie_top3",
    "composite",
]


def _valid_pt_paths(path: Path) -> list[Path]:
    candidates = sorted(path.glob("*.pt"))
    return [
        item for item in candidates
        if not item.name.startswith("._") and item.stat().st_size > 0
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layers", default="last8")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-tokens", type=int, default=2048)
    parser.add_argument("--limit-items", type=int, default=0)
    parser.add_argument("--stats", type=Path, default=Path("outputs/stats.pt"))
    parser.add_argument("--qa-json", type=Path, default=Path("dataset/halueval/qa_data.json"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/halueval_artifacts"))
    parser.add_argument("--scores-dir", type=Path, default=Path("outputs/halueval_scores"))
    parser.add_argument("--ragtruth-scores-dir", type=Path, default=Path("outputs/scores"))
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


def _pca_deviation_per_layer(hidden: torch.Tensor, pca_stats: dict, n_components: int) -> torch.Tensor:
    state = _finite(hidden)
    mean = _finite(pca_stats["mean"]).unsqueeze(1)
    components = _finite(pca_stats["components"])[:, :n_components, :]
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
    if float(weights.sum().item()) <= 0.0:
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
        total += weights[i] * signs[i] * ((sample_features[name] - train_median[i]) / scale)
    return float(total)


def _metric_score(sample: dict, metric: str) -> float | None:
    if metric == "composite":
        if "composite_sample_score" in sample:
            return float(sample["composite_sample_score"])
        return None
    if metric not in sample:
        return None
    return float(sample[metric].float().max().item())


def _score_split(artifacts_dir: Path, scores_dir: Path, stats_path: Path) -> None:
    split_dir = artifacts_dir / "test"
    paths = _valid_pt_paths(split_dir)
    if not paths:
        raise FileNotFoundError(f"No artifacts in {split_dir}")

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    mahal_stats = stats["mahalanobis"]
    if "inv_cov_tri" in mahal_stats:
        mahal_stats = {
            "mean": mahal_stats["mean"],
            "inv_cov": unpack_symmetric(mahal_stats["inv_cov_tri"], dim=mahal_stats["mean"].shape[-1]),
        }
    pca_stats = stats["pca"]
    cie_top3_layers = stats["cie_top3_layers"]
    metric_pooling: dict[str, str] = stats["metric_pooling"]
    metric_layers: dict[str, list[int]] = stats["metric_layers"]
    metric_layer_weights: dict[str, list[float]] = stats["metric_layer_weights"]
    pca_components_selected = int(stats["pca_components_selected"])
    feature_names: list[str] = list(stats["composite_features"])
    train_median: list[float] = [float(x) for x in stats["composite_train_median"]]
    train_iqr: list[float] = [float(x) for x in stats["composite_train_iqr"]]
    signs: list[float] = [float(x) for x in stats["composite_signs"]]
    weights: list[float] = [float(x) for x in stats["composite_weights"]]

    scores_dir.mkdir(parents=True, exist_ok=True)
    skipped_bad = 0
    for path in tqdm(paths, desc="score halueval", unit="sample", dynamic_ncols=True, file=sys.stdout):
        try:
            artifact = load_artifact(path)
        except (pickle.UnpicklingError, RuntimeError, EOFError, ValueError) as error:
            skipped_bad += 1
            print(f"[e5] skip unreadable artifact {path.name}: {error}")
            continue
        hidden = artifact["hidden_states"]

        cosine_per_layer = _cosine_drift_per_layer(hidden)
        mahal_per_layer = mahalanobis_per_layer(hidden, mahal_stats)
        pca_per_layer_selected = _pca_deviation_per_layer(hidden, pca_stats, pca_components_selected)

        scores: dict[str, torch.Tensor] = {
            "cosine_drift": cosine_per_layer.mean(dim=0),
            "mahalanobis": mahalanobis(hidden, mahal_stats),
            "pca_deviation": pca_deviation(hidden, pca_stats),
            "cie_top3": cie_top3(hidden, mahal_stats, cie_top3_layers),
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
        for metric_name in feature_names:
            sample_features[metric_name] = _sample_feature(
                per_layer_tokens=per_layer_for_features[metric_name],
                layers=[int(index) for index in metric_layers[metric_name]],
                pooling=metric_pooling[metric_name],
                layer_weights=[float(value) for value in metric_layer_weights[metric_name]],
            )

        composite_sample_score = _composite_score(
            sample_features=sample_features,
            feature_names=feature_names,
            train_median=train_median,
            train_iqr=train_iqr,
            signs=signs,
            weights=weights,
        )

        torch.save(
            {
                "sample_id": artifact["sample_id"],
                "has_hallucination": int(artifact["has_hallucination"]),
                "token_labels": artifact["token_labels"],
                **scores,
                "sample_features": sample_features,
                "composite_sample_score": composite_sample_score,
            },
            scores_dir / f"{artifact['sample_id']}.pt",
        )
    if skipped_bad:
        print(f"[e5] skipped {skipped_bad} unreadable artifacts while scoring")


def _evaluate_scores(scores_dir: Path) -> dict[str, float]:
    paths = _valid_pt_paths(scores_dir)
    if not paths:
        raise FileNotFoundError(f"No score files in {scores_dir}")

    labels: list[int] = []
    pooled: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=True)
        labels.append(int(sample["has_hallucination"]))
        for metric in METRIC_NAMES:
            score = _metric_score(sample, metric)
            if score is not None:
                pooled[metric].append(score)

    return {
        metric: auroc(labels, scores)
        for metric, scores in pooled.items()
        if len(scores) == len(labels)
    }


def main() -> None:
    args = parse_args()
    qa_path = ensure_qa_dataset(args.qa_json)
    samples = load_qa(qa_path)
    if args.limit_items:
        samples = samples[: 2 * args.limit_items]
    print(f"[e5] loaded {len(samples)} HaluEval-QA samples from {qa_path}")

    runner = InferenceRunner(
        model_name=args.model,
        layers_spec=args.layers,
        device=args.device,
        max_seq_tokens=args.max_seq_tokens,
    )
    skipped = 0
    for sample in tqdm(samples, desc="infer halueval", unit="sample", dynamic_ncols=True, file=sys.stdout):
        try:
            artifact = runner.run(sample, split="test")
            save(artifact, args.artifacts_dir)
        except ValueError as error:
            skipped += 1
            print(f"[e5] skip {sample.sample_id}: {error}")
    print(f"[e5] inference complete: kept {len(samples) - skipped} / {len(samples)} samples")

    _score_split(args.artifacts_dir, args.scores_dir, args.stats)

    halueval = _evaluate_scores(args.scores_dir)
    ragtruth_paths = sorted(args.ragtruth_scores_dir.glob("*.pt"))
    if not ragtruth_paths:
        print(
            f"\n[e5] HaluEval scoring finished, but no RAGTruth comparison scores were found in "
            f"{args.ragtruth_scores_dir}."
        )
        print("[e5] Current HaluEval AUROC:")
        for metric in METRIC_NAMES:
            if metric in halueval:
                print(f"  {metric:<20} {halueval[metric]:.4f}")
        print(
            "\n[e5] To get the transfer-drop table, generate a test-only RAGTruth score directory and rerun. "
            "Example: python pipeline/3-score.py --split test --output-dir outputs/scores_test"
        )
        return

    ragtruth = _evaluate_scores(args.ragtruth_scores_dir)

    header = f"{'Metric':<20} {'RAGTruth':>9} {'HaluEval':>9} {'Drop':>9}"
    print("\n" + header)
    print("-" * len(header))
    worst_metric = None
    worst_drop = float("-inf")
    for metric in METRIC_NAMES:
        if metric not in ragtruth or metric not in halueval:
            continue
        drop = ragtruth[metric] - halueval[metric]
        if drop > worst_drop:
            worst_drop = drop
            worst_metric = metric
        print(f"{metric:<20} {ragtruth[metric]:>9.4f} {halueval[metric]:>9.4f} {drop:>9.4f}")

    if worst_metric is not None:
        print(f"\n[e5] most brittle metric: {worst_metric} (drop {worst_drop:.4f})")


if __name__ == "__main__":
    main()
