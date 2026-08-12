"""
E7 — failure-case analysis with targeted component replay.

This script selects three deterministic failure cases from the frozen RAGTruth
test scores:

  1. false negative: hallucinated sample with the lowest composite score
  2. false positive: faithful sample with the highest composite score
  3. metric disagreement: sample with the largest variance across robustly
     normalized sample features

The three cases must come from distinct source groups. For each selected case
the script reloads the sample metadata from the dataset join, replays the model
on that single sample to capture `self_attn` and `mlp` outputs, and writes a
compact dossier in JSON, Markdown, and PNG form.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.component_outputs import (  # noqa: E402
    capture_component_outputs,
    contiguous_thirds,
    load_model_and_tokenizer,
    pick_device,
    tokenize_sample,
    update_direction_drift,
)
from src.dataset import Sample, load as load_samples, split as split_samples  # noqa: E402


COMPONENTS: tuple[str, ...] = ("self_attn", "mlp")
RANGES_ORDER: tuple[str, ...] = ("early", "mid", "late")
TRACE_METRICS: tuple[str, ...] = (
    "cosine_drift",
    "mahalanobis",
    "logit_lens",
    "pca_deviation",
)
CASE_TYPES: tuple[str, ...] = (
    "false_negative",
    "false_positive",
    "metric_disagreement",
)
EPS: float = 1e-8


@dataclass
class ScoreRecord:
    sample_id: str
    source_id: str
    label: int
    case_rank_value: float
    disagreement: float
    composite_sample_score: float
    sample_features: dict[str, float]
    z_features: dict[str, float]
    contributions: dict[str, float]
    path: Path
    sample: Sample
    payload: dict[str, Any]


@dataclass
class ReplaySummary:
    strongest_range: str
    preferred_component: str
    preference_strength: float
    per_range: dict[str, dict[str, float]]
    per_layer_mean: dict[str, list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scores-dir", type=Path, default=Path("outputs/scores_test"))
    parser.add_argument("--stats", type=Path, default=Path("outputs/stats.pt"))
    parser.add_argument("--e6-json", type=Path, default=Path("outputs/e6/component_drift.json"))
    parser.add_argument("--response-jsonl", type=Path, default=Path("dataset/ragtruth/response.jsonl"))
    parser.add_argument("--source-jsonl", type=Path, default=Path("dataset/ragtruth/source_info.jsonl"))
    parser.add_argument("--max-seq-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e7"))
    return parser.parse_args()


def _load_global_e6_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_test_samples(
    response_jsonl: Path,
    source_jsonl: Path,
    scores_dir: Path,
) -> dict[str, Sample]:
    score_ids = {path.stem for path in scores_dir.glob("*.pt")}
    samples = load_samples(response_jsonl=response_jsonl, source_jsonl=source_jsonl)
    test_samples = split_samples(samples)["test"]
    return {sample.sample_id: sample for sample in test_samples if sample.sample_id in score_ids}


def _load_stats(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_score_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _robust_z(sample_features: dict[str, float], stats: dict[str, Any]) -> dict[str, float]:
    names: list[str] = list(stats["composite_features"])
    medians: list[float] = list(stats["composite_train_median"])
    iqrs: list[float] = list(stats["composite_train_iqr"])
    return {
        name: (float(sample_features[name]) - float(medians[idx])) / max(float(iqrs[idx]), EPS)
        for idx, name in enumerate(names)
    }


def _composite_contributions(z_features: dict[str, float], stats: dict[str, Any]) -> dict[str, float]:
    names: list[str] = list(stats["composite_features"])
    weights: list[float] = list(stats["composite_weights"])
    signs: list[float] = list(stats["composite_signs"])
    return {
        name: float(weights[idx]) * float(signs[idx]) * float(z_features[name])
        for idx, name in enumerate(names)
    }


def _metric_disagreement(z_features: dict[str, float]) -> float:
    values = np.array(list(z_features.values()), dtype=np.float64)
    return float(values.var())


def _build_score_records(
    scores_dir: Path,
    samples_by_id: dict[str, Sample],
    stats: dict[str, Any],
) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    for path in sorted(scores_dir.glob("*.pt")):
        payload = _load_score_payload(path)
        sample_id = str(payload["sample_id"])
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"Missing dataset metadata for sample_id={sample_id}")
        sample_features = {name: float(value) for name, value in payload["sample_features"].items()}
        z_features = _robust_z(sample_features=sample_features, stats=stats)
        contributions = _composite_contributions(z_features=z_features, stats=stats)
        disagreement = _metric_disagreement(z_features)
        records.append(
            ScoreRecord(
                sample_id=sample_id,
                source_id=sample.source_id,
                label=int(payload["has_hallucination"]),
                case_rank_value=float(payload["composite_sample_score"]),
                disagreement=disagreement,
                composite_sample_score=float(payload["composite_sample_score"]),
                sample_features=sample_features,
                z_features=z_features,
                contributions=contributions,
                path=path,
                sample=sample,
                payload=payload,
            )
        )
    return records


def _pick_case(
    candidates: list[ScoreRecord],
    used_sources: set[str],
    used_samples: set[str],
    case_name: str,
) -> ScoreRecord:
    for record in candidates:
        if record.source_id in used_sources:
            continue
        if record.sample_id in used_samples:
            continue
        return record
    raise ValueError(f"Could not select a distinct {case_name} case.")


def _select_cases(records: list[ScoreRecord]) -> dict[str, ScoreRecord]:
    false_negative_candidates = sorted(
        [record for record in records if record.label == 1],
        key=lambda record: (record.composite_sample_score, record.sample_id),
    )
    false_positive_candidates = sorted(
        [record for record in records if record.label == 0],
        key=lambda record: (-record.composite_sample_score, record.sample_id),
    )
    disagreement_candidates = sorted(
        records,
        key=lambda record: (-record.disagreement, record.sample_id),
    )

    used_sources: set[str] = set()
    used_samples: set[str] = set()
    selected: dict[str, ScoreRecord] = {}

    for case_name, candidates in (
        ("false_negative", false_negative_candidates),
        ("false_positive", false_positive_candidates),
        ("metric_disagreement", disagreement_candidates),
    ):
        record = _pick_case(
            candidates=candidates,
            used_sources=used_sources,
            used_samples=used_samples,
            case_name=case_name,
        )
        used_sources.add(record.source_id)
        used_samples.add(record.sample_id)
        selected[case_name] = record

    return selected


def _excerpt(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _span_summaries(sample: Sample) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for span in sample.spans[:3]:
        snippet = sample.response[span.start:span.end]
        rows.append(
            {
                "start": int(span.start),
                "end": int(span.end),
                "label": span.label,
                "text": _excerpt(snippet, 100),
            }
        )
    return rows


def _normalized_trace(values: torch.Tensor) -> list[float]:
    array = values.detach().cpu().float().numpy()
    lo = float(array.min())
    hi = float(array.max())
    if hi - lo <= EPS:
        return [0.0 for _ in array]
    return [float((x - lo) / (hi - lo)) for x in array]


def _per_sample_reduction(
    drift: torch.Tensor,
    answer_start: int,
    ranges: dict[str, list[int]],
) -> tuple[np.ndarray, dict[str, float]]:
    answer_drift = drift[:, answer_start:]
    per_layer = answer_drift.mean(dim=1).numpy()
    per_range = {
        name: float(answer_drift[layer_ids].mean().item())
        for name, layer_ids in ranges.items()
    }
    return per_layer, per_range


def _component_reading(per_range: dict[str, dict[str, float]]) -> tuple[str, str, float]:
    strongest_range = max(
        RANGES_ORDER,
        key=lambda range_name: max(per_range["self_attn"][range_name], per_range["mlp"][range_name]),
    )
    attn = per_range["self_attn"][strongest_range]
    mlp = per_range["mlp"][strongest_range]
    if math.isclose(mlp, attn, rel_tol=0.0, abs_tol=1e-4):
        return strongest_range, "balanced", 0.0
    if mlp > attn:
        return strongest_range, "mlp", mlp - attn
    return strongest_range, "self_attn", attn - mlp


def _replay_case(
    model: Any,
    tokenizer: Any,
    sample: Sample,
    device: str,
    max_seq_tokens: int,
) -> ReplaySummary:
    tokenized = tokenize_sample(
        tokenizer=tokenizer,
        sample=sample,
        max_seq_tokens=max_seq_tokens,
        device=device,
    )
    if tokenized is None:
        raise ValueError(f"Replay tokenization failed for sample_id={sample.sample_id}")
    captures = capture_component_outputs(model=model, input_ids=tokenized.input_ids)
    ranges = contiguous_thirds(model.config.num_hidden_layers)

    per_layer_mean: dict[str, list[float]] = {}
    per_range: dict[str, dict[str, float]] = {}
    for component in COMPONENTS:
        drift = update_direction_drift(captures[component])
        layer_mean, range_mean = _per_sample_reduction(
            drift=drift,
            answer_start=tokenized.answer_start,
            ranges=ranges,
        )
        per_layer_mean[component] = [float(value) for value in layer_mean.tolist()]
        per_range[component] = {name: float(range_mean[name]) for name in RANGES_ORDER}

    strongest_range, preferred_component, preference_strength = _component_reading(per_range=per_range)
    return ReplaySummary(
        strongest_range=strongest_range,
        preferred_component=preferred_component,
        preference_strength=float(preference_strength),
        per_range=per_range,
        per_layer_mean=per_layer_mean,
    )


def _misfired_metrics_text(case_type: str, z_features: dict[str, float], contributions: dict[str, float]) -> str:
    if case_type == "false_negative":
        weakest = min(contributions.items(), key=lambda item: (item[1], item[0]))
        return (
            f"The composite stayed too low because {weakest[0]} contributed least "
            f"({weakest[1]:+.3f}), so the hallucinated sample did not separate strongly."
        )
    if case_type == "false_positive":
        strongest = max(contributions.items(), key=lambda item: (item[1], item[0]))
        return (
            f"The composite was pushed up mainly by {strongest[0]} "
            f"({strongest[1]:+.3f}), creating a high anomaly score on a faithful answer."
        )
    hi = max(z_features.items(), key=lambda item: (item[1], item[0]))
    lo = min(z_features.items(), key=lambda item: (item[1], item[0]))
    return (
        f"The metrics disagree sharply: {hi[0]} is high ({hi[1]:+.3f}) while "
        f"{lo[0]} is low ({lo[1]:+.3f}), so the sample mixes conflicting signals."
    )


def _mechanistic_text(case_type: str, replay: ReplaySummary) -> str:
    component_name = "FFN/MLP" if replay.preferred_component == "mlp" else "self-attention"
    if replay.preferred_component == "balanced":
        return (
            f"Replay shows the strongest drift in the {replay.strongest_range} layers, "
            "but FFN and attention are nearly matched, so the failure looks mechanistically mixed."
        )
    if case_type == "false_negative":
        return (
            f"Replay shows the strongest update-direction drift in the {replay.strongest_range} layers, "
            f"leaning toward {component_name}, which suggests the detector under-reacted to a localized "
            "internal shift rather than a broad anomaly across all metrics."
        )
    if case_type == "false_positive":
        return (
            f"Replay is strongest in the {replay.strongest_range} layers and leans toward {component_name}, "
            "so the detector likely treated a sharp but faithful internal update as hallucination-like."
        )
    return (
        f"Replay peaks in the {replay.strongest_range} layers and leans toward {component_name}, "
        "which fits a borderline case where different representation metrics react to different internal cues."
    )


def _plot_case(
    record: ScoreRecord,
    replay: ReplaySummary,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [2.2, 1.8, 1.2]})

    ax = axes[0]
    n_tokens = len(record.payload["token_labels"])
    xs = np.arange(n_tokens)
    for metric_name in TRACE_METRICS:
        ax.plot(xs, _normalized_trace(record.payload[metric_name]), linewidth=1.6, label=metric_name)
    token_labels = record.payload["token_labels"].numpy()
    for idx, label in enumerate(token_labels):
        if int(label) == 1:
            ax.axvspan(idx - 0.5, idx + 0.5, color="red", alpha=0.08)
    ax.set_title(f"{record.sample_id} token traces (normalized)")
    ax.set_ylabel("relative score")
    ax.set_xlabel("answer token")
    ax.legend(fontsize=8, ncol=2)

    bx = axes[1]
    layer_indices = np.arange(len(replay.per_layer_mean["self_attn"]))
    bx.plot(layer_indices, replay.per_layer_mean["self_attn"], marker="o", markersize=3, label="self_attn")
    bx.plot(layer_indices, replay.per_layer_mean["mlp"], marker="o", markersize=3, label="mlp")
    bx.set_title("per-layer update-direction drift")
    bx.set_xlabel("transformer layer")
    bx.set_ylabel("mean answer drift")
    bx.legend(fontsize=9)

    cx = axes[2]
    x = np.arange(len(RANGES_ORDER))
    width = 0.38
    attn_vals = [replay.per_range["self_attn"][name] for name in RANGES_ORDER]
    mlp_vals = [replay.per_range["mlp"][name] for name in RANGES_ORDER]
    cx.bar(x - width / 2, attn_vals, width, label="self_attn")
    cx.bar(x + width / 2, mlp_vals, width, label="mlp")
    cx.set_xticks(x)
    cx.set_xticklabels(RANGES_ORDER)
    cx.set_ylabel("range mean")
    cx.set_title("range-pooled replay evidence")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _case_json(
    case_type: str,
    record: ScoreRecord,
    replay: ReplaySummary,
    trace_path: Path,
) -> dict[str, Any]:
    return {
        "case_type": case_type,
        "sample_id": record.sample_id,
        "source_id": record.source_id,
        "label": record.label,
        "prompt_excerpt": _excerpt(record.sample.prompt, 220),
        "response_excerpt": _excerpt(record.sample.response, 220),
        "hallucination_spans": _span_summaries(record.sample),
        "sample_features": record.sample_features,
        "z_features": record.z_features,
        "composite_contributions": record.contributions,
        "composite_sample_score": record.composite_sample_score,
        "metric_disagreement": record.disagreement,
        "misfired_metrics": _misfired_metrics_text(
            case_type=case_type,
            z_features=record.z_features,
            contributions=record.contributions,
        ),
        "replay_summary": asdict(replay),
        "mechanistic_explanation": _mechanistic_text(case_type=case_type, replay=replay),
        "trace_plot": str(trace_path),
    }


def _markdown(global_e6: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    localization = global_e6["localization"]
    lines = [
        "# E7 Failure Cases",
        "",
        "Population E6 context:",
        (
            f"- Global localization: **{localization['claim']}** "
            f"(best self_attn AUROC {localization['attn_best_auroc']:.4f}, "
            f"best mlp AUROC {localization['mlp_best_auroc']:.4f})"
        ),
        "- The case writeups below use targeted 3-sample replay for case-specific evidence.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['case_type'].replace('_', ' ').title()} — sample {case['sample_id']}",
                "",
                f"- `source_id`: `{case['source_id']}`",
                f"- label: `{case['label']}`",
                f"- composite score: `{case['composite_sample_score']:.4f}`",
                f"- prompt excerpt: {case['prompt_excerpt']}",
                f"- response excerpt: {case['response_excerpt']}",
                f"- metric read: {case['misfired_metrics']}",
                (
                    f"- replay evidence: strongest range = `{case['replay_summary']['strongest_range']}`, "
                    f"preferred component = `{case['replay_summary']['preferred_component']}`"
                ),
                f"- mechanistic explanation: {case['mechanistic_explanation']}",
                f"- trace plot: `{case['trace_plot']}`",
                "",
            ]
        )
        if case["hallucination_spans"]:
            lines.append("Hallucination spans:")
            for span in case["hallucination_spans"]:
                lines.append(
                    f"- `{span['label']}` [{span['start']}, {span['end']}): {span['text']}"
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples_by_id = _load_test_samples(
        response_jsonl=args.response_jsonl,
        source_jsonl=args.source_jsonl,
        scores_dir=args.scores_dir,
    )
    stats = _load_stats(args.stats)
    global_e6 = _load_global_e6_summary(args.e6_json)
    records = _build_score_records(
        scores_dir=args.scores_dir,
        samples_by_id=samples_by_id,
        stats=stats,
    )
    selected = _select_cases(records=records)

    print(f"[e7] selected cases: {', '.join(f'{k}={v.sample_id}' for k, v in selected.items())}")
    print(f"[e7] loading {args.model} on {device}")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    cases_json: list[dict[str, Any]] = []
    traces_dir = args.output_dir / "failure_traces"
    for case_name in CASE_TYPES:
        record = selected[case_name]
        replay = _replay_case(
            model=model,
            tokenizer=tokenizer,
            sample=record.sample,
            device=device,
            max_seq_tokens=args.max_seq_tokens,
        )
        trace_path = traces_dir / f"{record.sample_id}.png"
        _plot_case(record=record, replay=replay, out_path=trace_path)
        cases_json.append(
            _case_json(
                case_type=case_name,
                record=record,
                replay=replay,
                trace_path=trace_path,
            )
        )

    payload = {
        "global_e6": global_e6["localization"],
        "n_cases": len(cases_json),
        "cases": cases_json,
    }
    (args.output_dir / "failures.json").write_text(json.dumps(payload, indent=2))
    (args.output_dir / "failures.md").write_text(_markdown(global_e6=global_e6, cases=cases_json))
    print(f"[e7] wrote {args.output_dir / 'failures.json'}")
    print(f"[e7] wrote {args.output_dir / 'failures.md'}")
    print(f"[e7] wrote {traces_dir}")


if __name__ == "__main__":
    main()
