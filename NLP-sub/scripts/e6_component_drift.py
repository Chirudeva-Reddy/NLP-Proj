"""
E6 — FFN vs attention update-direction drift across the full model depth.

For each held-out RAGTruth test sample we re-run the frozen model, capture the
per-layer `self_attn` and `mlp` sublayer outputs, and compute their per-token
direction-drift (1 - cos between adjacent tokens). Two summaries are produced:

  1. Mean drift table: mean drift for hallucinated and faithful samples, by
     layer range (early / mid / late) and component. Reported as the class
     gap (hallucinated - faithful).
  2. Sample-level separability: each sample is reduced to one scalar per
     (component, layer range) by averaging drift over layers in the range
     and over answer tokens, then AUROC vs has_hallucination is computed.

Localization rule:
  - FFN-localized  if best `mlp` AUROC > best `self_attn` AUROC AND the
    largest positive class gap also occurs in `mlp`.
  - attention-localized if the reverse holds.
  - mixed if AUROC ranking and class-gap ranking disagree.

Outputs:
  outputs/e6/component_drift.csv
  outputs/e6/component_drift.json
  outputs/e6/component_drift.png   (per-layer AUROC curves + range bars)

Runtime is dominated by the full-depth forward pass. On MPS with Qwen 2.5 1.5B
and ~2.6k test samples expect 15-45 minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.component_outputs import (
    capture_component_outputs,
    contiguous_thirds,
    load_model_and_tokenizer,
    pick_device,
    tokenize_sample,
    update_direction_drift,
)
from src.dataset import load as load_samples, split as split_samples
from src.evaluate import auroc


COMPONENTS: tuple[str, ...] = ("self_attn", "mlp")
RANGES_ORDER: tuple[str, ...] = ("early", "mid", "late")


def _component_label(component: str) -> str:
    if component == "mlp":
        return "ffn"
    return component


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--device", default="auto")
    p.add_argument("--response-jsonl", type=Path, default=Path("dataset/ragtruth/response.jsonl"))
    p.add_argument("--source-jsonl", type=Path, default=Path("dataset/ragtruth/source_info.jsonl"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("outputs/artifacts/test"),
                   help="Used only to filter to sample_ids that were kept by the frozen inference run.")
    p.add_argument("--max-seq-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0, help="Cap samples for smoke testing.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/e6"))
    return p.parse_args()


def _frozen_sample_ids(artifacts_dir: Path) -> set[str]:
    return {path.stem for path in artifacts_dir.glob("*.pt")}


def _per_sample_reduction(
    drift: torch.Tensor,
    answer_start: int,
    ranges: dict[str, list[int]],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Return (per_layer_sample_peak, per_range_sample_scalar).
    per_layer_sample_peak: (n_layers,) max drift over answer tokens for this sample.
    per_range_sample_scalar: {range: scalar} max over layers in range and answer tokens.
    
    *Optimized: Uses max pooling instead of mean pooling to isolate the onset spike 
    and prevent signal dilution across the generated sequence.*
    """
    answer_drift = drift[:, answer_start:]
    
    # Extract the peak anomaly across the answer tokens per layer
    per_layer_peak = answer_drift.max(dim=1).values
    
    per_range: dict[str, float] = {}
    for name, layer_ids in ranges.items():
        # Pool the absolute max signal within the specific layer range
        per_range[name] = float(answer_drift[layer_ids].max().item())
        
    return per_layer_peak, per_range


def _plot(
    layer_indices: list[int],
    per_layer_auroc: dict[str, list[float]],
    per_range_auroc: dict[tuple[str, str], float],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    for comp in COMPONENTS:
        ax.plot(layer_indices, per_layer_auroc[comp], marker="o", markersize=3, label=_component_label(comp))
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("per-layer drift AUROC")
    ax.set_title("E6 — per-layer AUROC of update-direction drift")
    ax.legend(fontsize=9)

    bx = axes[1]
    x = np.arange(len(RANGES_ORDER))
    width = 0.38
    attn_vals = [per_range_auroc[("self_attn", r)] for r in RANGES_ORDER]
    mlp_vals = [per_range_auroc[("mlp", r)] for r in RANGES_ORDER]
    bx.bar(x - width / 2, attn_vals, width, label="self_attn")
    bx.bar(x + width / 2, mlp_vals, width, label="ffn")
    bx.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    bx.set_xticks(x)
    bx.set_xticklabels(RANGES_ORDER)
    bx.set_ylabel("range-pooled AUROC")
    bx.set_title("range-pooled")
    bx.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _localization_claim(
    per_range_auroc: dict[tuple[str, str], float],
    per_range_gap: dict[tuple[str, str], float],
) -> dict:
    attn_best = max(per_range_auroc[("self_attn", r)] for r in RANGES_ORDER)
    mlp_best  = max(per_range_auroc[("mlp", r)] for r in RANGES_ORDER)
    attn_gap_best = max(per_range_gap[("self_attn", r)] for r in RANGES_ORDER)
    mlp_gap_best  = max(per_range_gap[("mlp", r)] for r in RANGES_ORDER)

    auroc_winner = "mlp" if mlp_best > attn_best else "self_attn"
    gap_winner = "mlp" if mlp_gap_best > attn_gap_best else "self_attn"

    if auroc_winner == gap_winner:
        claim = "FFN-localized" if auroc_winner == "mlp" else "attention-localized"
    else:
        claim = "mixed"

    return {
        "claim": claim,
        "attn_best_auroc": attn_best,
        "mlp_best_auroc": mlp_best,
        "attn_best_class_gap": attn_gap_best,
        "mlp_best_class_gap": mlp_gap_best,
        "ffn_best_auroc": mlp_best,
        "ffn_best_class_gap": mlp_gap_best,
    }


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frozen_ids = _frozen_sample_ids(args.artifacts_dir)
    print(f"[e6] frozen test artifacts: {len(frozen_ids)}")

    samples = load_samples(response_jsonl=args.response_jsonl, source_jsonl=args.source_jsonl)
    test_samples = split_samples(samples)["test"]
    test_samples = [s for s in test_samples if s.sample_id in frozen_ids]
    if args.limit:
        test_samples = test_samples[: args.limit]
    print(f"[e6] samples to replay: {len(test_samples)}")

    print(f"[e6] loading {args.model} on {device}")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    n_layers = model.config.num_hidden_layers
    ranges = contiguous_thirds(n_layers)
    print(f"[e6] n_layers={n_layers}  ranges={ranges}")

    per_layer_sums: dict[tuple[str, int], float] = {(c, "hal"): 0.0 for c in COMPONENTS}
    per_layer_counts: dict[str, int] = {"hal": 0, "faith": 0}
    per_layer_by_class: dict[tuple[str, str], np.ndarray] = {
        (c, cls): np.zeros(n_layers) for c in COMPONENTS for cls in ("hal", "faith")
    }
    class_n: dict[str, int] = {"hal": 0, "faith": 0}

    per_layer_labels: list[int] = []
    per_layer_scores: dict[str, list[np.ndarray]] = {c: [] for c in COMPONENTS}

    range_labels: list[int] = []
    range_scores: dict[tuple[str, str], list[float]] = {
        (c, r): [] for c in COMPONENTS for r in RANGES_ORDER
    }

    n_skipped = 0
    for sample in tqdm(test_samples, desc="e6", dynamic_ncols=True, file=sys.stdout):
        tok = tokenize_sample(tokenizer, sample, args.max_seq_tokens, device)
        if tok is None:
            n_skipped += 1
            continue

        try:
            captures = capture_component_outputs(model, tok.input_ids)
        except RuntimeError:
            n_skipped += 1
            continue

        label = 1 if sample.spans else 0
        cls = "hal" if label == 1 else "faith"
        class_n[cls] += 1

        per_layer_labels.append(label)
        range_labels.append(label)

        for comp in COMPONENTS:
            drift = update_direction_drift(captures[comp])
            per_layer, per_range = _per_sample_reduction(drift, tok.answer_start, ranges)
            per_layer_np = per_layer.numpy()
            per_layer_scores[comp].append(per_layer_np)
            per_layer_by_class[(comp, cls)] += per_layer_np
            for r in RANGES_ORDER:
                range_scores[(comp, r)].append(per_range[r])

    print(f"[e6] used {sum(class_n.values())} samples (skipped {n_skipped}): "
          f"{class_n['hal']} hallucinated, {class_n['faith']} faithful")

    layer_indices = list(range(n_layers))
    per_layer_auroc: dict[str, list[float]] = {}
    for comp in COMPONENTS:
        stacked = np.stack(per_layer_scores[comp], axis=0)
        per_layer_auroc[comp] = [
            auroc(per_layer_labels, stacked[:, l].tolist()) for l in layer_indices
        ]

    per_range_auroc: dict[tuple[str, str], float] = {}
    per_range_mean: dict[tuple[str, str, str], float] = {}
    per_range_gap: dict[tuple[str, str], float] = {}
    for comp in COMPONENTS:
        for r in RANGES_ORDER:
            scores = range_scores[(comp, r)]
            per_range_auroc[(comp, r)] = auroc(range_labels, scores)
            arr = np.array(scores)
            mean_hal = float(arr[np.array(range_labels) == 1].mean())
            mean_faith = float(arr[np.array(range_labels) == 0].mean())
            per_range_mean[(comp, r, "hal")] = mean_hal
            per_range_mean[(comp, r, "faith")] = mean_faith
            per_range_gap[(comp, r)] = mean_hal - mean_faith

    # Build CSV rows
    rows: list[dict] = []
    for comp in COMPONENTS:
        for r in RANGES_ORDER:
            rows.append({
                "component": _component_label(comp),
                "range": r,
                "auroc": per_range_auroc[(comp, r)],
                "mean_drift_hallucinated": per_range_mean[(comp, r, "hal")],
                "mean_drift_faithful": per_range_mean[(comp, r, "faith")],
                "class_gap": per_range_gap[(comp, r)],
            })

    per_layer_rows: list[dict] = []
    for comp in COMPONENTS:
        for l in layer_indices:
            per_layer_rows.append({
                "component": _component_label(comp),
                "layer": l,
                "auroc": per_layer_auroc[comp][l],
            })

    localization = _localization_claim(per_range_auroc, per_range_gap)

    # Print summary
    print(f"\n[e6] localization claim: {localization['claim']}")
    print(f"     attn best AUROC = {localization['attn_best_auroc']:.4f}  "
          f"(class gap = {localization['attn_best_class_gap']:+.5f})")
    print(f"     ffn  best AUROC = {localization['ffn_best_auroc']:.4f}  "
          f"(class gap = {localization['ffn_best_class_gap']:+.5f})")

    print(f"\n{'component':<10} {'range':<6} {'AUROC':>7}  {'mean_hal':>10}  {'mean_faith':>10}  {'gap':>10}")
    print("-" * 62)
    for row in rows:
        print(f"{row['component']:<10} {row['range']:<6} "
              f"{row['auroc']:>7.4f}  {row['mean_drift_hallucinated']:>10.5f}  "
              f"{row['mean_drift_faithful']:>10.5f}  {row['class_gap']:>+10.5f}")

    # Write CSV (flat)
    with (args.output_dir / "component_drift.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "component", "range", "auroc",
            "mean_drift_hallucinated", "mean_drift_faithful", "class_gap",
        ])
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "component_drift.json").write_text(json.dumps({
        "n_samples_used": sum(class_n.values()),
        "n_hallucinated": class_n["hal"],
        "n_faithful": class_n["faith"],
        "n_layers": n_layers,
        "ranges": ranges,
        "range_table": rows,
        "per_layer_auroc": per_layer_rows,
        "localization": localization,
    }, indent=2))

    _plot(layer_indices, per_layer_auroc, per_range_auroc,
          args.output_dir / "component_drift.png")
    print(f"\n[e6] wrote {args.output_dir}/component_drift.{{csv,json,png}}")


if __name__ == "__main__":
    main()