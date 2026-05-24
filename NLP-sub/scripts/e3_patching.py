"""
E3 — Bidirectional causal intervention via activation patching.

Fix over the prior version: we patch at an ANSWER-REGION position where the
faithful and hallucinated runs genuinely have different activations, instead
of the last prompt token (which is a no-op under causal attention when both
runs share the same prompt).

Patch position
  P = hal.first_hal - 1
    one token before the first hallucinated token in the hallucinated run.
    Guaranteed inside the answer region of both runs (answer_start matches
    across the pair by construction), so donor and recipient activations at P
    are genuinely different.

Directions (recipient-token readout)
  faith_to_hal: recipient = hal, donor = faithful.
    Measure delta logp of hal's natural next token at P+1.
  hal_to_faith: recipient = faithful, donor = hal.
    Measure delta logp of faithful's natural next token at P+1.

Components (4 rubric buckets, aligned with model depth)
  early_attn    — self_attn output in layers [0, 25%)
  mid_ffn       — mlp       output in layers [25%, 75%)
  late_ffn      — mlp       output in layers [75%, 100%)
  copying_heads — self_attn output in layers [75%, 100%)

Bucket CIE per pair = delta_logp from patching the full contiguous bucket
in one intervention run.
Primary test: two-sided Wilcoxon signed-rank per (bucket, direction).
Diagnostic: one-sided ("less") Wilcoxon kept in JSON.

Critical? rule
  Both direction means negative (patched donor suppresses recipient's natural
  continuation) AND both directions reach p_two_sided < 0.05.

Controls
  no_op          — donor == recipient at same position. Should give ~0.
  shuffled_donor — donor from a different pair. Used to bound spurious effects.

Outputs
  outputs/e3/cie_bidirectional.{csv,json,md,png}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import Sample, load, split


DIRECTIONS: tuple[str, ...] = ("faith_to_hal", "hal_to_faith")
BUCKET_ORDER: tuple[str, ...] = ("early_attn", "mid_ffn", "late_ffn", "copying_heads")


@dataclass
class Encoded:
    input_ids: torch.Tensor
    answer_start: int
    first_hal: int | None
    n_seq: int


@dataclass
class UsablePair:
    faith: Encoded
    hal: Encoded
    patch_position: int
    target_position: int
    hal_target_token: int
    faith_target_token: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--min-usable", type=int, default=50)
    p.add_argument("--max-seq-tokens", type=int, default=2048)
    p.add_argument("--response-jsonl", type=Path,
                   default=Path("dataset/ragtruth/response.jsonl"))
    p.add_argument("--source-jsonl", type=Path,
                   default=Path("dataset/ragtruth/source_info.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/e3"))
    p.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Override selected model dtype (default: auto based on device)",
    )
    p.add_argument(
        "--debug-pairs",
        type=int,
        default=0,
        help="Run in debug mode for first N pairs with per-pair diagnostics",
    )
    p.add_argument("--with-controls", action="store_true", default=False)
    p.add_argument("--no-controls", dest="with_controls", action="store_false")
    p.add_argument("--no-op-pairs", type=int, default=0,
                   help="Number of pairs to sanity-check hook idempotency on.")
    p.add_argument(
        "--patch-mode",
        choices=("bucket", "layer_mean"),
        default="bucket",
        help="bucket=patch full contiguous bucket once (fast); "
             "layer_mean=legacy per-layer intervention mean (slow).",
    )
    p.add_argument(
        "--patch-alpha",
        type=float,
        default=1.0,
        help="Interpolation strength for donor activation in [0,1]. "
             "1.0=full replacement, 0.0=no-op.",
    )
    p.add_argument(
        "--critical-min-abs-delta",
        type=float,
        default=0.1,
        help="Practical-effect threshold: both directions must satisfy "
             "|mean delta logp| >= threshold to be marked critical.",
    )
    p.add_argument(
        "--critical-rule",
        choices=("absolute", "relative"),
        default="absolute",
        help="absolute: fixed |mean delta| threshold in both directions; "
             "relative: threshold scales with strongest component effect among statistically valid buckets.",
    )
    p.add_argument(
        "--critical-relative-min-ratio",
        type=float,
        default=0.25,
        help="Only used when --critical-rule relative. "
             "Required min(effect_strength / max_effect_strength) where "
             "effect_strength = min(|CIE f->h|, |CIE h->f|).",
    )
    return p.parse_args()


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # float16 on MPS produces NaN logits for Qwen2.5; bfloat16 is stable.
        return torch.bfloat16
    return torch.float32


def _buckets(n_layers: int) -> dict[str, tuple[str, list[int]]]:
    q1 = max(1, int(round(n_layers * 0.25)))
    q3 = max(q1 + 1, int(round(n_layers * 0.75)))
    return {
        "early_attn":    ("self_attn", list(range(0, q1))),
        "mid_ffn":       ("mlp",       list(range(q1, q3))),
        "late_ffn":      ("mlp",       list(range(q3, n_layers))),
        "copying_heads": ("self_attn", list(range(q3, n_layers))),
    }


def _pair_samples(samples: list[Sample], n_needed: int) -> list[tuple[Sample, Sample]]:
    by_src: dict[str, list[Sample]] = {}
    for s in samples:
        by_src.setdefault(s.source_id, []).append(s)
    pairs: list[tuple[Sample, Sample]] = []
    for _, group in sorted(by_src.items()):
        hal = [s for s in group if s.spans]
        faith = [s for s in group if not s.spans]
        if hal and faith:
            pairs.append((faith[0], hal[0]))
        if len(pairs) >= n_needed:
            break
    return pairs


def _encode(tokenizer, sample: Sample, max_len: int, device: str) -> Encoded | None:
    full = sample.prompt + sample.response
    enc = tokenizer(
        full,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    ans_char_start = len(sample.prompt)
    answer_start = next(
        (i for i, (s, e) in enumerate(offsets) if e > 0 and s >= ans_char_start),
        len(offsets),
    )
    if answer_start >= len(offsets) - 1:
        return None
    if len(offsets) >= max_len:
        return None

    first_hal: int | None = None
    if sample.spans:
        for i in range(answer_start, len(offsets)):
            cs, ce = offsets[i]
            rs = cs - ans_char_start
            re_ = ce - ans_char_start
            if any(rs < sp.end and re_ > sp.start for sp in sample.spans):
                first_hal = i
                break
        if first_hal is None or first_hal <= answer_start:
            return None

    return Encoded(
        input_ids=enc["input_ids"].to(device),
        answer_start=answer_start,
        first_hal=first_hal,
        n_seq=len(offsets),
    )


def _capture_one_position_and_logp(
    model,
    input_ids: torch.Tensor,
    position: int,
    n_layers: int,
    target_position: int,
    target_token: int,
) -> tuple[dict[tuple[int, str], torch.Tensor], float]:
    caps: dict[tuple[int, str], torch.Tensor] = {}
    blocks = model.model.layers
    handles = []

    def make_hook(idx: int, comp: str):
        def hook(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            caps[(idx, comp)] = t[:, position, :].detach().clone()
        return hook

    for l in range(n_layers):
        handles.append(blocks[l].self_attn.register_forward_hook(make_hook(l, "self_attn")))
        handles.append(blocks[l].mlp.register_forward_hook(make_hook(l, "mlp")))
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids)
        # Defensive checks: ensure readout index in range
        seq_len = out.logits.shape[1]
        if not (0 <= (target_position - 1) < seq_len):
            return caps, float("nan")

        logits = out.logits[0, target_position - 1]
        # sanitize logits in case of NaN/Inf (can happen on some backends/dtypes)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        logp_val = F.log_softmax(logits.float(), dim=-1)[target_token]
        logp = float(logp_val.item())
    finally:
        for h in handles:
            h.remove()
    return caps, logp


def _patched_logp_bucket(
    model,
    input_ids: torch.Tensor,
    patch_keys: list[tuple[int, str]],
    donor_caps: dict[tuple[int, str], torch.Tensor],
    patch_position: int,
    target_position: int,
    target_token: int,
    patch_alpha: float,
) -> float:
    blocks = model.model.layers
    handles = []

    def make_hook(replacement: torch.Tensor):
        def hook(_m, _i, output):
            if isinstance(output, tuple):
                t = output[0].clone()
                donor = replacement.to(t.dtype)
                orig = t[:, patch_position, :]
                t[:, patch_position, :] = (1.0 - patch_alpha) * orig + patch_alpha * donor
                return (t,) + output[1:]
            t = output.clone()
            donor = replacement.to(t.dtype)
            orig = t[:, patch_position, :]
            t[:, patch_position, :] = (1.0 - patch_alpha) * orig + patch_alpha * donor
            return t

        return hook

    for layer_idx, component in patch_keys:
        key = (layer_idx, component)
        if key not in donor_caps:
            # missing donor activation for this layer/component -> abort this patched run
            for h in handles:
                h.remove()
            return float("nan")
        block = blocks[layer_idx]
        target_mod = block.self_attn if component == "self_attn" else block.mlp
        replacement = donor_caps[key]
        handles.append(target_mod.register_forward_hook(make_hook(replacement)))

    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids)
        seq_len = out.logits.shape[1]
        if not (0 <= (target_position - 1) < seq_len):
            return float("nan")
        logits = out.logits[0, target_position - 1]
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        return float(F.log_softmax(logits.float(), dim=-1)[target_token].item())
    finally:
        for handle in handles:
            handle.remove()


def _bucket_cie(
    model, buckets: dict[str, tuple[str, list[int]]],
    recipient_ids: torch.Tensor, donor_caps: dict[tuple[int, str], torch.Tensor],
    patch_position: int, target_position: int, target_token: int,
    logp_clean: float,
    patch_mode: str,
    patch_alpha: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (component, layers) in buckets.items():
        if patch_mode == "bucket":
            patch_keys = [(l, component) for l in layers]
            lp = _patched_logp_bucket(
                model=model,
                input_ids=recipient_ids,
                patch_keys=patch_keys,
                donor_caps=donor_caps,
                patch_position=patch_position,
                target_position=target_position,
                target_token=target_token,
                patch_alpha=patch_alpha,
            )
            out[name] = float(lp - logp_clean)
            continue

        # Legacy mode: mean over per-layer interventions in the bucket.
        if patch_mode != "layer_mean":
            raise ValueError(f"Unknown patch_mode={patch_mode}")
        deltas: list[float] = []
        for layer_idx in layers:
            lp = _patched_logp_bucket(
                model=model,
                input_ids=recipient_ids,
                patch_keys=[(layer_idx, component)],
                donor_caps=donor_caps,
                patch_position=patch_position,
                target_position=target_position,
                target_token=target_token,
                patch_alpha=patch_alpha,
            )
            deltas.append(float(lp - logp_clean))
        out[name] = float(np.mean(deltas))
    return out


def _wilcoxon(values: list[float], alternative: str) -> float:
    if len(values) < 5:
        return float("nan")
    arr = np.array(values)
    if np.allclose(arr, 0):
        return 1.0
    try:
        return float(wilcoxon(arr, alternative=alternative).pvalue)
    except ValueError:
        return float("nan")


def _plot_heatmap(
    means: dict[tuple[str, str], float],
    p_two: dict[tuple[str, str], float],
    out_path: Path,
) -> None:
    grid = np.zeros((len(BUCKET_ORDER), len(DIRECTIONS)))
    for i, b in enumerate(BUCKET_ORDER):
        for j, d in enumerate(DIRECTIONS):
            grid[i, j] = means.get((b, d), np.nan)
    vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(DIRECTIONS)))
    ax.set_xticklabels(["faith → hal", "hal → faith"])
    ax.set_yticks(range(len(BUCKET_ORDER)))
    ax.set_yticklabels(BUCKET_ORDER)
    for i in range(len(BUCKET_ORDER)):
        for j in range(len(DIRECTIONS)):
            p = p_two.get((BUCKET_ORDER[i], DIRECTIONS[j]), float("nan"))
            marker = "*" if p == p and p < 0.05 else ""
            ax.text(j, i, f"{grid[i, j]:+.3f}{marker}",
                    ha="center", va="center", color="black", fontsize=10)
    ax.set_title("E3 - bidirectional CIE (mean Δlog p)")
    fig.colorbar(im, ax=ax, label="Δlog p")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _encode_all(
    tokenizer, pairs: list[tuple[Sample, Sample]], max_len: int, device: str,
) -> list[UsablePair]:
    usable: list[UsablePair] = []
    skipped = {"encode": 0, "answer_mismatch": 0, "no_span": 0, "position": 0}
    for faith, hal in tqdm(pairs, desc="encoding", dynamic_ncols=True):
        fe = _encode(tokenizer, faith, max_len, device)
        he = _encode(tokenizer, hal, max_len, device)
        if fe is None or he is None:
            skipped["encode"] += 1
            continue
        if fe.answer_start != he.answer_start:
            skipped["answer_mismatch"] += 1
            continue
        if he.first_hal is None:
            skipped["no_span"] += 1
            continue
        P = he.first_hal - 1
        if P < he.answer_start or P < fe.answer_start:
            skipped["position"] += 1
            continue
        if P + 1 >= fe.n_seq or P + 1 >= he.n_seq:
            skipped["position"] += 1
            continue
        usable.append(UsablePair(
            faith=fe,
            hal=he,
            patch_position=P,
            target_position=P + 1,
            hal_target_token=int(he.input_ids[0, P + 1].item()),
            faith_target_token=int(fe.input_ids[0, P + 1].item()),
        ))
    print(f"[e3] encoded usable={len(usable)}  skipped={skipped}")
    return usable


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.patch_alpha <= 1.0):
        raise SystemExit("--patch-alpha must be in [0, 1].")
    if args.critical_min_abs_delta < 0.0:
        raise SystemExit("--critical-min-abs-delta must be >= 0.")
    if not (0.0 <= args.critical_relative_min_ratio <= 1.0):
        raise SystemExit("--critical-relative-min-ratio must be in [0, 1].")

    device = _pick_device(args.device)
    # Allow explicit dtype override via CLI; otherwise pick default based on device
    if getattr(args, "dtype", "auto") != "auto":
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        model_dtype = dtype_map[args.dtype]
    else:
        model_dtype = _pick_dtype(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[e3] loading {args.model} on {device} dtype={model_dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=model_dtype,
        attn_implementation="eager",
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    buckets = _buckets(n_layers)
    print(f"[e3] n_layers={n_layers}")
    for name, (comp, layers) in buckets.items():
        print(f"  {name:<15} {comp:<10} layers {layers[0]}..{layers[-1]} ({len(layers)})")

    samples = load(args.response_jsonl, args.source_jsonl)
    test_samples = split(samples)["test"]
    candidates = _pair_samples(test_samples, args.n_pairs * 3)
    print(f"[e3] candidate pairs: {len(candidates)}")

    usable = _encode_all(tokenizer, candidates, args.max_seq_tokens, device)
    if len(usable) < args.min_usable:
        raise SystemExit(f"Only {len(usable)} usable pairs (< min {args.min_usable}).")
    usable = usable[: args.n_pairs]
    print(
        f"[e3] running {len(usable)} pairs x 2 directions, "
        f"with_controls={args.with_controls}, patch_mode={args.patch_mode}, "
        f"patch_alpha={args.patch_alpha:.3f}, "
        f"critical_min_abs_delta={args.critical_min_abs_delta:.3f}"
    )

    deltas: dict[tuple[str, str], list[float]] = {
        (b, d): [] for b in BUCKET_ORDER for d in DIRECTIONS
    }
    shuf_deltas: dict[tuple[str, str], list[float]] = {
        (b, d): [] for b in BUCKET_ORDER for d in DIRECTIONS
    }
    no_op_deltas: dict[str, list[float]] = {b: [] for b in BUCKET_ORDER}

    # diagnostics and skip counters
    processed_pairs = 0
    used_pairs = 0
    skipped_logp_nan = 0
    skipped_incomplete_caps = 0
    skipped_missing_keys = 0
    expected_keys: list[tuple[int, str]] = []
    for l in range(n_layers):
        expected_keys.append((l, "self_attn"))
        expected_keys.append((l, "mlp"))

    for idx, up in enumerate(tqdm(usable, desc="pairs", dynamic_ncols=True, file=sys.stdout)):
        processed_pairs += 1
        # Causal-attention truncation: logits at position target_position - 1
        # depend only on tokens 0..target_position-1. Trimming the suffix is
        # mathematically identical and cuts forward-pass cost substantially
        # (attention is O(seq^2)).
        faith_ids = up.faith.input_ids[:, : up.target_position]
        hal_ids   = up.hal.input_ids[:,   : up.target_position]
        faith_caps, logp_faith_clean = _capture_one_position_and_logp(
            model=model,
            input_ids=faith_ids,
            position=up.patch_position,
            n_layers=n_layers,
            target_position=up.target_position,
            target_token=up.faith_target_token,
        )
        hal_caps, logp_hal_clean = _capture_one_position_and_logp(
            model=model,
            input_ids=hal_ids,
            position=up.patch_position,
            n_layers=n_layers,
            target_position=up.target_position,
            target_token=up.hal_target_token,
        )

        # debug prints for initial pairs if requested
        if args.debug_pairs and idx < args.debug_pairs:
            print(f"[debug] pair={idx} patch_pos={up.patch_position} target_pos={up.target_position}")
            print(f"  faith input len={up.faith.n_seq} hal input len={up.hal.n_seq}")
            print(f"  logp_faith_clean={logp_faith_clean} logp_hal_clean={logp_hal_clean}")

        # skip pairs with NaN/Inf clean log-probabilities
        if not np.isfinite(logp_faith_clean) or not np.isfinite(logp_hal_clean):
            skipped_logp_nan += 1
            if args.debug_pairs and idx < args.debug_pairs:
                print("  skipping pair: NaN/Inf clean logp")
            continue

        # ensure captured activations exist for all expected keys
        missing_faith = [k for k in expected_keys if k not in faith_caps]
        missing_hal = [k for k in expected_keys if k not in hal_caps]
        if missing_faith or missing_hal:
            skipped_incomplete_caps += 1
            if args.debug_pairs and idx < args.debug_pairs:
                print(f"  skipping pair: incomplete caps; missing_faith={missing_faith[:4]} missing_hal={missing_hal[:4]}")
            continue

        cie_fh = _bucket_cie(
            model, buckets,
            recipient_ids=hal_ids, donor_caps=faith_caps,
            patch_position=up.patch_position, target_position=up.target_position,
            target_token=up.hal_target_token, logp_clean=logp_hal_clean,
            patch_mode=args.patch_mode,
            patch_alpha=args.patch_alpha,
        )
        cie_hf = _bucket_cie(
            model, buckets,
            recipient_ids=faith_ids, donor_caps=hal_caps,
            patch_position=up.patch_position, target_position=up.target_position,
            target_token=up.faith_target_token, logp_clean=logp_faith_clean,
            patch_mode=args.patch_mode,
            patch_alpha=args.patch_alpha,
        )

        # if patched result produced NaNs (missing donor keys or other), skip
        if any(not np.isfinite(v) for v in cie_fh.values()) or any(not np.isfinite(v) for v in cie_hf.values()):
            skipped_missing_keys += 1
            if args.debug_pairs and idx < args.debug_pairs:
                print(f"  skipping pair: patched result contains NaN; cie_fh={cie_fh} cie_hf={cie_hf}")
            continue

        for b in BUCKET_ORDER:
            deltas[(b, "faith_to_hal")].append(cie_fh[b])
            deltas[(b, "hal_to_faith")].append(cie_hf[b])

        used_pairs += 1

        if args.with_controls:
            j = (idx + 1) % len(usable)
            other = usable[j]
            other_faith_ok = up.patch_position < other.faith.n_seq
            other_hal_ok = up.patch_position < other.hal.n_seq
            if other_faith_ok:
                other_faith_caps, _ = _capture_one_position_and_logp(
                    model=model,
                    input_ids=other.faith.input_ids[:, : up.target_position],
                    position=up.patch_position,
                    n_layers=n_layers,
                    target_position=up.target_position,
                    target_token=up.hal_target_token,
                )
                if other_faith_caps:
                    shuf_fh = _bucket_cie(
                        model, buckets,
                        recipient_ids=hal_ids, donor_caps=other_faith_caps,
                        patch_position=up.patch_position, target_position=up.target_position,
                        target_token=up.hal_target_token, logp_clean=logp_hal_clean,
                        patch_mode=args.patch_mode,
                        patch_alpha=args.patch_alpha,
                    )
                    for b in BUCKET_ORDER:
                        shuf_deltas[(b, "faith_to_hal")].append(shuf_fh[b])
            if other_hal_ok:
                other_hal_caps, _ = _capture_one_position_and_logp(
                    model=model,
                    input_ids=other.hal.input_ids[:, : up.target_position],
                    position=up.patch_position,
                    n_layers=n_layers,
                    target_position=up.target_position,
                    target_token=up.faith_target_token,
                )
                if other_hal_caps:
                    shuf_hf = _bucket_cie(
                        model, buckets,
                        recipient_ids=faith_ids, donor_caps=other_hal_caps,
                        patch_position=up.patch_position, target_position=up.target_position,
                        target_token=up.faith_target_token, logp_clean=logp_faith_clean,
                        patch_mode=args.patch_mode,
                        patch_alpha=args.patch_alpha,
                    )
                    for b in BUCKET_ORDER:
                        shuf_deltas[(b, "hal_to_faith")].append(shuf_hf[b])

        if args.no_op_pairs > 0 and idx < args.no_op_pairs:
            no_op = _bucket_cie(
                model, buckets,
                recipient_ids=hal_ids, donor_caps=hal_caps,
                patch_position=up.patch_position, target_position=up.target_position,
                target_token=up.hal_target_token, logp_clean=logp_hal_clean,
                patch_mode=args.patch_mode,
                patch_alpha=args.patch_alpha,
            )
            for b in BUCKET_ORDER:
                no_op_deltas[b].append(no_op[b])

    print("\n[e3] raw delta logp samples (first 5 per cell):")
    for d in DIRECTIONS:
        for b in BUCKET_ORDER:
            vals = deltas[(b, d)][:5]
            print(f"  {d:<13} {b:<15} {[f'{v:+.4f}' for v in vals]}")

    print("\n[e3] no_op control (donor == recipient, expected ~ 0):")
    for b in BUCKET_ORDER:
        arr = np.array(no_op_deltas[b]) if no_op_deltas[b] else np.array([])
        if arr.size > 0:
            print(f"  {b:<15} mean={arr.mean():+.6f}  max|.|={np.abs(arr).max():.6f}")

    # summary of processed/used/skipped pairs
    print()
    print(f"[e3] processed_pairs={processed_pairs} used_pairs={used_pairs} skipped_logp_nan={skipped_logp_nan} skipped_incomplete_caps={skipped_incomplete_caps} skipped_missing_keys={skipped_missing_keys}")

    means: dict[tuple[str, str], float] = {}
    p_two: dict[tuple[str, str], float] = {}
    p_less: dict[tuple[str, str], float] = {}
    shuf_means: dict[tuple[str, str], float] = {}
    rows: list[dict] = []
    for direction in DIRECTIONS:
        for b in BUCKET_ORDER:
            vals = deltas[(b, direction)]
            mean = float(np.mean(vals)) if vals else float("nan")
            median = float(np.median(vals)) if vals else float("nan")
            p2 = _wilcoxon(vals, "two-sided")
            pl = _wilcoxon(vals, "less")
            means[(b, direction)] = mean
            p_two[(b, direction)] = p2
            p_less[(b, direction)] = pl
            shuf_vals = shuf_deltas[(b, direction)]
            shuf_mean = float(np.mean(shuf_vals)) if shuf_vals else float("nan")
            shuf_means[(b, direction)] = shuf_mean
            rows.append({
                "bucket": b, "direction": direction, "n": len(vals),
                "mean_delta_logp": mean,
                "median_delta_logp": median,
                "p_two_sided": p2, "p_less": pl,
                "shuffled_donor_mean": shuf_mean,
            })

    critical: dict[str, bool] = {}
    effect_strength: dict[str, float] = {}
    eligible: dict[str, bool] = {}
    for b in BUCKET_ORDER:
        fh_mean = means[(b, "faith_to_hal")]
        hf_mean = means[(b, "hal_to_faith")]
        both_neg = (fh_mean < 0) and (hf_mean < 0)
        both_sig = (p_two[(b, "faith_to_hal")] < 0.05) and (p_two[(b, "hal_to_faith")] < 0.05)
        eligible[b] = both_neg and both_sig
        effect_strength[b] = float(min(abs(fh_mean), abs(hf_mean)))

    max_eligible_effect = max((effect_strength[b] for b in BUCKET_ORDER if eligible[b]), default=0.0)
    for b in BUCKET_ORDER:
        if not eligible[b]:
            critical[b] = False
            continue

        if args.critical_rule == "absolute":
            critical[b] = effect_strength[b] >= args.critical_min_abs_delta
            continue

        relative_floor = args.critical_relative_min_ratio * max_eligible_effect
        threshold = max(args.critical_min_abs_delta, relative_floor)
        critical[b] = effect_strength[b] >= threshold

    print("\n=== E3 bidirectional CIE table ===")
    print(f"{'bucket':<15} {'CIE f->h':>10} {'p(two)':>10} {'CIE h->f':>10} {'p(two)':>10} {'Critical':>10}")
    print("-" * 70)
    for b in BUCKET_ORDER:
        fh = means[(b, "faith_to_hal")]
        hf = means[(b, "hal_to_faith")]
        pfh = p_two[(b, "faith_to_hal")]
        phf = p_two[(b, "hal_to_faith")]
        crit = "YES" if critical[b] else ""
        print(f"{b:<15} {fh:>+10.4f} {pfh:>10.3e} {hf:>+10.4f} {phf:>10.3e} {crit:>10}")

    with (args.output_dir / "cie_bidirectional.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "bucket", "direction", "n", "mean_delta_logp",
            "median_delta_logp", "p_two_sided", "p_less", "shuffled_donor_mean",
        ])
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "cie_bidirectional.json").write_text(json.dumps({
        "n_pairs_used": used_pairs,
        "n_layers": n_layers,
        "patch_mode": args.patch_mode,
        "patch_alpha": args.patch_alpha,
        "critical_min_abs_delta": args.critical_min_abs_delta,
        "critical_rule": args.critical_rule,
        "critical_relative_min_ratio": args.critical_relative_min_ratio,
        "critical_effect_strength": effect_strength,
        "critical_max_eligible_effect": max_eligible_effect,
        "buckets": {
            name: {"component": c, "layers": layers}
            for name, (c, layers) in buckets.items()
        },
        "rows": rows,
        "critical": critical,
        "no_op_control": {
            b: ({
                "mean": float(np.array(no_op_deltas[b]).mean()),
                "max_abs": float(np.abs(np.array(no_op_deltas[b])).max()),
                "n": len(no_op_deltas[b]),
            } if no_op_deltas[b] else None)
            for b in BUCKET_ORDER
        },
    }, indent=2))

    md = [
        "# E3 - Bidirectional Causal Intervention (Activation Patching)",
        "",
        f"N pairs used: **{used_pairs}**  |  n_layers: **{n_layers}**",
        f"Patch mode: **{args.patch_mode}**",
        f"Patch alpha: **{args.patch_alpha:.3f}**",
        f"Critical rule: **{args.critical_rule}**",
        f"Critical min |mean Δlogp|: **{args.critical_min_abs_delta:.3f}**",
        "",
        "Patch position: one token before the first hallucinated token (answer-region).",
        "",
        "## Rubric table",
        "",
        "| Component | CIE faith->hal | CIE hal->faith | Critical? |",
        "|-----------|----------------|----------------|-----------|",
    ]
    for b in BUCKET_ORDER:
        fh = means[(b, "faith_to_hal")]
        hf = means[(b, "hal_to_faith")]
        crit = "yes" if critical[b] else ""
        md.append(f"| {b} | {fh:+.4f} | {hf:+.4f} | {crit} |")

    md += [
        "",
        "## Significance details (two-sided Wilcoxon signed-rank)",
        "",
        "| Component | Direction | n | mean delta logp | median delta logp | p (two-sided) | p (less) | ",
        "|-----------|-----------|---|-----------------|-------------------|---------------|----------|",
    ]
    for row in rows:
        md.append(
            f"| {row['bucket']} | {row['direction']} | {row['n']} | "
            f"{row['mean_delta_logp']:+.4f} | {row['median_delta_logp']:+.4f} | {row['p_two_sided']:.3e} | "
            f"{row['p_less']:.3e} | {row['shuffled_donor_mean']:+.4f} |"
        )

    md += ["", "## Controls", "", "**no_op** (donor == recipient, expected ~ 0):"]
    for b in BUCKET_ORDER:
        arr = np.array(no_op_deltas[b]) if no_op_deltas[b] else np.array([])
        if arr.size > 0:
            md.append(f"- {b}: mean={arr.mean():+.6f}, max|.|={np.abs(arr).max():.6f}")
    md += ["", "**shuffled_donor** means are shown in the significance table above."]

    md += [
        "",
        "## Interpretation and Mechanism",
        "",
        "### How to read these results",
        "",
        "- `mean_delta_logp` is the average change in log probability at the readout token when donor activations are patched into the recipient. Negative values indicate that donor activations suppress the recipient's natural continuation (i.e., the donor reduces probability of the original token).",
        "- We test two directions: faithful→hallucinated (donor=faithful into hallucinated recipient) and hallucinated→faithful (donor=hallucinated into faithful recipient). A component is plausible causal evidence when both directions produce negative mean and are statistically significant.",
        "",
        "### Mechanistic interpretation if mid/late FFN are critical",
        "",
        "The late (and mid) FFN layers are position-wise nonlinear transformations applied after attention has integrated cross-token information. If replacing these FFN outputs reliably reduces the probability of the original token (large negative Δlogp), this indicates those FFN activations contain features that downstream computation uses to produce the hallucinated token. In short: FFNs can encode generation-critical features, and patching substitutes those features, changing the final logits.",
        "",
        "Key points that support this mechanism:",
        "",
        "- FFNs are per-position, high-dimensional transforms that can store pattern embeddings which later influence the output layer.",
        "- Late FFNs act after extensive attention integration, so they contain context-aware, generation-ready features rather than raw token signals.",
        "- A consistent two-direction effect (both negative mean and significant p-values) argues for a causal role rather than simple correlation.",
        "",
        "### If results disagree with ReDeEP (no effect in FFNs)",
        "",
        "Possible explanations and checks:",
        "",
        "- Insufficient sample size: increase `--n-pairs` and verify `used_pairs` equals requested pairs.",
        "- Partial interpolation (`--patch-alpha` < 1.0) reduces effect size; try `--patch-alpha 1.0` to test full replacement.",
        "- Dataset or tokenization differences: ensure donor and recipient readout tokens are aligned and both inside the answer region; verify the `no_op` control is near zero.",
        "- Numeric instability (MPS float16): rerun with `--dtype float32` or on CUDA if available.",
        "- Effect localized to specific heads rather than pooled FFN outputs: run per-layer or per-head interventions (use `--patch-mode layer_mean` or extend the script to patch individual attention heads).",
        "",
        "### Suggested follow-ups to strengthen claims",
        "",
        "- Report effect-size distributions (e.g., bootstrap CIs) and plot per-pair Δlogp, not only means.",
        "- Perform per-layer ablation (patch each layer individually) to localize the signal within the 'late' range.",
        "- Try per-head patching for self-attention modules if copying behavior is suspected.",
        "- Confirm `no_op` and `shuffled_donor` controls are near zero to rule out artifacts.",
        "",
        "### Example report language you can copy into your writeup",
        "",
        "- If results agree: \"We find that mid and late FFN layers are the critical components: patching these layers produces a consistent negative Δlogp in both directions (mean ≈ X, p < 0.05), indicating these layers causally contribute to producing hallucinated tokens. This supports ReDeEP's mechanism that FFN transformations in late layers encode generation-critical features.\"",
        "- If results disagree: \"We did not observe a consistent causal effect in FFN layers. Possible reasons include insufficient sample size, interpolation strength, or that the causal signal is carried in specific attention heads rather than pooled FFN outputs. We recommend per-layer/per-head follow-ups and rerunning with `--patch-alpha 1.0` and `--dtype float32`.\"",
        "",
        "Caveat: CIE is an intervention-based test and depends on exact patch position, token alignment, and model internals. Interpret results together with controls and follow-up probes.",
    ]

    (args.output_dir / "cie_bidirectional.md").write_text("\n".join(md))

    _plot_heatmap(means, p_two, args.output_dir / "cie_bidirectional.png")
    print(f"\n[e3] wrote {args.output_dir}/cie_bidirectional.{{csv,json,md,png}}")


if __name__ == "__main__":
    main()
