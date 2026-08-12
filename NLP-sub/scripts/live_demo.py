"""
Live demo wrapper for single-sample hidden-state extraction and token scoring.

Supports two runtime modes:
  - score_provided: evaluator supplies retrieved context + candidate passage
  - generate_then_score: evaluator supplies only retrieved context

Supports two frozen scoring profiles:
  - local
  - ssd_last18
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import Sample
from src.inference import InferenceRunner
from src.scoring import (
    aggregate_token_scores,
    load_frozen_scoring_config,
    score_artifact,
)


LOCAL_MODEL = "Qwen/Qwen2.5-1.5B"
LOCAL_LAYERS = "last8"
LOCAL_STATS = Path("outputs/stats.pt")

SSD_LAST18_MODEL = "Qwen/Qwen2.5-1.5B"
SSD_LAST18_LAYERS = "last18"
SSD_LAST18_STATS = Path("/Volumes/My Passport/NLP/outputs/stats_qwen15b_instruct_last18.pt")

REP_METRICS = ["cosine_drift", "mahalanobis", "logit_lens", "pca_deviation", "cie_top3"]
BASELINE_METRICS = ["attention_entropy", "logit_confidence"]
CONSENSUS_METRICS = ["mahalanobis", "logit_lens", "pca_deviation"]
DEFAULT_INSTRUCTION = "Use only the retrieved context to produce the answer."


@dataclass(frozen=True)
class DemoProfile:
    name: str
    model_name: str
    layers_spec: str
    stats_path: Path


@dataclass(frozen=True)
class DemoInput:
    instruction: str
    context: str
    passage: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=("local", "ssd_last18"))
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--context-file", type=Path, default=None)
    parser.add_argument("--passage-file", type=Path, default=None)
    parser.add_argument("--instruction-file", type=Path, default=None)
    parser.add_argument("--instruction-text", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--show-baselines", action="store_true", default=False)
    parser.add_argument("--show-aggregates", action="store_true", default=False)
    parser.add_argument("--aggregate", choices=("max", "mean", "both"), default="both")
    return parser.parse_args()


def _resolve_profile(args: argparse.Namespace) -> DemoProfile:
    if args.profile == "ssd_last18":
        if args.model is not None and args.model != SSD_LAST18_MODEL:
            raise ValueError(
                f"ssd_last18 profile is pinned to model {SSD_LAST18_MODEL}; got override {args.model}."
            )
        if args.layers is not None and args.layers != SSD_LAST18_LAYERS:
            raise ValueError(
                f"ssd_last18 profile is pinned to layers {SSD_LAST18_LAYERS}; got override {args.layers}."
            )
        mount_path = Path("/Volumes/My Passport")
        if not mount_path.exists():
            raise FileNotFoundError("External SSD /Volumes/My Passport is not mounted.")
        if not SSD_LAST18_STATS.exists():
            raise FileNotFoundError(f"Frozen SSD stats not found: {SSD_LAST18_STATS}")
        return DemoProfile(
            name="ssd_last18",
            model_name=SSD_LAST18_MODEL,
            layers_spec=SSD_LAST18_LAYERS,
            stats_path=SSD_LAST18_STATS,
        )

    model_name = args.model or LOCAL_MODEL
    layers_spec = args.layers or LOCAL_LAYERS
    return DemoProfile(
        name="local",
        model_name=model_name,
        layers_spec=layers_spec,
        stats_path=LOCAL_STATS,
    )


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding="utf-8")


def _validate_input_args(args: argparse.Namespace) -> None:
    if args.input_file is None and args.context_file is None:
        raise ValueError("Provide either --input-file or --context-file.")
    if args.input_file is not None and any(
        value is not None
        for value in (args.context_file, args.passage_file, args.instruction_file, args.instruction_text)
    ):
        raise ValueError(
            "--input-file cannot be combined with --context-file, --passage-file, "
            "--instruction-file, or --instruction-text."
        )


def _build_instruction(args: argparse.Namespace) -> str:
    if args.instruction_text:
        return args.instruction_text.strip()
    if args.instruction_file:
        return _read_text(args.instruction_file).strip()
    return DEFAULT_INSTRUCTION


def _read_demo_input(path: Path) -> DemoInput:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    raw_payload = path.read_text(encoding="utf-8")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise TypeError(f"Demo input file must contain a JSON object: {path}")

    context_value = payload.get("context")
    if not isinstance(context_value, str) or not context_value.strip():
        raise ValueError(f"Demo input file must include a non-empty string field 'context': {path}")

    instruction_value = payload.get("instruction", DEFAULT_INSTRUCTION)
    if not isinstance(instruction_value, str) or not instruction_value.strip():
        raise ValueError(f"Demo input field 'instruction' must be a non-empty string: {path}")

    passage_value = payload.get("passage")
    if passage_value is None:
        normalized_passage: str | None = None
    elif isinstance(passage_value, str):
        normalized_passage = passage_value.rstrip()
        if normalized_passage == "":
            normalized_passage = None
    else:
        raise ValueError(f"Demo input field 'passage' must be a string if present: {path}")

    return DemoInput(
        instruction=instruction_value.strip(),
        context=context_value,
        passage=normalized_passage,
    )


def _resolve_demo_input(args: argparse.Namespace) -> DemoInput:
    if args.input_file is not None:
        return _read_demo_input(args.input_file)

    instruction = _build_instruction(args)
    if args.context_file is None:
        raise ValueError("Context file is required when --input-file is not used.")
    context = _read_text(args.context_file)
    passage = _read_text(args.passage_file).rstrip() if args.passage_file is not None else None
    if passage == "":
        raise ValueError("Provided passage is empty.")
    return DemoInput(
        instruction=instruction,
        context=context,
        passage=passage,
    )


def _build_prompt(instruction: str, context: str) -> str:
    return (
        f"{instruction}\n\n"
        "Retrieved context:\n"
        f"{context.strip()}\n\n"
        "Answer:"
    )


def _display_token(text: str) -> str:
    rendered = (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace(" ", "<sp>")
    )
    if rendered == "":
        return "<empty>"
    if rendered == "<sp>":
        return "<space>"
    return rendered


def _answer_token_texts(runner: InferenceRunner, prompt: str, response: str, expected_tokens: int) -> list[str]:
    encoded = runner.tokenizer(
        prompt + response,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=runner.max_seq_tokens,
        add_special_tokens=True,
    )
    offsets = encoded["offset_mapping"][0].tolist()
    answer_char_start = len(prompt)
    answer_start = next(
        (i for i, (start, end) in enumerate(offsets) if end > 0 and start >= answer_char_start),
        len(offsets),
    )
    if answer_start >= len(offsets):
        raise ValueError("Prompt/response tokenization truncated before the answer region.")

    answer_ids = encoded["input_ids"][0, answer_start:].tolist()
    token_texts = [
        _display_token(
            runner.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        for token_id in answer_ids
    ]
    if len(token_texts) != expected_tokens:
        raise ValueError(
            f"Token text count {len(token_texts)} does not match score length {expected_tokens}."
        )
    return token_texts


def _aggregate_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["max", "mean"]
    return [mode]


def _clip_cell(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _build_table_lines(headers: list[str], rows: list[list[str]], left_align_columns: set[int]) -> list[str]:
    widths = [
        max(len(header), max((len(row[column_index]) for row in rows), default=0))
        for column_index, header in enumerate(headers)
    ]
    border = "-+-".join("-" * width for width in widths)

    def _format_row(values: list[str]) -> str:
        formatted_cells: list[str] = []
        for column_index, value in enumerate(values):
            width = widths[column_index]
            if column_index in left_align_columns:
                formatted_cells.append(value.ljust(width))
            else:
                formatted_cells.append(value.rjust(width))
        return " | ".join(formatted_cells)

    lines = [_format_row(headers), border]
    lines.extend(_format_row(row) for row in rows)
    return lines


def _print_token_table(score_payload: dict, token_texts: list[str], include_baselines: bool) -> None:
    metric_names = list(REP_METRICS)
    if include_baselines:
        metric_names.extend(BASELINE_METRICS)

    headers = ["idx", "token"] + metric_names
    rows: list[list[str]] = []
    for index, token_text in enumerate(token_texts):
        token_cell = _clip_cell(token_text, 24)
        value_cells = [f"{float(score_payload[name][index].item()):.4f}" for name in metric_names]
        rows.append([str(index), token_cell, *value_cells])

    print("\nToken-level scores")
    for line in _build_table_lines(headers=headers, rows=rows, left_align_columns={1}):
        print(line)


def _descending_fractional_ranks(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Expected 1-D tensor for ranking, got shape {tuple(values.shape)}.")
    n_tokens = int(values.shape[0])
    if n_tokens == 0:
        raise ValueError("Cannot rank an empty token tensor.")
    if n_tokens == 1:
        return torch.ones(1, dtype=torch.float32)

    order = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty(n_tokens, dtype=torch.float32)
    for rank_index, token_index in enumerate(order.tolist()):
        ranks[token_index] = float(n_tokens - 1 - rank_index) / float(n_tokens - 1)
    return ranks


def _consensus_rankings(score_payload: dict, token_texts: list[str]) -> list[dict[str, float | int | str]]:
    missing_metrics = [name for name in CONSENSUS_METRICS if name not in score_payload]
    if missing_metrics:
        raise KeyError(f"Consensus ranking requires metrics missing from score payload: {missing_metrics}")

    rank_vectors = [_descending_fractional_ranks(score_payload[name].float()) for name in CONSENSUS_METRICS]
    stacked_ranks = torch.stack(rank_vectors, dim=0)
    consensus = stacked_ranks.mean(dim=0)
    sort_order = torch.argsort(consensus, descending=True, stable=True).tolist()

    ranked: list[dict[str, float | int | str]] = []
    for token_index in sort_order:
        ranked.append(
            {
                "idx": token_index,
                "token": token_texts[token_index],
                "consensus": float(consensus[token_index].item()),
                "mahalanobis": float(score_payload["mahalanobis"][token_index].item()),
                "logit_lens": float(score_payload["logit_lens"][token_index].item()),
                "pca_deviation": float(score_payload["pca_deviation"][token_index].item()),
            }
        )
    return ranked


def _print_consensus_ranking(score_payload: dict, token_texts: list[str]) -> None:
    headers = ["rank", "idx", "token", "consensus", "mahalanobis", "logit_lens", "pca_deviation"]
    rows: list[list[str]] = []
    for rank_index, entry in enumerate(_consensus_rankings(score_payload=score_payload, token_texts=token_texts), start=1):
        rows.append(
            [
                str(rank_index),
                str(int(entry["idx"])),
                _clip_cell(str(entry["token"]), 24),
                f"{float(entry['consensus']):.4f}",
                f"{float(entry['mahalanobis']):.4f}",
                f"{float(entry['logit_lens']):.4f}",
                f"{float(entry['pca_deviation']):.4f}",
            ]
        )

    print("\nConsensus suspicious-token ranking")
    print("Based on within-column percentile rank across mahalanobis, logit_lens, and pca_deviation.")
    for line in _build_table_lines(headers=headers, rows=rows, left_align_columns={2}):
        print(line)


def _print_aggregate_summary(score_payload: dict, aggregate_mode: str) -> None:
    for mode in _aggregate_modes(aggregate_mode):
        aggregated = aggregate_token_scores(score_payload, mode)
        print(f"\nAggregate scores ({mode})")
        ordered_names = REP_METRICS + BASELINE_METRICS + ["composite"]
        for name in ordered_names:
            if name in aggregated:
                print(f"{name:<20} {aggregated[name]:.4f}")
    print("\nComposite sample features")
    for name, value in score_payload["sample_features"].items():
        print(f"{name:<20} {value:.4f}")


def main() -> None:
    args = parse_args()
    _validate_input_args(args)
    profile = _resolve_profile(args)
    config = load_frozen_scoring_config(profile.stats_path)

    demo_input = _resolve_demo_input(args)
    prompt = _build_prompt(instruction=demo_input.instruction, context=demo_input.context)

    runner = InferenceRunner(
        model_name=profile.model_name,
        layers_spec=profile.layers_spec,
        device=args.device,
        max_seq_tokens=args.max_seq_tokens,
    )

    if demo_input.passage is not None:
        mode = "score_provided"
        response = demo_input.passage
    else:
        mode = "generate_then_score"
        response = runner.generate_response(prompt=prompt, max_new_tokens=args.max_new_tokens)

    sample = Sample(
        sample_id=f"live-demo-{profile.name}",
        source_id=f"live-demo-{profile.name}",
        prompt=prompt,
        response=response,
    )
    artifact = runner.run(sample, split="live_demo")
    score_payload = score_artifact(artifact, config)
    token_texts = _answer_token_texts(
        runner=runner,
        prompt=prompt,
        response=response,
        expected_tokens=int(score_payload["cosine_drift"].shape[0]),
    )

    print(f"profile: {profile.name}")
    print(f"mode: {mode}")
    print(f"model: {profile.model_name}")
    print(f"layers: {profile.layers_spec}")
    print(f"stats: {profile.stats_path}")
    print(f"answer_tokens: {len(token_texts)}")
    print("\nAnswer text")
    print(response)

    _print_token_table(score_payload=score_payload, token_texts=token_texts, include_baselines=args.show_baselines)
    _print_consensus_ranking(score_payload=score_payload, token_texts=token_texts)
    if args.show_aggregates:
        _print_aggregate_summary(score_payload=score_payload, aggregate_mode=args.aggregate)


if __name__ == "__main__":
    main()
