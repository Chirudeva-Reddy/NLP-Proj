"""
Export frozen-profile live demo runs to a single JSON file for the web demo.

Runs every JSON input under examples/live_demo_inputs/ through the same
InferenceRunner + frozen scoring path used by scripts/live_demo.py, and writes
token-level scores, consensus ranks and aggregates to docs/site/demo_runs.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.live_demo import (
    BASELINE_METRICS,
    CONSENSUS_METRICS,
    REP_METRICS,
    DemoInput,
    _answer_token_texts,
    _build_prompt,
    _consensus_rankings,
    _read_demo_input,
)
from src.dataset import Sample
from src.inference import InferenceRunner
from src.scoring import (
    aggregate_token_scores,
    load_frozen_scoring_config,
    score_artifact,
)


ALL_METRICS: list[str] = REP_METRICS + BASELINE_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layers", default="last8")
    parser.add_argument("--stats", type=Path, default=Path("outputs/stats.pt"))
    parser.add_argument("--inputs-dir", type=Path, default=Path("examples/live_demo_inputs"))
    parser.add_argument("--output", type=Path, default=Path("../docs/site/demo_runs.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def _run_one(
    runner: InferenceRunner,
    config: object,
    demo_input: DemoInput,
    run_id: str,
    max_new_tokens: int,
) -> dict[str, object]:
    prompt = _build_prompt(instruction=demo_input.instruction, context=demo_input.context)
    if demo_input.passage is not None:
        mode = "score_provided"
        response = demo_input.passage
    else:
        mode = "generate_then_score"
        response = runner.generate_response(prompt=prompt, max_new_tokens=max_new_tokens)

    sample = Sample(
        sample_id=run_id,
        source_id=run_id,
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

    consensus_by_index: dict[int, float] = {
        int(entry["idx"]): float(entry["consensus"])
        for entry in _consensus_rankings(score_payload=score_payload, token_texts=token_texts)
    }

    tokens: list[dict[str, object]] = []
    for index, token_text in enumerate(token_texts):
        token_record: dict[str, object] = {
            "idx": index,
            "token": token_text,
            "consensus": round(consensus_by_index[index], 4),
        }
        for metric_name in ALL_METRICS:
            token_record[metric_name] = round(float(score_payload[metric_name][index].item()), 4)
        tokens.append(token_record)

    aggregates: dict[str, dict[str, float]] = {}
    for aggregate_mode in ("max", "mean"):
        aggregated = aggregate_token_scores(score_payload, aggregate_mode)
        aggregates[aggregate_mode] = {
            name: round(float(value), 4) for name, value in aggregated.items()
        }

    sample_features: dict[str, float] = {
        name: round(float(value), 4) for name, value in score_payload["sample_features"].items()
    }

    return {
        "id": run_id,
        "mode": mode,
        "instruction": demo_input.instruction,
        "context": demo_input.context,
        "response": response,
        "consensus_metrics": CONSENSUS_METRICS,
        "tokens": tokens,
        "aggregates": aggregates,
        "sample_features": sample_features,
    }


def main() -> None:
    args = parse_args()
    if not args.stats.exists():
        raise FileNotFoundError(f"Frozen stats not found: {args.stats}")
    if not args.inputs_dir.exists():
        raise FileNotFoundError(f"Demo inputs directory not found: {args.inputs_dir}")

    config = load_frozen_scoring_config(args.stats)
    runner = InferenceRunner(
        model_name=args.model,
        layers_spec=args.layers,
        device=args.device,
        max_seq_tokens=args.max_seq_tokens,
    )

    input_paths = sorted(args.inputs_dir.glob("*.json"))
    if not input_paths:
        raise FileNotFoundError(f"No demo input JSON files found under {args.inputs_dir}")

    runs: list[dict[str, object]] = []
    for input_path in input_paths:
        run_id = input_path.stem
        print(f"scoring {run_id}", flush=True)
        runs.append(
            _run_one(
                runner=runner,
                config=config,
                demo_input=_read_demo_input(input_path),
                run_id=run_id,
                max_new_tokens=args.max_new_tokens,
            )
        )

    payload = {
        "model": args.model,
        "layers": args.layers,
        "metrics": ALL_METRICS,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
