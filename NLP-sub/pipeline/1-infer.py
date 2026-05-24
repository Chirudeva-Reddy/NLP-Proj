"""
Step 1 — Run the LM over every sample and save one artifact per sample.

Output layout:
  outputs/artifacts/{train,val,test}/{sample_id}.pt

Each artifact contains the answer-only hidden states and per-layer logit-lens
KL divergences — everything Person 2 metrics and per-sample AUROC need.
See src/inference.py for the artifact schema.

Usage:
  python pipeline/1-infer.py [--model NAME] [--layers SPEC] [--device DEVICE]

  --model    HF model id. Default distilgpt2 (matches the Person 2 spec).
             Try gpt2-medium for stronger representations.
  --layers   "last4" (default), "all", or comma-separated indices e.g. "2,3,4,5".
  --device   "auto", "cpu", "mps"h.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.dataset import load, split
from src.inference import InferenceRunner, save


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--layers", default="last16")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/artifacts"))
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--max-seq-tokens", type=int, default=2048,
        help="Skip samples whose prompt+response exceeds this many tokens. "
             "Caps per-sample attention cost (O(seq²)) to keep throughput stable.",
    )
    p.add_argument("--limit", type=int, default=0, help="Cap samples per split (smoke test)")
    p.add_argument(
        "--total-limit", type=int, default=0,
        help="Cap total samples before splitting, keeping the 70-15-15 ratio.",
    )
    return p.parse_args()


def _validate_output_dir(output_dir: Path) -> None:
    """Confirm the target exists and is writable before spending inference time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_probe"
    probe.write_bytes(b"")
    probe.unlink()
    # On macOS external disks mount under /Volumes/<name>. If the user passed
    # such a path, make sure the volume is currently mounted so we don't start
    # writing ~tens of GB to the boot drive by mistake.
    parts = output_dir.resolve().parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        mount = Path("/") / parts[1] / parts[2]
        if not mount.is_mount():
            raise SystemExit(f"External volume {mount} is not mounted; refusing to start.")
    print(f"[output] writing artifacts to {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    _validate_output_dir(args.output_dir)

    samples = load(
        response_jsonl=Path("dataset/ragtruth/response.jsonl"),
        source_jsonl=Path("dataset/ragtruth/source_info.jsonl"),
    )
    splits = split(samples)

    if args.total_limit:
        train_n = int(args.total_limit * 0.70)
        val_n   = int(args.total_limit * 0.15)
        test_n  = args.total_limit - train_n - val_n
        splits["train"] = splits["train"][:train_n]
        splits["val"]   = splits["val"][:val_n]
        splits["test"]  = splits["test"][:test_n]

    runner = InferenceRunner(
        model_name=args.model,
        layers_spec=args.layers,
        device=args.device,
        max_seq_tokens=args.max_seq_tokens,
    )

    counts: dict[str, int] = {}
    for split_name, split_samples in splits.items():
        if args.limit:
            split_samples = split_samples[: args.limit]
        skipped = 0
        bar = tqdm(split_samples, desc=split_name, unit="sample", dynamic_ncols=True)
        for sample in bar:
            try:
                artifact = runner.run(sample, split=split_name)
                save(artifact, args.output_dir)
            except ValueError as e:
                skipped += 1
                bar.write(f"  skip {sample.sample_id}: {e}")
        counts[split_name] = len(split_samples) - skipped

    print("done:", counts)


if __name__ == "__main__":
    main()
