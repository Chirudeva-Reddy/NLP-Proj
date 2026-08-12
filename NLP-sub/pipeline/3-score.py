"""
Step 3 — Score artifacts and build strict composite sample scores.

Outputs one .pt file per sample with:
  - raw per-token metrics (unchanged from previous workflow)
  - sample_features (frozen strict-composite inputs)
  - composite_sample_score (frozen weighted-zscore score)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import load as load_artifact
from src.scoring import (
    load_frozen_scoring_config,
    score_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/artifacts"))
    parser.add_argument("--stats", type=Path, default=Path("outputs/stats.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scores"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = args.artifacts_dir / args.split
    paths = sorted(split_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No artifacts in {split_dir}")
    config = load_frozen_scoring_config(args.stats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scoring {len(paths)} {args.split} artifacts …")

    for path in tqdm(paths, desc=f"score {args.split}", unit="sample", dynamic_ncols=True, file=sys.stdout):
        artifact = load_artifact(path)
        torch.save(score_artifact(artifact, config), args.output_dir / f"{artifact['sample_id']}.pt")

    print("Done.")


if __name__ == "__main__":
    main()
