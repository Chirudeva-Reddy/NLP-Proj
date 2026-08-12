"""
Load and split the RAGTruth dataset.

RAGTruth provides two files:
  response.jsonl   — one LLM response per line, with hallucination spans as
                     character offsets into the response text.
  source_info.jsonl — one source per line, with a ready-to-use `prompt` field.

The two files are joined on source_id. We then split by source group (not by
individual sample) so that all responses from the same retrieved document go
to the same split — this prevents near-duplicate leakage across train/test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Span:
    """A hallucinated character span within the response text."""
    start: int  # inclusive character offset in response
    end: int    # exclusive character offset in response
    label: str  # e.g. "Evident Conflict", "Intrinsic Hallucination"


@dataclass
class Sample:
    sample_id: str
    source_id: str
    prompt: str          # full formatted prompt from source_info (append response to get model input)
    response: str        # the LLM's answer text
    spans: list[Span] = field(default_factory=list)
    sample_label: int | None = None


def load(response_jsonl: Path, source_jsonl: Path) -> list[Sample]:
    """Merge response and source files into a flat list of Sample objects."""
    # Index prompts by source_id for O(1) lookup during the response scan
    prompts: dict[str, str] = {}
    with source_jsonl.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts[row["source_id"]] = row["prompt"]

    samples: list[Sample] = []
    with response_jsonl.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            spans = [
                Span(
                    start=lbl["start"],
                    end=lbl["end"],
                    label=lbl.get("label_type", "hallucinated"),
                )
                for lbl in row.get("labels", [])
                # Exclude implicit hallucinations: these flag missing information
                # rather than clearly wrong statements, making them ambiguous labels.
                if not lbl.get("implicit_true", False)
            ]
            samples.append(Sample(
                sample_id=str(row["id"]),
                source_id=str(row["source_id"]),
                prompt=prompts[row["source_id"]],
                response=row["response"],
                spans=spans,
            ))
    return samples


def split(
    samples: list[Sample],
    *,
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """
    Assign samples to train / val / test splits by source group.

    All samples sharing a source_id go to the same split, so the same
    retrieved passages never appear in both training and evaluation data.
    The hash-based sort is deterministic: given the same seed the split is
    always identical, which is required for reproducible AUROC comparisons.
    """
    # Collect all responses that share the same source into one group
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.source_id, []).append(s)

    # Deterministic shuffle via hash so the split never changes between runs
    ordered = sorted(
        groups.items(),
        key=lambda kv: hashlib.sha256(f"{kv[0]}:{seed}".encode()).hexdigest(),
    )

    n = len(ordered)
    n_train = int(train * n)
    n_val = int(val * n)

    result: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for i, (_, group) in enumerate(ordered):
        if i < n_train:
            result["train"].extend(group)
        elif i < n_train + n_val:
            result["val"].extend(group)
        else:
            result["test"].extend(group)
    return result
