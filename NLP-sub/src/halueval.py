"""
Load HaluEval subsets into the repo's Sample format.
"""

from __future__ import annotations

import json
import ssl
from pathlib import Path
from urllib.request import urlopen
from json import JSONDecodeError

import certifi

from .dataset import Sample


HALUEVAL_QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


def ensure_qa_dataset(path: Path) -> Path:
    """Download the official HaluEval QA JSON once if it is not already present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(HALUEVAL_QA_URL, context=context) as response:
            path.write_bytes(response.read())
    return path


def load_qa(path: Path) -> list[Sample]:
    """Create one faithful and one hallucinated sample per HaluEval QA item."""
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]
    else:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                rows = parsed
            elif isinstance(parsed, dict):
                rows = [parsed]
            else:
                raise ValueError(f"Unsupported JSON root type in {path}: {type(parsed).__name__}")
        except JSONDecodeError:
            # Some local copies are JSONL but saved with a .json suffix.
            rows = [
                json.loads(line)
                for line in raw_text.splitlines()
                if line.strip()
            ]
    samples: list[Sample] = []
    for idx, row in enumerate(rows):
        source_id = f"halueval-qa-{idx}"
        prompt = (
            "Answer the question using only the provided knowledge.\n\n"
            f"Knowledge: {row['knowledge']}\n\n"
            f"Question: {row['question']}\n\n"
            "Answer:"
        )
        samples.append(
            Sample(
                sample_id=f"{source_id}-right",
                source_id=source_id,
                prompt=prompt,
                response=row["right_answer"],
                sample_label=0,
            )
        )
        samples.append(
            Sample(
                sample_id=f"{source_id}-hallucinated",
                source_id=source_id,
                prompt=prompt,
                response=row["hallucinated_answer"],
                sample_label=1,
            )
        )
    return samples
