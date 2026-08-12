"""
Replay-only utilities for E6 and E7.

The main inference pipeline saves residual-stream hidden states for a compact
last-N slice. E6 and E7 need something the main pipeline does not store:

  - per-layer `self_attn` output (the attention sublayer's contribution)
  - per-layer `mlp`       output (the FFN sublayer's contribution)
  - full model depth, not only the saved slice

This module owns the forward-hook machinery to extract those tensors from the
frozen model, without modifying the main pipeline. The functions here mirror
the tokenization and answer-slicing logic of `src/inference.py` so that a
sample_id here maps to the same answer-token span as in the frozen artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset import Sample


@dataclass
class TokenizedSample:
    input_ids: torch.Tensor       # (1, seq)
    answer_start: int             # first answer token index in the full sequence
    n_seq: int                    # total tokens


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name: str, device: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def tokenize_sample(tokenizer, sample: Sample, max_seq_tokens: int, device: str) -> TokenizedSample | None:
    """
    Reproduce the tokenization used by the frozen inference run so that the
    answer_start index matches the saved artifact. Returns None if the sample
    would have been skipped (truncation or over-length).
    """
    full = sample.prompt + sample.response
    encoded = tokenizer(
        full,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_tokens,
        add_special_tokens=True,
    )
    offsets = encoded["offset_mapping"][0].tolist()
    ans_char_start = len(sample.prompt)
    answer_start = next(
        (i for i, (s, e) in enumerate(offsets) if e > 0 and s >= ans_char_start),
        len(offsets),
    )
    if answer_start >= len(offsets):
        return None
    if len(offsets) >= max_seq_tokens:
        return None
    return TokenizedSample(
        input_ids=encoded["input_ids"].to(device),
        answer_start=answer_start,
        n_seq=len(offsets),
    )


def _stack_capture(per_layer: list[torch.Tensor | None], n_layers: int) -> torch.Tensor:
    """Stack per-layer captures (each (1, seq, d)) into (n_layers, seq, d) on CPU fp32."""
    out = []
    for l in range(n_layers):
        tensor = per_layer[l]
        if tensor is None:
            raise RuntimeError(f"No capture for layer {l}")
        out.append(tensor[0].float().cpu())
    return torch.stack(out, dim=0)


def capture_component_outputs(model, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Run a forward pass and return dict with keys 'self_attn' and 'mlp',
    each shape (n_layers, seq, hidden_dim) on CPU fp32.
    """
    n_layers = model.config.num_hidden_layers
    blocks = model.model.layers

    attn_caps: list[torch.Tensor | None] = [None] * n_layers
    mlp_caps: list[torch.Tensor | None] = [None] * n_layers

    def make_attn_hook(idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            attn_caps[idx] = tensor.detach()
        return hook

    def make_mlp_hook(idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            mlp_caps[idx] = tensor.detach()
        return hook

    handles = []
    for l in range(n_layers):
        handles.append(blocks[l].self_attn.register_forward_hook(make_attn_hook(l)))
        handles.append(blocks[l].mlp.register_forward_hook(make_mlp_hook(l)))

    try:
        with torch.inference_mode():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    return {
        "self_attn": _stack_capture(attn_caps, n_layers),
        "mlp":       _stack_capture(mlp_caps, n_layers),
    }


def update_direction_drift(component_states: torch.Tensor) -> torch.Tensor:
    """
    1 - cos(state[l, t], state[l, t-1]) along the token axis.

    Expects shape (n_layers, n_tokens, hidden_dim). The first token's drift is
    defined as 0 since there is no predecessor. Operates per-layer independently.
    Returns shape (n_layers, n_tokens).

    Interpretation note: when called on `self_attn` or `mlp` outputs, this
    measures drift of the *update vector* the sublayer contributes, NOT drift
    of the cumulative residual. Label figures accordingly.
    """
    n_layers, n_tokens, _ = component_states.shape
    drift = torch.zeros(n_layers, n_tokens)
    if n_tokens <= 1:
        return drift
    prev = component_states[:, :-1, :]
    curr = component_states[:, 1:, :]
    cos = torch.nn.functional.cosine_similarity(curr, prev, dim=-1, eps=1e-8)
    drift[:, 1:] = 1.0 - cos
    return drift


def contiguous_thirds(n_layers: int) -> dict[str, list[int]]:
    """Split range(n_layers) into early/mid/late contiguous thirds."""
    import numpy as np
    parts = np.array_split(range(n_layers), 3)
    return {
        "early": list(map(int, parts[0])),
        "mid":   list(map(int, parts[1])),
        "late":  list(map(int, parts[2])),
    }
