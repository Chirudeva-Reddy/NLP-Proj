"""
Run a causal LM over each sample and save a compact artifact.

Per-sample fields saved as a single .pt file:
  sample_id                     — str
  split                         — "train" | "val" | "test"
  has_hallucination             — int (1 if the response has any hallucination span)
  token_labels                  — int8 (n_answer_tokens,), per-token hallucination label
  hidden_states                 — float32 (n_layers, n_answer_tokens, hidden_dim)
  context_mean                  — float32 (n_layers, hidden_dim), mean of prompt tokens
  logit_lens_per_layer          — float32 (n_layers, n_answer_tokens), KL(p_final || p_layer)
  attention_entropy_per_layer   — float32 (n_layers, n_answer_tokens), mean-over-heads entropy of the
                                   attention row for each answer query position
  logit_confidence              — float32 (n_answer_tokens,), negative log-likelihood that the LM
                                   assigns to each emitted answer token (higher = less confident)
  answer_start_token_idx        — int, always 0 in this format (hidden_states is answer-only)
  answer_end_token_idx          — int, n_answer_tokens

The only logit-dependent signal we compute here is logit-lens KL divergence.
We compute it inline (not from saved logits) because saving full logits would
dominate storage: vocab=50257 × float32 ≈ 200 KB per answer token.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset import Sample


def _to_storage_float16(tensor: torch.Tensor) -> torch.Tensor:
    """Clamp to the finite fp16 range before compact storage."""
    return tensor.clamp(min=-65504.0, max=65504.0).to(torch.float16)


def resolve_layers(spec: str, n_transformer_layers: int) -> list[int]:
    """
    Resolve a layer selector to 0-based transformer layer indices.

    "lastN" → the final N transformer layers (e.g. last4, last12).
    "all"   → every transformer layer.
    Otherwise: comma-separated indices, e.g. "2,3,4,5".
    """
    if spec == "all":
        return list(range(n_transformer_layers))
    if spec.startswith("last"):
        k = int(spec[4:])
        return list(range(max(0, n_transformer_layers - k), n_transformer_layers))
    return [int(x) for x in spec.split(",")]


class InferenceRunner:
    """Loads the model once and produces one artifact per sample."""

    def __init__(
        self,
        model_name: str,
        layers_spec: str = "last4",
        device: str = "auto",
        max_seq_tokens: int = 1024,
    ):
        # max_seq_tokens hard-caps how long a prompt+response we'll process.
        # Attention is O(seq²), so one 3.5k-token sample costs ~12× more compute
        # than a 1k-token one. Capping keeps per-sample time predictable and
        # protects the throughput of the full 17k run.
        self.max_seq_tokens = max_seq_tokens
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # attn_implementation="eager" is required to get attention weights back
        # via output_attentions=True; SDPA/flash paths return an empty tuple.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()

        n_transformer_layers = self.model.config.num_hidden_layers
        self.layers = resolve_layers(layers_spec, n_transformer_layers)
        print(f"[inference] {model_name} on {device} | layers {self.layers} of {n_transformer_layers}")

    def run(self, sample: Sample, split: str) -> dict:
        full_text = sample.prompt + sample.response
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_tokens,
            add_special_tokens=True,
        )
        offsets: list[tuple[int, int]] = encoded["offset_mapping"][0].tolist()
        answer_char_start = len(sample.prompt)

        answer_start = next(
            (i for i, (s, e) in enumerate(offsets) if e > 0 and s >= answer_char_start),
            len(offsets),
        )
        if answer_start >= len(offsets):
            raise ValueError(f"Sample {sample.sample_id} truncated before the response.")
        # Skip samples that would dominate wall-clock time. At seq=max_seq_tokens
        # attention is ~O(seq²); better to drop the long tail than lose an hour
        # per outlier. These skips are logged and excluded from all downstream
        # splits, so the evaluation set stays consistent.
        if len(offsets) >= self.max_seq_tokens:
            raise ValueError(
                f"Sample {sample.sample_id} exceeds max_seq_tokens={self.max_seq_tokens} "
                f"(got {len(offsets)}); skipping to preserve throughput."
            )

        input_ids = encoded["input_ids"].to(self.device)
        attn_mask = encoded["attention_mask"].to(self.device)

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        # hidden_states is length n_layers+1: index 0 is the embedding layer,
        # indices 1..n are transformer outputs. We want transformer outputs for
        # the layers we requested, so slice with l+1.
        selected = torch.stack(
            [out.hidden_states[l + 1][0] for l in self.layers], dim=0
        ).cpu().float()                                       # (n_layers, seq, d)
        context_mean  = selected[:, :answer_start, :].mean(dim=1)        # (n_layers, d)
        answer_hidden = selected[:, answer_start:, :]                     # (n_layers, n_ans, d)

        logit_lens_per_layer = _compute_logit_lens(
            out.hidden_states, answer_start, self.model, self.layers
        )
        attention_entropy_per_layer = _compute_attention_entropy(
            out.attentions, answer_start, self.layers
        )
        logit_confidence = _compute_logit_confidence(
            out.logits, input_ids, answer_start
        )

        answer_offsets = offsets[answer_start:]
        token_labels = _label_tokens(answer_offsets, answer_char_start, sample.spans)
        sample_label = sample.sample_label if sample.sample_label is not None else int(any(l == 1 for l in token_labels))

        return {
            "sample_id":                    sample.sample_id,
            "split":                        split,
            "has_hallucination":            int(sample_label),
            "token_labels":                 torch.tensor(token_labels, dtype=torch.int8),
            "hidden_states":                answer_hidden,
            "context_mean":                 context_mean,
            "logit_lens_per_layer":         logit_lens_per_layer,
            "attention_entropy_per_layer":  attention_entropy_per_layer,
            "logit_confidence":             logit_confidence,
            "answer_start_token_idx":       0,
            "answer_end_token_idx":         answer_hidden.shape[1],
        }

    def generate_response(self, prompt: str, max_new_tokens: int) -> str:
        """
        Generate one deterministic answer from a prompt for live-demo scoring.

        The prompt is truncated to leave room for the requested continuation so
        the follow-up `run()` call can still fit inside `max_seq_tokens`.
        """
        max_prompt_tokens = max(1, self.max_seq_tokens - max_new_tokens)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer is missing both pad_token_id and eos_token_id.")

        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = input_ids.shape[1]
        continuation_ids = generated[0, prompt_len:]
        if continuation_ids.numel() == 0:
            raise ValueError("Model generation produced an empty continuation.")

        response = self.tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not response.strip():
            raise ValueError("Model generation produced only whitespace.")
        return response


def _compute_logit_lens(
    hidden_states: tuple,
    answer_start: int,
    model,
    layers: list[int],
) -> torch.Tensor:
    """
    KL(p_final || p_layer) per answer token for each requested layer.

    p_final comes from the true final layer (index -1 of hidden_states) after
    the same LM head used at decode time. For each requested layer index we
    project the intermediate hidden state through norm + LM head to get that
    layer's implied distribution and compute the divergence.
    """
    norm    = model.model.norm if hasattr(model, "model") and hasattr(model.model, "norm") else None
    lm_head = model.get_output_embeddings()

    def _project(h: torch.Tensor) -> torch.Tensor:
        return lm_head(norm(h)) if norm is not None else lm_head(h)

    # Do the softmax / KL on-device in the model dtype (bf16 here). Only the
    # final per-token KL scalar is upcast to fp32 on CPU — avoids a vocab-sized
    # fp32 blow-up per layer for a metric that's indistinguishable to fp16.
    with torch.inference_mode():
        h_final = hidden_states[-1][0, answer_start:, :]
        log_p_final = F.log_softmax(_project(h_final), dim=-1)
        p_final     = log_p_final.exp()

        per_layer = []
        for l in layers:
            h_l = hidden_states[l + 1][0, answer_start:, :]
            log_p_l = F.log_softmax(_project(h_l), dim=-1)
            kl = (p_final * (log_p_final - log_p_l)).sum(dim=-1).float().cpu()
            per_layer.append(kl)
    return torch.stack(per_layer, dim=0)  # (n_layers, n_ans)


def _compute_attention_entropy(
    attentions: tuple,
    answer_start: int,
    layers: list[int],
) -> torch.Tensor:
    """
    Per-answer-token attention entropy, averaged over heads, for each requested layer.

    attentions[l] has shape (1, heads, seq, seq). We take rows corresponding to
    answer query positions and compute Shannon entropy over the key dimension,
    then mean over heads. High entropy = diffuse attention = weaker grounding.
    """
    per_layer = []
    for l in layers:
        att = attentions[l][0, :, answer_start:, :].float()      # (heads, n_ans, seq)
        log_att = att.clamp_min(1e-12).log()
        entropy = -(att * log_att).sum(dim=-1)                   # (heads, n_ans)
        per_layer.append(entropy.mean(dim=0).cpu())              # (n_ans,)
    return torch.stack(per_layer, dim=0)                         # (n_layers, n_ans)


def _compute_logit_confidence(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_start: int,
) -> torch.Tensor:
    """
    Negative log-likelihood the LM assigns to each emitted answer token.

    The token at full-sequence position p is predicted from logits at p-1.
    For the answer span starting at answer_start, confidence for the i-th answer
    token is -log p(input_ids[answer_start + i] | context up to answer_start + i - 1).
    Higher NLL = less confident = more likely to be hallucinated.
    """
    n_ans = input_ids.shape[1] - answer_start
    pred_positions = torch.arange(answer_start - 1, answer_start - 1 + n_ans, device=logits.device)
    pred_positions = pred_positions.clamp_min(0)
    target_ids = input_ids[0, answer_start:]                               # (n_ans,)
    log_probs  = F.log_softmax(logits[0, pred_positions].float(), dim=-1)   # (n_ans, vocab)
    nll = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)          # (n_ans,)
    return nll.cpu()


def _label_tokens(
    offsets: list[tuple[int, int]],
    answer_char_start: int,
    spans,
) -> list[int]:
    """
    Token-level binary label: 1 if the token's character range overlaps any
    hallucination span when expressed relative to the start of the response.
    """
    labels = []
    for char_start, char_end in offsets:
        rel_start = char_start - answer_char_start
        rel_end   = char_end   - answer_char_start
        is_hal = any(rel_start < span.end and rel_end > span.start for span in spans)
        labels.append(1 if is_hal else 0)
    return labels


_FP16_STORAGE_KEYS = (
    "hidden_states",
    "context_mean",
    "logit_lens_per_layer",
    "attention_entropy_per_layer",
    "logit_confidence",
)


def save(artifact: dict, out_dir: Path) -> Path:
    """
    Save artifact to out_dir/{split}/{sample_id}.pt with fp16 compaction.

    The forward runs in bf16; we downcast analysis tensors to fp16 on disk for
    a ~2× space saving vs fp32. load() upcasts back to fp32 transparently.
    AUROC is rank-based and these tensors are high-dimensional/aggregated, so
    fp16 round-off does not change scoring beyond noise.
    """
    path = out_dir / artifact["split"] / f"{artifact['sample_id']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(artifact)
    for key in _FP16_STORAGE_KEYS:
        if key in payload and isinstance(payload[key], torch.Tensor):
            payload[key] = _to_storage_float16(payload[key])

    torch.save(payload, path)
    return path


def load(path: Path) -> dict:
    """
    Load a saved artifact back to CPU tensors.

    Legacy quantized/float16 artifacts are still accepted and normalized to
    float32 so old runs remain readable.
    """
    a = torch.load(path, map_location="cpu", weights_only=False)
    if "hidden_states_q8" in a:
        q = a.pop("hidden_states_q8").float()
        scale = a.pop("hidden_states_scale").float().unsqueeze(1)
        a["hidden_states"] = q * scale
    for key in ("hidden_states", "context_mean", "logit_lens_per_layer", "attention_entropy_per_layer", "logit_confidence"):
        if key in a and a[key].dtype != torch.float32:
            a[key] = a[key].float()
    return a
