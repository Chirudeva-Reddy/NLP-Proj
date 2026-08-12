"""
Per-token hallucination detection metrics (Person 2 spec).

Every compute_* function takes answer-only hidden states and returns a 1-D
tensor of length n_answer_tokens. Higher score = more likely hallucinated.

Metrics:
  attention_entropy — pre-computed per-layer attention-row entropy (baseline 1),
                      averaged over layers.
  logit_confidence  — pre-computed per-token NLL of the emitted token (baseline 2).
  cosine_drift      — intra-answer drift, 1 - cos(h[t], h[t-1]) averaged over layers.
                      drift[0] is forced to 0 since there is no preceding token.
  mahalanobis       — per-layer Mahalanobis distance to the train faithful-token
                      distribution, averaged over layers.
  pca_deviation     — per-layer residual norm after projecting onto the train-fitted
                      PCA subspace, averaged over layers.
  logit_lens        — per-layer KL(p_final || p_layer) from projecting each layer's
                      hidden state through the LM head; averaged over layers.
  cie_top3          — Mahalanobis restricted to the 3 layers with the highest per-layer
                      AUROC on the validation split. A surrogate for Causal-Indirect-Effect
                      ranking until E3 activation-patching results are available.
  composite         — z-score normalise each metric (mean/std from train) then
                      average across metrics.

All metrics consume `hidden` shaped (n_layers, n_tokens, hidden_dim).
"""

from __future__ import annotations

from collections.abc import Iterable
import sys

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def pack_symmetric(matrices: torch.Tensor) -> torch.Tensor:
    """
    Pack a stack of symmetric (d, d) matrices into upper-triangle form.

    Input shape (n, d, d). Output shape (n, d*(d+1)/2). Halves storage of
    inv_cov in stats.pt without precision loss.
    """
    n, d, _ = matrices.shape
    idx = torch.triu_indices(d, d)
    return matrices[:, idx[0], idx[1]].contiguous()


def unpack_symmetric(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of pack_symmetric; returns (n, dim, dim) symmetric matrices."""
    n = packed.shape[0]
    idx = torch.triu_indices(dim, dim)
    full = torch.zeros(n, dim, dim, dtype=packed.dtype)
    full[:, idx[0], idx[1]] = packed
    full[:, idx[1], idx[0]] = packed
    return full


def _finite(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(tensor).float()
    if torch.isfinite(tensor).all():
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


# ── per-token scoring ─────────────────────────────────────────────────────────

def cosine_drift(hidden: torch.Tensor) -> torch.Tensor:
    """
    Intra-answer cosine drift averaged across layers.

    For each layer l and answer token t > 0:
        drift[l, t] = 1 - cos(h[l, t], h[l, t-1])
    The first token has no predecessor, so drift[:, 0] = 0.
    Final score per token = mean over layers.
    """
    h = _finite(hidden)  # (n_layers, n_tokens, d)
    n_tokens = h.shape[1]
    drift = torch.zeros(h.shape[0], n_tokens)
    if n_tokens > 1:
        sim = F.cosine_similarity(h[:, 1:], h[:, :-1], dim=-1, eps=1e-8)  # (n_layers, n_tokens-1)
        drift[:, 1:] = 1.0 - sim
    return drift.mean(dim=0)  # (n_tokens,)


def mahalanobis(hidden: torch.Tensor, stats: dict) -> torch.Tensor:
    """Mahalanobis distance from the train faithful-token distribution, mean-pooled over layers."""
    return mahalanobis_per_layer(hidden, stats).mean(dim=0)


def mahalanobis_per_layer(hidden: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    Per-layer Mahalanobis distance without averaging.

    Returns shape (n_layers, n_tokens). Used by the E2 layer-profile plot and by
    the CIE top-3 surrogate.
    """
    h   = _finite(hidden)
    mu  = _finite(stats["mean"]).unsqueeze(1)     # (n_layers, 1, d)
    inv = _finite(stats["inv_cov"])                # (n_layers, d, d)
    diff = h - mu
    sq = torch.einsum("ltd,lde,lte->lt", diff, inv, diff).clamp_min(0)
    return sq.sqrt()                                # (n_layers, n_tokens)


def cie_top3(hidden: torch.Tensor, stats: dict, top3_layers: list[int]) -> torch.Tensor:
    """
    Surrogate Causal-Indirect-Effect metric using the 3 layers with the highest
    per-layer Mahalanobis AUROC on the validation split (picked in 2-fit.py).

    Returns per-token Mahalanobis averaged over those 3 layers only.
    """
    per_layer = mahalanobis_per_layer(hidden, stats)        # (n_layers, n_tokens)
    return per_layer[top3_layers].mean(dim=0)               # (n_tokens,)


def attention_entropy(entropy_per_layer: torch.Tensor) -> torch.Tensor:
    """Per-token attention entropy averaged over the stored layers."""
    return _finite(entropy_per_layer).mean(dim=0)


def logit_confidence(nll_per_token: torch.Tensor) -> torch.Tensor:
    """Identity: per-token NLL is already the score (higher = less confident)."""
    return _finite(nll_per_token)


def pca_deviation(hidden: torch.Tensor, stats: dict) -> torch.Tensor:
    """PCA reconstruction-error norm per token, mean-pooled over layers."""
    h     = _finite(hidden)
    mu    = _finite(stats["mean"]).unsqueeze(1)    # (n_layers, 1, d)
    comps = _finite(stats["components"])           # (n_layers, k, d)
    diff  = h - mu
    # diff @ compᵀ @ comp reconstructs the component-subspace part;
    # the residual norm is the part that lives outside the PCA subspace.
    proj  = torch.einsum("ltd,lkd->ltk", diff, comps)
    recon = torch.einsum("ltk,lkd->ltd", proj, comps)
    return (diff - recon).norm(dim=-1).mean(dim=0)  # (n_tokens,)


def logit_lens(kl_per_layer: torch.Tensor) -> torch.Tensor:
    """
    Logit-lens KL divergence as an anomaly score.

    Expects per-layer KL(p_final || p_layer) already computed during inference,
    shape (n_layers, n_tokens). Averages over layers.
    """
    return _finite(kl_per_layer).mean(dim=0)


def composite(
    scores: dict[str, torch.Tensor],
    normalizers: dict[str, tuple[float, float] | tuple[float, float, float]],
) -> torch.Tensor:
    """
    Z-score normalise each metric then average across metrics.

    Normalizer entries may be (mean, std) or (mean, std, sign). The sign is
    +1 when higher raw score = more hallucinated on val, -1 otherwise; it
    prevents metrics with an inverted direction from cancelling real signal.
    """
    parts = []
    for name, tensor in scores.items():
        if name not in normalizers:
            continue
        entry = normalizers[name]
        if len(entry) == 3:
            mean, std, sign = entry
        else:
            mean, std = entry
            sign = 1.0
        parts.append(sign * (tensor.float() - mean) / max(std, 1e-8))
    return torch.stack(parts).mean(dim=0)


# ── train-fit helpers ─────────────────────────────────────────────────────────

def _accumulate_moments(artifacts: Iterable[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Streaming per-layer sum, outer-product sum, and token count over
    faithful tokens only (token_labels == 0). Filtering here is what makes
    μ/Σ and the PCA basis represent the faithful distribution; including
    hallucinated tokens contaminates the reference and flattens AUROC.
    """
    sums = outers = counts = None
    for a in tqdm(artifacts, desc="moments", unit="artifact", dynamic_ncols=True, file=sys.stdout):
        h = _finite(a["hidden_states"])            # (n_layers, n_tokens, d)
        labels = a["token_labels"].to(torch.bool)
        keep = ~labels
        if not keep.any():
            continue
        h = h[:, keep, :]
        n_layers, n_tokens, d = h.shape
        if sums is None:
            sums   = torch.zeros(n_layers, d)
            outers = torch.zeros(n_layers, d, d)
            counts = torch.zeros(n_layers)
        sums   += h.sum(dim=1)
        outers += torch.einsum("ltd,lte->lde", h, h)
        counts += n_tokens
    if sums is None:
        raise ValueError("No artifacts supplied for fitting.")
    return sums, outers, counts


def fit_mahalanobis(artifacts: list[dict], regularization: float = 1e-4) -> dict:
    """
    Per-layer mean and pinv-regularised inverse covariance.

    Covariance = E[xxᵀ] - μμᵀ, scaled by (n-1). Ridge λI keeps the matrix
    invertible when some directions have near-zero variance.
    """
    sums, outers, counts = _accumulate_moments(artifacts)
    n_layers, d = sums.shape

    mean = sums / counts.unsqueeze(1)                                      # (n_layers, d)
    centered = outers - counts.view(-1, 1, 1) * torch.einsum("ld,le->lde", mean, mean)
    cov = centered / (counts - 1).clamp_min(1).view(-1, 1, 1)
    cov = cov + regularization * torch.eye(d).unsqueeze(0)
    inv_cov = torch.linalg.pinv(_finite(cov))
    return {"mean": mean, "inv_cov": inv_cov}


def fit_pca(artifacts: list[dict], n_components: int = 16) -> dict:
    """
    Per-layer PCA via eigendecomposition of the covariance matrix.

    Uses torch.linalg.eigh on the (d, d) covariance rather than SVD on the
    (N, d) data matrix — much faster when N is large and d is fixed,
    and needs no external dependency.
    """
    sums, outers, counts = _accumulate_moments(artifacts)
    n_layers, d = sums.shape

    mean = sums / counts.unsqueeze(1)
    centered = outers - counts.view(-1, 1, 1) * torch.einsum("ld,le->lde", mean, mean)
    cov = centered / (counts - 1).clamp_min(1).view(-1, 1, 1)

    components = torch.zeros(n_layers, n_components, d)
    for l in tqdm(range(n_layers), desc="pca layers", unit="layer", dynamic_ncols=True, file=sys.stdout):
        eigvals, eigvecs = torch.linalg.eigh(_finite(cov[l]))  # ascending
        order = torch.argsort(eigvals, descending=True)
        top = min(n_components, eigvecs.shape[1])
        components[l, :top] = eigvecs[:, order[:top]].T
    return {"mean": mean, "components": components}


def fit_normalizers(metric_values: dict[str, list[torch.Tensor]]) -> dict[str, tuple[float, float]]:
    """Compute (mean, std) per metric over all train tokens for composite scoring."""
    normalizers: dict[str, tuple[float, float]] = {}
    for name, tensors in metric_values.items():
        if not tensors:
            continue
        flat = torch.cat([_finite(t).reshape(-1) for t in tensors])
        std = float(flat.std(unbiased=False))
        normalizers[name] = (float(flat.mean()), std if std > 1e-8 else 1.0)
    return normalizers
