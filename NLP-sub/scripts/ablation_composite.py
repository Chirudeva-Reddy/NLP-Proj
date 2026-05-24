"""
Ablation: composite variants using frozen artifacts and stats.

Variants compared:
  1. current-4   — 4-feature weighted-zscore (replicates existing composite)
  2. logreg-4    — 4-feature logistic regression
  3. zscore-6    — 6-feature weighted-zscore (+attention_entropy, +logit_confidence)
  4. logreg-6    — 6-feature logistic regression

Strict split discipline:
  train  → fit normalizers / logistic regression
  val    → choose pooling for new features and logistic C
  test   → final evaluation ONCE per variant, reported at the end

No pipeline files are modified. outputs/stats.pt stays frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.evaluate import auroc
from src.metrics import mahalanobis_per_layer, pca_deviation, unpack_symmetric

ARTIFACT_DIRS: dict[str, Path] = {
    "train": Path("outputs/artifacts/train"),
    "val":   Path("outputs/artifacts/val"),
    "test":  Path("outputs/artifacts/test"),
}
STATS_PATH: Path = Path("outputs/stats.pt")
OUT_DIR: Path    = Path("outputs")
POOLS            = ("max", "mean", "p90")
SCORES_DIR: Path = Path("outputs/scores_test")
EVAL_AGGREGATE: str = "max"
METRIC_TABLE_ROWS: list[tuple[str, str]] = [
    ("Attention entropy (Baseline 1)", "attention_entropy"),
    ("Logit confidence (Baseline 2)", "logit_confidence"),
    ("Cosine drift", "cosine_drift"),
    ("Mahalanobis distance", "mahalanobis"),
    ("Logit lens divergence", "logit_lens"),
    ("PCA deviation", "pca_deviation"),
    ("CIE top-3 layers", "cie_top3"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", type=Path, default=Path("outputs/artifacts"),
                   help="Root dir containing train/, val/, test/ subdirs.")
    p.add_argument("--stats", type=Path, default=Path("outputs/stats.pt"),
                   help="Path to frozen stats.pt produced by pipeline/2-fit.py.")
    p.add_argument("--scores-dir", type=Path, default=Path("outputs/scores_test"),
                   help="Path to existing score .pt files for the standalone metric rows.")
    p.add_argument("--aggregate", choices=["max", "mean"], default="max",
                   help="Per-sample token aggregation for standalone test AUROC rows.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Where to write ablation_composite.{json,md}.")
    return p.parse_args()


def _pool(tensor: torch.Tensor, mode: str) -> float:
    t = tensor.float()
    if mode == "max":
        return float(t.max().item())
    if mode == "mean":
        return float(t.mean().item())
    if mode == "p90":
        return float(torch.quantile(t, 0.9).item())
    raise ValueError(mode)


def _metric_score(sample: dict, metric: str, aggregate: str) -> float:
    if metric not in sample:
        raise KeyError(f"Missing metric '{metric}' in score file.")
    return _pool(sample[metric].float(), aggregate)


def _finite(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _cosine_drift_per_layer(hidden: torch.Tensor) -> torch.Tensor:
    state = _finite(hidden)
    n_layers, n_tokens, _ = state.shape
    drift = torch.zeros(n_layers, n_tokens)
    if n_tokens > 1:
        sim = F.cosine_similarity(state[:, 1:], state[:, :-1], dim=-1, eps=1e-8)
        drift[:, 1:] = 1.0 - sim
    return drift


def _pca_dev_per_layer(hidden: torch.Tensor, pca_stats: dict, n_comp: int) -> torch.Tensor:
    state = _finite(hidden)
    mean = _finite(pca_stats["mean"]).unsqueeze(1)
    comps = _finite(pca_stats["components"])[:, :n_comp, :]
    diff = state - mean
    proj = torch.einsum("ltd,lkd->ltk", diff, comps)
    recon = torch.einsum("ltk,lkd->ltd", proj, comps)
    return (diff - recon).norm(dim=-1)


def _sample_feature(per_layer: torch.Tensor, layers: list[int],
                     weights: list[float], pool: str) -> float:
    sel = per_layer[layers]
    w = torch.tensor(weights, dtype=sel.dtype)
    ws = float(w.sum())
    reduced = (sel * (w / ws).unsqueeze(1)).sum(dim=0) if ws > 1e-8 else sel.mean(dim=0)
    return _pool(reduced, pool)


def _load_split(split: str, stats: dict) -> tuple[list[int], dict[str, list[float]]]:
    """Load all artifacts for a split, returning labels and per-feature value lists."""
    mahal_s = stats["mahalanobis"]
    if "inv_cov_tri" in mahal_s:
        mahal_s = {
            "mean": mahal_s["mean"],
            "inv_cov": unpack_symmetric(mahal_s["inv_cov_tri"], dim=mahal_s["mean"].shape[-1]),
        }
    pca_s   = stats["pca"]
    n_comp  = int(stats["pca_components_selected"])
    m_layers: dict = stats["metric_layers"]
    m_pool:   dict = stats["metric_pooling"]
    m_w:      dict = stats["metric_layer_weights"]

    labels: list[int] = []
    feats: dict[str, list[float]] = {
        k: [] for k in ["cosine_drift", "mahalanobis", "logit_lens", "pca_deviation",
                        "attn_entropy_max", "attn_entropy_mean", "attn_entropy_p90",
                        "logit_conf_max",   "logit_conf_mean",   "logit_conf_p90"]
    }

    paths = sorted(ARTIFACT_DIRS[split].glob("*.pt"))
    for path in tqdm(paths, desc=split, unit="sample", dynamic_ncols=True):
        a = torch.load(path, map_location="cpu", weights_only=True)
        if "hidden_states_q8" in a:
            q = a.pop("hidden_states_q8").float()
            scale = a.pop("hidden_states_scale").float().unsqueeze(1)
            a["hidden_states"] = q * scale
        for key in ("hidden_states", "context_mean", "logit_lens_per_layer",
                    "attention_entropy_per_layer", "logit_confidence"):
            if key in a and a[key].dtype != torch.float32:
                a[key] = a[key].float()

        hidden = _finite(a["hidden_states"])
        labels.append(int(a.get("has_hallucination", int(a["token_labels"].any()))))

        cd   = _cosine_drift_per_layer(hidden)
        mahl = mahalanobis_per_layer(hidden, mahal_s)
        pdev = _pca_dev_per_layer(hidden, pca_s, n_comp)
        ll   = _finite(a["logit_lens_per_layer"]) if "logit_lens_per_layer" in a else torch.zeros(1, 1)

        feats["cosine_drift"].append(  _sample_feature(cd,   m_layers["cosine_drift"],  m_w["cosine_drift"],  m_pool["cosine_drift"]))
        feats["mahalanobis"].append(   _sample_feature(mahl, m_layers["mahalanobis"],   m_w["mahalanobis"],   m_pool["mahalanobis"]))
        feats["logit_lens"].append(    _sample_feature(ll,   m_layers["logit_lens"],    m_w["logit_lens"],    m_pool["logit_lens"]))
        feats["pca_deviation"].append( _sample_feature(pdev, m_layers["pca_deviation"], m_w["pca_deviation"], m_pool["pca_deviation"]))

        if "attention_entropy_per_layer" in a:
            ae = _finite(a["attention_entropy_per_layer"]).mean(dim=0)
            for p in POOLS:
                feats[f"attn_entropy_{p}"].append(_pool(ae, p))
        else:
            for p in POOLS:
                feats[f"attn_entropy_{p}"].append(0.0)

        if "logit_confidence" in a:
            lc = _finite(a["logit_confidence"])
            for p in POOLS:
                feats[f"logit_conf_{p}"].append(_pool(lc, p))
        else:
            for p in POOLS:
                feats[f"logit_conf_{p}"].append(0.0)

    return labels, feats


def _weighted_zscore(
    train_feats: dict[str, list[float]],
    eval_feats:  dict[str, list[float]],
    train_labels: list[int],
    feature_names: list[str],
) -> list[float]:
    """Weighted z-score composite: fit median/IQR on train, apply to eval."""
    scores = np.zeros(len(next(iter(eval_feats.values()))))
    for name in feature_names:
        tr = np.array(train_feats[name])
        ev = np.array(eval_feats[name])
        med = float(np.median(tr))
        iqr = float(np.percentile(tr, 75) - np.percentile(tr, 25))
        scale = iqr if iqr > 1e-8 else 1.0
        z_tr = (tr - med) / scale
        z_ev = (ev - med) / scale
        tr_auc = auroc(train_labels, z_tr.tolist())
        sign = 1.0 if tr_auc >= 0.5 else -1.0
        w = max(0.0, abs(tr_auc - 0.5))
        scores += sign * w * z_ev
    return scores.tolist()


def _logreg(
    train_feats: dict[str, list[float]],
    val_feats:   dict[str, list[float]],
    eval_feats:  dict[str, list[float]],
    train_labels: list[int],
    val_labels:   list[int],
    feature_names: list[str],
) -> list[float]:
    X_tr = np.column_stack([train_feats[n] for n in feature_names])
    X_val = np.column_stack([val_feats[n]  for n in feature_names])
    X_ev  = np.column_stack([eval_feats[n] for n in feature_names])
    scaler = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_ev_s  = scaler.transform(X_ev)

    best_c, best_auc = 1.0, 0.0
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42).fit(X_tr_s, train_labels)
        auc = auroc(val_labels, clf.predict_proba(X_val_s)[:, 1].tolist())
        if auc > best_auc:
            best_auc, best_c = auc, c

    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42).fit(X_tr_s, train_labels)
    return clf.predict_proba(X_ev_s)[:, 1].tolist()


def _best_pool_for(
    train_feats: dict, val_feats: dict,
    train_labels: list[int], val_labels: list[int],
    prefix: str,
) -> str:
    best_p, best_auc = "max", 0.0
    for p in POOLS:
        auc = auroc(val_labels, val_feats[f"{prefix}_{p}"])
        if auc < 0.5:
            auc = 1.0 - auc
        if auc > best_auc:
            best_auc, best_p = auc, p
    return best_p


def _load_test_metric_rows(scores_dir: Path, aggregate: str) -> list[dict[str, float | str]]:
    paths = sorted(scores_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No score files found in {scores_dir}.")

    labels: list[int] = []
    pooled: dict[str, list[float]] = {metric_key: [] for _, metric_key in METRIC_TABLE_ROWS}
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        labels.append(int(sample.get("has_hallucination", int(sample["token_labels"].any()))))
        for _, metric_key in METRIC_TABLE_ROWS:
            pooled[metric_key].append(_metric_score(sample, metric_key, aggregate))

    rows: list[dict[str, float | str]] = []
    for metric_label, metric_key in METRIC_TABLE_ROWS:
        rows.append({
            "metric": metric_label,
            "test_auroc": float(auroc(labels, pooled[metric_key])),
        })
    return rows


def main() -> None:
    args = parse_args()
    global ARTIFACT_DIRS, STATS_PATH, OUT_DIR, SCORES_DIR, EVAL_AGGREGATE
    ARTIFACT_DIRS = {
        "train": args.artifacts_dir / "train",
        "val":   args.artifacts_dir / "val",
        "test":  args.artifacts_dir / "test",
    }
    STATS_PATH = args.stats
    SCORES_DIR = args.scores_dir
    EVAL_AGGREGATE = args.aggregate
    OUT_DIR    = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)

    tr_labels, tr_feats = _load_split("train", stats)
    va_labels, va_feats = _load_split("val",   stats)
    te_labels, te_feats = _load_split("test",  stats)

    base_4 = ["cosine_drift", "mahalanobis", "logit_lens", "pca_deviation"]

    ae_pool  = _best_pool_for(tr_feats, va_feats, tr_labels, va_labels, "attn_entropy")
    lc_pool  = _best_pool_for(tr_feats, va_feats, tr_labels, va_labels, "logit_conf")
    ae_key   = f"attn_entropy_{ae_pool}"
    lc_key   = f"logit_conf_{lc_pool}"
    feat_6   = base_4 + [ae_key, lc_key]
    print(f"[ablation] new feature pools chosen on val: {ae_key}, {lc_key}")

    results = {}

    for feat_names, label in [(base_4, "zscore-4"), (feat_6, "zscore-6")]:
        va_scores = _weighted_zscore(tr_feats, va_feats, tr_labels, feat_names)
        te_scores = _weighted_zscore(tr_feats, te_feats, tr_labels, feat_names)
        results[label] = {
            "val_auroc":  auroc(va_labels, va_scores),
            "test_auroc": auroc(te_labels, te_scores),
        }

    for feat_names, label in [(base_4, "logreg-4"), (feat_6, "logreg-6")]:
        va_scores = _logreg(tr_feats, va_feats, va_feats, tr_labels, va_labels, feat_names)
        te_scores = _logreg(tr_feats, va_feats, te_feats, tr_labels, va_labels, feat_names)
        results[label] = {
            "val_auroc":  auroc(va_labels, va_scores),
            "test_auroc": auroc(te_labels, te_scores),
        }

    existing_test = 0.6511180027134192
    best_var = max(results, key=lambda k: results[k]["val_auroc"])

    print(f"\n{'variant':<12} {'val AUROC':>10}  {'test AUROC':>10}  {'Δ vs current':>13}")
    print("-" * 52)
    for var, r in results.items():
        marker = " ← best val" if var == best_var else ""
        delta = r["test_auroc"] - existing_test
        print(f"{var:<12} {r['val_auroc']:>10.4f}  {r['test_auroc']:>10.4f}  {delta:>+13.4f}{marker}")
    print(f"\n{'current':12} {'':>10}  {existing_test:>10.4f}  {'(baseline)':>13}")

    report = {
        "existing_composite_test_auroc": existing_test,
        "new_feature_pools": {ae_key: "chosen on val", lc_key: "chosen on val"},
        "feature_sets": {"4-feature": base_4, "6-feature": feat_6},
        "variants": results,
        "test_auroc_table": _load_test_metric_rows(SCORES_DIR, EVAL_AGGREGATE) + [
            {"metric": "Full composite (zscore-4)", "test_auroc": float(results["zscore-4"]["test_auroc"])},
            {"metric": "Full composite (zscore-6)", "test_auroc": float(results["zscore-6"]["test_auroc"])},
            {"metric": "Full composite (logreg-4)", "test_auroc": float(results["logreg-4"]["test_auroc"])},
            {"metric": "Full composite (logreg-6)", "test_auroc": float(results["logreg-6"]["test_auroc"])},
        ],
        "best_val_variant": best_var,
        "best_test_auroc": results[best_var]["test_auroc"],
        "delta_vs_existing": results[best_var]["test_auroc"] - existing_test,
    }
    (OUT_DIR / "ablation_composite.json").write_text(json.dumps(report, indent=2))

    md = ["# Composite Ablation Results", "",
          f"Baseline (current composite): **{existing_test:.4f}**", "",
          f"| Variant | Val AUROC | Test AUROC | Δ vs baseline |",
          f"|---------|-----------|------------|---------------|"]
    for var, r in results.items():
        d = r["test_auroc"] - existing_test
        md.append(f"| {var} | {r['val_auroc']:.4f} | {r['test_auroc']:.4f} | {d:+.4f} |")
    md += [
        "",
        "## Main Table Style Test AUROC",
        "",
        "| Metric / Composite | Test AUROC |",
        "|--------------------|------------|",
    ]
    for row in report["test_auroc_table"]:
        md.append(f"| {row['metric']} | {row['test_auroc']:.4f} |")
    md += ["", f"**Best variant (by val):** {best_var} → test AUROC {results[best_var]['test_auroc']:.4f}",
           "", f"New feature pools selected on val: `{ae_key}`, `{lc_key}`"]
    (OUT_DIR / "ablation_composite.md").write_text("\n".join(md))

    print(f"\n[ablation] wrote outputs/ablation_composite.json + .md")


if __name__ == "__main__":
    main()
