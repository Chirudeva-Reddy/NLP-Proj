# NLP Track B — Pipeline

This repository runs a four-step hallucination-detection workflow on RAGTruth.

```
pipeline/1-infer.py  ->  pipeline/2-fit.py  ->  pipeline/3-score.py  ->  pipeline/4-eval.py
```

## Extended docs

- 📄 **Full Project Report (PDF)**: [`../docs/CS_F429_Project_Report.pdf`](../docs/CS_F429_Project_Report.pdf)
- 📌 **Master Project README**: [`../README.md`](../README.md)
- 📖 **Component Runbook**: [`../docs/PIPELINE_RUNBOOK.md`](../docs/PIPELINE_RUNBOOK.md)
- 🛡️ **Metrics Defense & Viva Guide**: [`../METRICS_VIVA.md`](../METRICS_VIVA.md)

## Metrics

Baselines (reported as comparators):

- `attention_entropy`
- `logit_confidence`

Representation metrics (used for strict composite tuning):

- `cosine_drift`
- `mahalanobis`
- `logit_lens`
- `pca_deviation`

Auxiliary metric:

- `cie_top3`

Strict composite:

- Built from representation metrics only.
- Tuned on validation at **sample level** (pooling + layer subset + PCA rank).
- Uses val-derived nonnegative layer weights inside the selected subset for
  each representation metric.
- Final score is a frozen non-supervised weighted z-score with TRAIN
  `median/IQR` scaling saved as `composite_sample_score`.

## Layout

```
NLP/
|- dataset/                 # RAGTruth source files (response.jsonl, source_info.jsonl)
|- pipeline/
|  |- 1-infer.py            # Inference + artifact save
|  |- 2-fit.py              # Train fit + val tuning + frozen weighted-zscore composite
|  |- 3-score.py            # Raw token metrics + strict sample features/scores
|  |- 4-eval.py             # AUROC table over sample scores
|  `- plot.py               # E2 per-layer AUROC profile
|- src/
|  |- dataset.py            # Source-grouped train/val/test split
|  |- inference.py          # Model wrapper and artifact I/O
|  |- metrics.py            # Metric math
|  `- evaluate.py           # auroc, bootstrap_ci, spearman, f1, ece
`- outputs/                 # Generated artifacts and results (git-ignored)
```

## Quick start

```bash
# 1) Infer all splits
python pipeline/1-infer.py --model "qwen 2.5 1.5" --layers last16

# 2) Fit on train and tune strict composite on val
python pipeline/2-fit.py

# 3) Score test split using frozen config from stats.pt
python pipeline/3-score.py --split test

# 4) Evaluate test once
python pipeline/4-eval.py
```

## Live demo

Single-sample live scoring now has a dedicated wrapper:

```bash
python scripts/live_demo.py \
  --profile local \
  --input-file live_demo_input.example.json
```

Fill [live_demo_input.example.json](/Users/tacticalcamel/Desktop/NLP/NLP-sub/live_demo_input.example.json) with the retrieved context and, if available, the provided passage. If `passage` is left empty, the script will generate one answer first and then score its answer tokens.

You can still use separate text files if needed:

```bash
python scripts/live_demo.py \
  --profile local \
  --context-file /path/to/context.txt \
  --passage-file /path/to/answer.txt
```

## What each step outputs

| Step      | Output                              | Notes                                                                               |
| --------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| `1-infer` | `outputs/artifacts/{split}/{id}.pt` | Answer-only hidden states, token labels, and optional auxiliary tensors.            |
| `2-fit`   | `outputs/stats.pt`                  | Train-fit stats and validation-tuned strict-composite config; no test-time fitting. |
| `3-score` | `outputs/scores/{id}.pt`            | Raw per-token metrics, `sample_features`, and `composite_sample_score`.             |
| `4-eval`  | `stdout` table                      | AUROC, CI, Spearman, F1, and ECE by metric.                                         |

## `stats.pt` strict-composite keys

- `metric_pooling`: chosen pooling (`max` or `mean`) per representation metric
- `metric_layers`: chosen layer subset per representation metric
- `metric_layer_weights`: val-derived nonnegative weights (sum to 1) inside each selected layer subset
- `pca_components_selected`: chosen PCA rank (`4`, `8`, or `16`)
- `mahalanobis_regularization`: selected λ from `{1e-4, 1e-3, 1e-2}`
- `composite_rule`: `weighted_zscore`
- `composite_features`: ordered representation metrics used by composite
- `composite_train_median` and `composite_train_iqr`: robust train scaler parameters
- `composite_signs`: val-derived sign correction per feature
- `composite_weights`: val-derived simplex weights per feature
- `composite_val_auroc`: frozen validation composite AUROC

Legacy fitted stats remain available:

- `mahalanobis` (packed covariance form)
- `pca`
- `cie_top3_layers`

## Evaluation behavior

- For raw token metrics, `pipeline/4-eval.py` still applies `--aggregate max|mean`.
- For composite, if `composite_sample_score` exists it is used directly and
  aggregation is ignored.
- Baselines remain report-only comparators and are never inputs to the strict
  composite.

## E5 transfer snapshot (last18, frozen)

Cross-domain zero-shot transfer was run without re-fitting Mahalanobis `μ, Σ`,
PCA, or composite weights. Current table:

| Metric | AUROC RAGTruth | AUROC HaluEval | Drop (`RAGTruth - HaluEval`) |
| --- | ---: | ---: | ---: |
| Cosine drift | 0.6244 | 0.8345 | -0.2102 |
| Mahalanobis distance | 0.5190 | 0.7006 | -0.1815 |
| Logit lens divergence | 0.5917 | 0.4337 | +0.1580 |
| PCA deviation | 0.5764 | 0.7186 | -0.1421 |
| Full composite | 0.6508 | 0.6450 | +0.0058 |

Most brittle metric by the defined rule (`max Drop`) is **logit_lens**.
Composite on HaluEval is **0.6450** (>= 0.62 threshold).

## Dataset

Place RAGTruth files at:

```
dataset/response.jsonl
dataset/source_info.jsonl
```

Both are git-ignored.
