# NLP Track B Comprehensive Guide

This file is the full formula-and-implementation guide for the current `NLP-sub/` submission copy.

It is written for teammates who may need to explain:
- what each metric means,
- how each pipeline step works,
- how experiments E1-E8 were generated,
- and why train-fit statistics stay fixed at test time.

Short storage note:
- the frozen `last18` run also exists on the external SSD under `/Volumes/My Passport/NLP/outputs/`, including `artifacts_qwen15b_instruct_last18`, `scores_qwen15b_instruct_last18`, and `stats_qwen15b_instruct_last18.pt`.

---

## 1. What This Repository Does

The project detects hallucination using the model's internal representations instead of checking facts after generation.

The main path is:

```text
dataset -> inference artifacts -> train/val fitting -> scored test samples -> evaluation
```

The core pipeline lives in:

```text
pipeline/1-infer.py
pipeline/2-fit.py
pipeline/3-score.py
pipeline/4-eval.py
This provides us with the values used in e1.
pipeline/plot.py gives the line plot we used in the PPT for e2.
```
The core static result for E1,E2 lies in what we artifacts we are getting from the model pass in the form of a .pt file.

The analysis experiments live in:

```text
scripts/e3_patching.py
scripts/e4_temporal.py
scripts/e5_halueval.py
scripts/e6_component_drift.py
scripts/e7_failures.py
scripts/e8_sota_gap.py
```

---

## 2. Repository Map

### Core source files

| File | Role |
|---|---|
| `src/dataset.py` | Loads RAGTruth, joins `response.jsonl` with `source_info.jsonl`, and creates source-grouped train/val/test splits. |
| `src/inference.py` | Runs the language model, finds the answer span, saves answer-only hidden-state artifacts, and computes logit-lens, attention-entropy, and logit-confidence tensors. |
| `src/metrics.py` | Defines the metric formulas and the train-fit functions for Mahalanobis and PCA. |
| `src/evaluate.py` | Computes AUROC, bootstrap CI, Spearman correlation, F1, span-F1, and ECE. |
| `src/component_outputs.py` | Replays the full model and captures `self_attn` and `mlp`/`ffn` outputs for E6 and E7. |
| `src/halueval.py` | Loads HaluEval QA into the same `Sample` format used by the main pipeline. |

### Pipeline files

| File | Role |
|---|---|
| `pipeline/1-infer.py` | Creates artifacts for each split. |
| `pipeline/2-fit.py` | Fits train-only statistics and freezes validation-tuned config into `stats.pt`. |
| `pipeline/3-score.py` | Uses frozen stats to score each sample and compute the strict composite. |
| `pipeline/4-eval.py` | Collapses token scores to sample scores and reports evaluation metrics. |
| `pipeline/plot.py` | Builds the E2 layer-profile plot. |

### Experiment files

| File | Role |
|---|---|
| `scripts/e3_patching.py` | Bidirectional activation patching for causal intervention. |
| `scripts/e4_temporal.py` | Temporal precedence around the first hallucinated token. |
| `scripts/e5_halueval.py` | Frozen zero-shot transfer from RAGTruth to HaluEval QA. |
| `scripts/e6_component_drift.py` | FFN vs attention update-direction drift replay. |
| `scripts/e7_failures.py` | Deterministic failure-case analysis with replay-based explanations. |
| `scripts/e8_sota_gap.py` | Gap analysis against ReDeEP and LUMINA. |

---

## 3. Data Flow: From Raw Files to Scores

### 3.1 Dataset files

The main dataset is RAGTruth:

```text
dataset/ragtruth/response.jsonl
dataset/ragtruth/source_info.jsonl
```

`src/dataset.py::load()` does the join:
- `source_info.jsonl` provides the final prompt string for each `source_id`
- `response.jsonl` provides the generated response and hallucination spans

How the datasets are obtained in this codebase:
- RAGTruth is expected to be placed locally under `dataset/ragtruth/`
- HaluEval QA is loaded through `src/halueval.py`; if the local file is missing, `ensure_qa_dataset()` downloads the official QA file once and stores it under `dataset/halueval/qa_data.json`

Each joined item becomes a `Sample`:

```text
sample_id
source_id
prompt
response
spans
sample_label (optional)
```

### 3.2 Why the split is by `source_id`

`src/dataset.py::split()` groups by `source_id` before splitting.

Reason:
- all responses from the same retrieved source go to the same split
- this avoids near-duplicate leakage between train and test

So the split is not random by sample. It is deterministic by source group.

### 3.3 Prompt + response tokenization

`src/inference.py::InferenceRunner.run()` tokenizes:

```text
full_text = prompt + response
```

It requests `offset_mapping`, then finds the first token whose character start is inside the response:

```text
answer_start = first token index where token_start >= len(prompt)
```

This is how the code separates:
- prompt tokens
- answer tokens

### 3.4 Why hidden states are saved as answer-only

The artifact stores only the answer portion of the hidden states:

```text
hidden_states = selected[:, answer_start:, :]
```

Reason:
- the metrics score hallucination inside the generated answer
- storing only answer tokens saves space
- prompt information is still retained through `context_mean`

### 3.5 How token labels are created

`src/inference.py::_label_tokens()` compares each answer token's character span against the hallucination character spans in the response.

A token gets label `1` if its character range overlaps any hallucination span.

This gives:
- token-level labels: `token_labels`
- sample-level label: `has_hallucination`

---

## 4. Artifact Schema

Each artifact produced by `pipeline/1-infer.py` contains:

| Field | Meaning |
|---|---|
| `sample_id` | Unique sample identifier. |
| `split` | `train`, `val`, or `test`. |
| `has_hallucination` | Sample label: `1` if any hallucination span exists. |
| `token_labels` | Token-level hallucination labels for answer tokens. |
| `hidden_states` | Answer-only hidden states, shape `(n_layers, n_answer_tokens, hidden_dim)`. |
| `context_mean` | Mean prompt representation per saved layer, shape `(n_layers, hidden_dim)`. |
| `logit_lens_per_layer` | Per-layer KL divergence to final-layer distribution, shape `(n_layers, n_answer_tokens)`. |
| `attention_entropy_per_layer` | Per-layer mean-over-head attention entropy, shape `(n_layers, n_answer_tokens)`. |
| `logit_confidence` | Per-token negative log-likelihood of emitted answer tokens, shape `(n_answer_tokens,)`. |
| `answer_start_token_idx` | Always `0` in this compact format because `hidden_states` is answer-only. |
| `answer_end_token_idx` | Number of answer tokens. |

Code path:
- artifact construction: `src/inference.py::InferenceRunner.run`
- save/load format: `src/inference.py::save`, `src/inference.py::load`

---

## 5. Metric Formulas

All token-level metrics are anomaly scores.

General rule:
- higher score = more hallucination-like

### 5.1 Attention entropy

Purpose:
- baseline signal for how diffuse attention is while generating an answer token

Formula:

```text
attention_entropy[l,t] = - sum_k a[l,t,k] log(a[l,t,k])
score[t] = mean_l attention_entropy[l,t]
```

Symbols:
- `l` = saved layer index
- `t` = answer-token index
- `k` = attended key position
- `a[l,t,k]` = attention probability for token `t` at layer `l`

Meaning:
- low entropy = focused attention
- high entropy = diffuse attention

Code:
- raw tensor: `src/inference.py::_compute_attention_entropy`
- token score: `src/metrics.py::attention_entropy`

### 5.2 Logit confidence

Purpose:
- baseline signal for how unlikely the model thinks its own emitted token is

Formula:

```text
logit_confidence[t] = - log p(y_t | prefix)
```

Symbols:
- `y_t` = emitted answer token at position `t`
- `p(y_t | prefix)` = model probability of that token given previous context

Meaning:
- higher value = lower confidence

Code:
- raw tensor: `src/inference.py::_compute_logit_confidence`
- token score: `src/metrics.py::logit_confidence`

### 5.3 Cosine drift

Purpose:
- measures how much the hidden-state direction changes from one answer token to the next

Formula:

```text
cosine_drift[l,t] = 1 - cos(h[l,t], h[l,t-1])
cos(u, v) = (u . v) / (||u|| ||v||)
score[t] = mean_l cosine_drift[l,t]
```

Symbols:
- `h[l,t]` = hidden state at layer `l`, answer token `t`
- `h[l,t-1]` = previous answer token hidden state

Special case:

```text
cosine_drift[l,0] = 0
```

because the first answer token has no previous answer token.

Code:
- raw metric: `src/metrics.py::cosine_drift`
- per-layer version for scoring/fitting: `pipeline/2-fit.py::_cosine_drift_per_layer`, `pipeline/3-score.py::_cosine_drift_per_layer`

### 5.4 Mahalanobis distance

Purpose:
- measures how far a token lies from the train faithful-token distribution

Formula:

```text
mahalanobis[l,t] = sqrt((h - mu)^T Sigma^{-1} (h - mu))
score[t] = mean_l mahalanobis[l,t]
```

Symbols:
- `h` = token hidden state
- `mu` = train-set faithful-token mean for that layer
- `Sigma^{-1}` = inverse covariance for that layer

Meaning:
- large distance = token is off-manifold relative to train faithful tokens

Code:
- fit: `src/metrics.py::fit_mahalanobis`
- per-layer score: `src/metrics.py::mahalanobis_per_layer`
- token score: `src/metrics.py::mahalanobis`

### 5.5 PCA deviation

Purpose:
- measures how much a token lies outside the train faithful PCA subspace

Formula:

```text
diff = h - mu
Proj_PCA(diff) = projection of diff onto top PCA components
pca_deviation[l,t] = || diff - Proj_PCA(diff) ||
score[t] = mean_l pca_deviation[l,t]
```

Symbols:
- `mu` = train faithful mean
- `Proj_PCA(diff)` = reconstruction inside the train-fitted PCA subspace

Meaning:
- large residual = token does not fit the faithful low-dimensional subspace well

Code:
- fit: `src/metrics.py::fit_pca`
- token score: `src/metrics.py::pca_deviation`
- per-layer version used in fit/score: `pipeline/2-fit.py::_pca_deviation_per_layer`, `pipeline/3-score.py::_pca_deviation_per_layer`

### 5.6 Logit lens divergence

Purpose:
- measures how much an intermediate layer disagrees with the final layer about the next-token distribution

Formula:

```text
logit_lens[l,t] = KL(p_final[t] || p_layer_l[t])
KL(P || Q) = sum_i P(i) log(P(i) / Q(i))
score[t] = mean_l logit_lens[l,t]
```

Symbols:
- `p_final[t]` = final-layer token distribution at answer token `t`
- `p_layer_l[t]` = token distribution implied by intermediate layer `l`

Important note:
- in this repo, `logit_lens` is specifically `KL(p_final || p_layer)`

Code:
- raw tensor: `src/inference.py::_compute_logit_lens`
- token score: `src/metrics.py::logit_lens`

### 5.7 CIE top-3 surrogate

Purpose:
- auxiliary metric that reuses Mahalanobis but only on the top-3 validation-selected layers

Formula:

```text
cie_top3[t] = mean_{l in top3} mahalanobis[l,t]
```

Symbols:
- `top3` = three layers with best validation per-layer Mahalanobis AUROC

Important note:
- this is not true causal effect
- it is a surrogate selected in `pipeline/2-fit.py`

Code:
- layer selection: `pipeline/2-fit.py::_pick_cie_top3_layers`
- token score: `src/metrics.py::cie_top3`

### 5.8 Strict composite

Purpose:
- combines the four representation metrics into one frozen sample-level score

Input metrics:
- `cosine_drift`
- `mahalanobis`
- `logit_lens`
- `pca_deviation`

Not included in the strict composite:
- `attention_entropy`
- `logit_confidence`
- `cie_top3`

Formula:

```text
feature_i(sample) = pool_tokens(sum_l alpha_l * per_layer_metric_i[l,:])
z_i = (feature_i - median_i) / max(IQR_i, eps)
signed_z_i = sign_i * z_i
composite = sum_i weight_i * signed_z_i
```

Symbols:
- `alpha_l` = validation-derived nonnegative layer weights inside the selected layer subset
- `median_i`, `IQR_i` = train-only robust scaling statistics
- `sign_i` = validation-derived direction correction
- `weight_i` = validation-derived feature weight

Code:
- fitting rule: `pipeline/2-fit.py::_build_weighted_zscore_composite`
- scoring rule: `pipeline/3-score.py::_composite_score`

---

## 6. Same Formula Reused in E6

E6 uses the same cosine-drift idea, but on component outputs instead of residual hidden states.

Main E6 formula:

```text
update_direction_drift[l,t] = 1 - cos(z[l,t], z[l,t-1])
```

Symbols:
- `z[l,t]` = `self_attn` output or `ffn` output at layer `l`, token `t`

So the structure is the same as cosine drift, but:
- E1/E2 use hidden states `h`
- E6 uses component outputs `z`

Code:
- `src/component_outputs.py::update_direction_drift`

---

## 7. Train-Fit vs Frozen-Test

This is one of the most important parts of the pipeline.

### 7.1 What is fit on train only

From `pipeline/2-fit.py` and `src/metrics.py`:

- Mahalanobis `mu` and `Sigma`
- PCA mean and PCA components
- train robust scaling values for the strict composite (`median`, `IQR`)

These are fit on:

```text
train split only
faithful tokens only
```

Why faithful tokens only:
- the reference distribution is meant to represent normal grounded behavior
- if hallucinated tokens were included, the reference would be contaminated

### 7.2 What validation is used for

Validation does not fit the reference distribution.
Validation is used to choose:

- pooling (`max` or `mean`)
- selected layers
- PCA rank (`4`, `8`, or `16`)
- Mahalanobis regularization
- per-layer weights inside selected layer subsets
- sign correction
- feature weights in the composite
- `cie_top3` layers

### 7.3 What stays frozen at test time

At test time, `stats.pt` is loaded and reused unchanged.

So test uses:
- frozen `mu`
- frozen `Sigma^{-1}`
- frozen PCA basis
- frozen pooling/layer choices
- frozen layer weights
- frozen signs and composite weights

Code that consumes frozen stats:
- `pipeline/3-score.py`
- `scripts/e5_halueval.py`
- `scripts/e7_failures.py`

### 7.4 Why this matters

This prevents test leakage.

The detector is allowed to:
- learn the faithful geometry from train
- choose hyperparameters on validation

It is not allowed to:
- refit Mahalanobis on test
- refit PCA on test
- change composite weights on test

---

## 8. Pipeline Steps for E1 and E2

## 8.1 Step 1: Inference

File:
- `pipeline/1-infer.py`

Main work:
- load RAGTruth
- split by `source_id`
- run the model
- save one artifact per sample

Key formulas used here:
- answer span detection through `offset_mapping`
- `logit_lens`
- `attention_entropy`
- `logit_confidence`

Main output:

```text
outputs/artifacts/{train,val,test}/{sample_id}.pt
```

## 8.2 Step 2: Fit

File:
- `pipeline/2-fit.py`

Main work:
- fit Mahalanobis on train faithful tokens
- fit PCA on train faithful tokens
- tune layer/pooling/PCA-rank choices on validation
- build the strict weighted z-score composite
- save everything to `stats.pt`

Main output:

```text
outputs/stats.pt
```

Stored keys include:
- `mahalanobis`
- `pca`
- `cie_top3_layers`
- `metric_layers`
- `metric_pooling`
- `metric_layer_weights`
- `pca_components_selected`
- `composite_train_median`
- `composite_train_iqr`
- `composite_signs`
- `composite_weights`

## 8.3 Step 3: Score

File:
- `pipeline/3-score.py`

Main work:
- load artifacts
- compute token-level metrics
- compute `sample_features`
- compute `composite_sample_score`

Main output:

```text
outputs/scores_test/{sample_id}.pt
```

Saved fields include:
- token-level metrics
- `sample_features`
- `composite_sample_score`

## 8.4 Step 4: Evaluate

File:
- `pipeline/4-eval.py`

Main work:
- reduce token scores to one sample score
- compute evaluation metrics

Token aggregation:

```text
sample_score = max_t score[t]
```

or

```text
sample_score = mean_t score[t]
```

For the strict composite:
- if `composite_sample_score` exists, it is used directly

---

## 9. Evaluation Formulas

These functions live in `src/evaluate.py`.

### 9.1 AUROC

Formula idea:
- AUROC is the normalized Wilcoxon-Mann-Whitney statistic
- it measures how often a positive sample ranks above a negative sample

Code:
- `src/evaluate.py::auroc`

### 9.2 Bootstrap confidence interval

Procedure:
- resample `(label, score)` pairs with replacement
- compute AUROC many times
- take percentile bounds

Code:
- `src/evaluate.py::bootstrap_ci`

### 9.3 Spearman correlation

Formula idea:
- rank labels
- rank scores
- compute correlation between ranks

Code:
- `src/evaluate.py::spearman`

### 9.4 F1 and span-F1

`f1`:
- plain binary F1 after thresholding sample scores

`f1_span`:
- token span overlap F1
- mainly for token-level runs, not the main sample-level E1 table

Code:
- `src/evaluate.py::f1`
- `src/evaluate.py::f1_span`

### 9.5 ECE

Procedure:
- min-max normalize scores to `[0,1]`
- bin them
- compare predicted confidence vs observed hallucination rate

Code:
- `src/evaluate.py::ece`

---

## 10. E1/E2 Current Results

The current submission copy stores the main metric-style AUROC rows in:
- `NLP-sub/outputs/ablation_composite.md`
- `NLP-sub/outputs/ablation_composite.json`

Current test AUROC table:

| Metric / Composite | Test AUROC |
|---|---:|
| Attention entropy (Baseline 1) | 0.6106 |
| Logit confidence (Baseline 2) | 0.5497 |
| Cosine drift | 0.5989 |
| Mahalanobis distance | 0.5389 |
| Logit lens divergence | 0.5490 |
| PCA deviation | 0.5732 |
| CIE top-3 layers | 0.5138 |
| Full composite (current baseline composite) | 0.6511 |

Interpretation:
- in the current stored outputs, the strict composite beats every individual metric
- `attention_entropy` is the strongest standalone baseline among the main rows

### E2 layer profile

E2 is generated by:
- `pipeline/plot.py`

Main output files:
- `NLP-sub/outputs/e2/layer_profile_verified.png`
- `NLP-sub/outputs/e2/layer_profile_from_script.png`

This plot computes per-layer AUROC for each metric over the saved layer slice.

---

## 11. Experiment 3: Activation Patching

File:
- `scripts/e3_patching.py`

Goal:
- test whether replacing activations changes the next-token probability in a causal way

### 11.1 Pair construction

The script:
- loads test samples
- groups by `source_id`
- builds faithful/hallucinated pairs from the same source group

### 11.2 Patch position

The patch position is one token before the first hallucinated token in the hallucinated sample:

```text
P = first_hal - 1
```

Readout position:

```text
target_position = P + 1
```

This keeps patching inside the answer region, not the prompt boundary.

### 11.3 Directions

Two directions are evaluated:

```text
faith_to_hal
hal_to_faith
```

### 11.4 Main effect size

Formula:

```text
delta_logp = logp_patched(target_token) - logp_clean(target_token)
```

Meaning:
- negative value = patched donor activation suppresses the recipient's natural continuation

### 11.5 Significance test

The script reports:
- two-sided Wilcoxon signed-rank p-value
- one-sided `p_less` as a diagnostic

Criticality in the current script is based on:
- the configured criticality rule stored in `NLP-sub/outputs/e3/cie_bidirectional.json`
- current run uses `critical_rule = relative`

### 11.6 Current E3 result

Source files:
- `NLP-sub/outputs/e3_old/cie_bidirectional.md`
- `NLP-sub/outputs/e3_old/cie_bidirectional.json`
- `NLP-sub/outputs/e3_old/cie_bidirectional.png`

Current rubric table:

| Component | CIE faith->hal | CIE hal->faith | Critical? |
|---|---:|---:|---|
| early_attn | -1.0991 | -1.0782 | yes |
| mid_ffn | -0.0550 | -0.0299 | yes |
| late_ffn | -0.2732 | -0.1608 | yes |
| copying_heads | -0.0807 | -0.0358 | yes |

Interpretation:
- all four buckets show negative mean `delta_logp`
- all four buckets are marked significant in both directions in the stored `e3_old` run
- the strongest causal effect in this stored run is early attention
- the weakest effect is the copying-head proxy, but it still passes significance in the stored `e3_old` summary

---

## 12. Experiment 4: Temporal Precedence

File:
- `scripts/e4_temporal.py`

Goal:
- check whether a metric rises before the hallucination starts

### 12.1 Onset alignment

The script finds:

```text
t = first hallucinated token
```

Then reads metrics at:

```text
t-3, t-2, t-1, t, t+1
```

### 12.2 Statistical test

For offsets before onset:

```text
H1: metric(t-k) > metric(t)
```

For the post-onset control:

```text
H1: metric(t+1) < metric(t)
```

Test used:
- one-sided Mann-Whitney U

### 12.3 Current E4 result

Source files:
- `NLP-sub/outputs/e4/temporal.csv`
- `NLP-sub/outputs/e4/temporal.json`
- `NLP-sub/outputs/e4/temporal.png`
- `NLP-sub/outputs/e4/TEMPORAL_PRECEDENCE_NOTE.md`

Assignment-style table currently documented in the note:

Important note:
- the raw E4 script writes the main six metrics to `temporal.csv` / `temporal.json`
- the `cie_top3` surrogate row shown below comes from the companion note, which extends the table for reporting consistency

| Position | Cosine drift | Mahalanobis | Logit lens | PCA dev. | CIE (top3 surrogate) |
|---|---:|---:|---:|---:|---:|
| t-3 | 0.2668 | 37.9664 | 3.6684 | 87.4574 | 37.7385 |
| t-2 | 0.2872 | 34.0917 | 3.5820 | 80.0704 | 35.6801 |
| t-1 | 0.2821 | 31.4590 | 4.2375 | 72.2008 | 31.7477 |
| t (onset) | 0.2729 | 36.5722 | 3.8713 | 85.6095 | 36.3971 |
| t+1 | 0.2708 | 38.1772 | 3.5158 | 88.1041 | 38.2783 |

Current interpretation:
- `cosine_drift` peaks at `t-2`
- `logit_lens` peaks at `t-1`
- the current note treats this as meeting the rubric condition because an early peak exists and MWU is reported

---

## 13. Experiment 5: HaluEval Transfer

File:
- `scripts/e5_halueval.py`

Goal:
- apply frozen RAGTruth statistics to HaluEval QA without refitting

### 13.1 HaluEval loading

`src/halueval.py`:
- downloads or reads HaluEval QA
- creates one faithful sample and one hallucinated sample per item

### 13.2 No-refit rule

E5 reuses frozen RAGTruth:
- Mahalanobis stats
- PCA basis
- composite settings

No retraining is done on HaluEval.

### 13.3 Transfer-drop formula

Formula:

```text
drop = AUROC_RAGTruth - AUROC_HaluEval
```

Meaning:
- larger positive drop = worse transfer

### 13.4 Most brittle metric

Definition:

```text
most brittle metric = metric with largest positive drop
```

### 13.5 Current E5 result

Source:
- `NLP-sub/outputs/e5/E5_TRANSFER_EXPLANATION.md`

| Metric | AUROC RAGTruth | AUROC HaluEval | Drop |
|---|---:|---:|---:|
| Cosine drift | 0.6244 | 0.8345 | -0.2102 |
| Mahalanobis distance | 0.5190 | 0.7006 | -0.1815 |
| Logit lens divergence | 0.5917 | 0.4337 | +0.1580 |
| PCA deviation | 0.5764 | 0.7186 | -0.1421 |
| Full composite | 0.6508 | 0.6450 | +0.0058 |

Current interpretation:
- most brittle metric = `logit_lens`
- composite transfer is still above the rubric threshold at `0.6450`

---

## 14. Experiment 6: FFN vs Attention Decomposition

File:
- `scripts/e6_component_drift.py`

Goal:
- compare `self_attn` and `ffn` update-direction drift across layer ranges

### 14.1 Replay hooks

The script reloads the model and captures:
- `self_attn` outputs
- `mlp` outputs

from every layer using forward hooks in `src/component_outputs.py::capture_component_outputs`.

### 14.2 Drift formula

Formula:

```text
drift[l,t] = 1 - cos(z[l,t], z[l,t-1])
```

where `z` is either:
- `self_attn` output
- `ffn` output

### 14.3 Sample reduction

The current E6 script reduces each sample using max pooling over answer tokens inside each range.

It then computes:
- per-layer AUROC
- range-pooled AUROC
- class gap = mean drift on hallucinated samples minus mean drift on faithful samples

### 14.4 Current E6 result

Source:
- `NLP-sub/outputs/e6/component_drift.csv`
- `NLP-sub/outputs/e6/component_drift.json`
- `NLP-sub/outputs/e6/component_drift.png`
- `NLP-sub/outputs/e6/README.md`

| Component | Range | AUROC | Mean hall. | Mean faithful | Class gap |
|---|---|---:|---:|---:|---:|
| self_attn | early | 0.5444 | 1.1084 | 1.0946 | +0.0137 |
| self_attn | mid | 0.5903 | 1.1510 | 1.1018 | +0.0492 |
| self_attn | late | 0.5863 | 1.0785 | 1.0176 | +0.0609 |
| ffn | early | 0.5852 | 1.1496 | 1.1318 | +0.0178 |
| ffn | mid | 0.6191 | 1.0900 | 1.0530 | +0.0370 |
| ffn | late | 0.5920 | 1.2096 | 1.1806 | +0.0291 |

Current interpretation:
- best AUROC is `mid ffn`
- largest class gap is `late self_attn`
- current stored claim is `mixed`

---

## 15. Experiment 7: Failure Cases

File:
- `scripts/e7_failures.py`

Goal:
- explain specific detector failures with case-specific replay evidence

### 15.1 Case selection

The script selects three deterministic cases:

1. false negative
2. false positive
3. metric disagreement

Selection uses:
- `composite_sample_score`
- variance across normalized `sample_features`
- distinct `source_id`

### 15.2 Replay explanation

For each case, the script:
- reloads the original sample metadata
- replays the model
- captures `self_attn` and `ffn` outputs
- computes range-based drift summary

### 15.3 Current E7 result

Source:
- `NLP-sub/outputs/e7/failures.md`
- `NLP-sub/outputs/e7/failures.json`
- `NLP-sub/outputs/e7/failure_traces/*.png`

Current selected cases:

| Case type | Sample | Source | Main reading |
|---|---|---|---|
| False negative | `12310` | `14368` | Detector under-reacted because `cosine_drift` contributed least. |
| False positive | `1289` | `15808` | Detector over-reacted mainly because `cosine_drift` was high on a faithful sample. |
| Metric disagreement | `3574` | `12062` | `cosine_drift` was high while `mahalanobis` was low, so metrics conflicted. |

Current replay interpretation:
- all three stored cases peak in the early range
- all three lean toward `ffn`/`mlp`

---

## 16. Experiment 8: SOTA Gap Analysis

File:
- `scripts/e8_sota_gap.py`

Goal:
- compare the baseline and current composite against ReDeEP and LUMINA

### 16.1 Gap-closed formula

Formula:

```text
gap_closed = (ours - baseline) / (sota - baseline)
```

Symbols:
- `baseline` = `attention_entropy` AUROC
- `ours` = current composite AUROC
- `sota` = reference AUROC from ReDeEP or LUMINA

### 16.2 50 percent target

Formula:

```text
threshold_50pct = baseline + 0.5 * (sota - baseline)
delta_to_50pct = threshold_50pct - ours
```

### 16.3 Current E8 result

Source:
- `NLP-sub/outputs/e8/sota_gap.csv`
- `NLP-sub/outputs/e8/sota_gap.json`
- `NLP-sub/outputs/e8/sota_gap.md`

| Target | Baseline | Ours | SOTA | Abs gap | Gap closed | 50% target | Delta to 50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReDeEP | 0.6106 | 0.6511 | 0.8181 | 0.1670 | 19.51% | 0.7144 | 0.0633 |
| LUMINA | 0.6106 | 0.6511 | 0.8569 | 0.2058 | 16.44% | 0.7338 | 0.0827 |

Current interpretation:
- the composite is better than the baseline
- but it is still below the 50% gap-closed threshold for both SOTA references

---

## 17. End-to-End Output Map

Main result outputs in the submission copy:

| Experiment | Output folder | Main files |
|---|---|---|
| E1/E2 | `NLP-sub/outputs/e2` and `NLP-sub/outputs/ablation_*` | layer-profile plots and main AUROC tables |
| E3 | `NLP-sub/outputs/e3` | `cie_bidirectional.csv`, `.json`, `.md`, `.png` |
| E4 | `NLP-sub/outputs/e4` | `temporal.csv`, `.json`, `.png`, `TEMPORAL_PRECEDENCE_NOTE.md` |
| E5 | `NLP-sub/outputs/e5` | `E5_TRANSFER_EXPLANATION.md` |
| E6 | `NLP-sub/outputs/e6` | `component_drift.csv`, `.json`, `.png`, `README.md` |
| E7 | `NLP-sub/outputs/e7` | `failures.md`, `failures.json`, `failure_traces/` |
| E8 | `NLP-sub/outputs/e8` | `sota_gap.csv`, `.json`, `.md` |

---

## 18. Composite Ablations Appendix

These are extra analyses. They are not the core E1-E8 pipeline.

Main sources:
- `NLP-sub/outputs/ablation_composite.md`
- `NLP-sub/outputs/ablation_composite.json`
- `NLP-sub/outputs/ablation_logreg.json`

### 18.1 What each variant means

`zscore-4`
- uses the 4 representation metrics only:
  - `cosine_drift`
  - `mahalanobis`
  - `logit_lens`
  - `pca_deviation`
- standardizes them and combines them with a weighted z-score rule

`zscore-6`
- same as `zscore-4`
- adds two baseline-derived features:
  - `attn_entropy_max`
  - `logit_conf_mean`

`logreg-4`
- trains logistic regression on the 4-feature set

`logreg-6`
- trains logistic regression on the 6-feature set

### 18.2 Current ablation table

| Variant | Val AUROC | Test AUROC | Delta vs current composite |
|---|---:|---:|---:|
| zscore-4 | 0.6413 | 0.6524 | +0.0013 |
| zscore-6 | 0.6766 | 0.6906 | +0.0395 |
| logreg-4 | 0.6495 | 0.6572 | +0.0060 |
| logreg-6 | 0.7033 | 0.7132 | +0.0620 |

Current best ablation:
- `logreg-6` with test AUROC `0.7132`

Standalone 4-feature logistic regression sanity run from `ablation_logreg.json`:

| Variant | Val AUROC | Test AUROC |
|---|---:|---:|
| logreg-4 (standalone script) | 0.6485 | 0.6569 |

### 18.3 Why these are separate from the main pipeline

The main submission pipeline is:
- infer
- fit
- score
- evaluate

The ablation scripts are extra checks to see whether:
- adding the baselines helps
- replacing weighted z-score with logistic regression helps

So they are useful for analysis, but they are not the core frozen E1-E8 implementation.

---

## 19. Final Practical Summary

If a teammate needs the shortest accurate explanation:

1. `pipeline/1-infer.py` creates answer-only hidden-state artifacts.
2. `pipeline/2-fit.py` learns train-only geometry (`mu`, `Sigma`, PCA) and freezes validation-selected composite settings into `stats.pt`.
3. `pipeline/3-score.py` computes token metrics and one sample-level composite score per sample.
4. `pipeline/4-eval.py` reports AUROC, CI, Spearman, F1, and ECE.
5. E3-E8 are analysis scripts built on top of those frozen outputs or replayed model internals.

The most important rule across the whole project is:

```text
train fits the reference statistics
validation chooses the configuration
test only reads frozen settings
```

That rule is what keeps the reported results leakage-safe.
