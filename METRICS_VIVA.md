# Track B Metrics Viva Guide

This file is not a generic README. It is a **defense document** for Track B.
It answers four questions:

1. What is the assignment actually asking us to compute?
2. What does each metric mean mathematically and mechanistically?
3. Where is each metric computed in the code, and where is it actually used?
4. What is correct to claim in viva/report, and what would be misleading?


---

## 1. What Track B Is Actually Asking

From the assignment PDF, Track B is asking for **hidden-state-based hallucination detection** first, and **mechanistic decomposition/causal analysis** second.

### Track B interpretation table

| Experiment | What the assignment wants                                                                                | Proper object of analysis                                                         | What this repo currently does                                                                                                                                                                                                                   |
| ---------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E1/E2      | Build 5 representation metrics + 2 baselines, then a composite on RAGTruth test; show layer localization | Primarily **saved hidden-state representations** plus auxiliary per-layer tensors | Correct overall direction: uses hidden states for `cosine_drift`, `mahalanobis`, `pca_deviation`, surrogate `cie_top3`; uses saved per-layer KL and attention entropy; builds strict composite in `pipeline/2-fit.py` and `pipeline/3-score.py` |
| E3         | Activation patching / causal intervention in **both directions**                                         | **Internal activations**, not just saved artifact scores                          | Partially implemented: `scripts/e3_patching.py` performs real patching, but current script is still one-directional (`faithful -> hallucinated`)                                                                                                |
| E4         | Temporal precedence of Track B representation metrics                                                    | **Per-token metric traces** aligned to first hallucinated token                   | Implemented in `scripts/e4_temporal.py`; currently includes baselines too, but the Track B report should foreground the 5 representation metrics                                                                                                |
| E5         | HaluEval zero-shot transfer with **no refit** of `mu`, `Sigma`, or PCA                                   | Frozen RAGTruth stats applied to a new domain                                     | Implemented in `scripts/e5_halueval.py`                                                                                                                                                                                                         |
| E6         | FFN vs attention decomposition                                                                           | **`self_attn` and `mlp` outputs**, replayed from the model                        | Implemented separately in `scripts/e6_component_drift.py` via `src/component_outputs.py`                                                                                                                                                        |
| Live demo  | Extract hidden states and compute all 5 representation metrics on unseen input                           | End-to-end hidden-state pipeline                                                  | Main path is consistent with this requirement                                                                                                                                                                                                   |

### The most important interpretation

For **E1/E2/E5 and the live demo**, the assignment is asking for a detector built on the **representation stream**:

- saved hidden states
- per-layer distance/divergence signals derived from those hidden states
- token-level scores -> sample-level scores -> AUROC

For **E3 and E6**, the assignment moves deeper into mechanism analysis:

- patching internal activations
- decomposing FFN output vs attention output

So if your question is:

> "Should we have used only `self_attn` / `mlp` instead of hidden states?"

The answer is:

- **No**, not for the main Track B detector.
- **Yes**, but only for the specific analysis experiments that explicitly ask for internal component decomposition.

Using only `self_attn` or only `mlp` for the main detector would have been a **different project** from what E1/E2 are asking.

---

## 2. What a Score Means in This Repo

### Score hierarchy

| Level                               | What it is                                                                           | Example                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Token-level metric                  | One anomaly score per answer token                                                   | `cosine_drift[t]`, `mahalanobis[t]`, `logit_lens[t]`             |
| Sample-level pooled metric          | One score per response after aggregating tokens                                      | `max_t cosine_drift[t]` in `pipeline/4-eval.py`                  |
| Sample feature for strict composite | One frozen scalar per metric after layer selection + token pooling + layer weighting | `sample_features["mahalanobis"]` in `pipeline/3-score.py`        |
| Final strict composite score        | One scalar per sample after robust scaling + weighting                               | `composite_sample_score`                                         |
| AUROC                               | Ranking quality over many samples                                                    | Does the metric rank hallucinated responses above faithful ones? |

### What high and low scores mean

| Quantity                 | Meaning of a higher value                             |
| ------------------------ | ----------------------------------------------------- |
| Raw token metric score   | "More suspicious / more hallucination-like" by design |
| Sample pooled metric     | Response looks more hallucination-like                |
| `composite_sample_score` | Stronger final hallucination signal                   |
| AUROC > 0.5              | Metric direction is aligned with hallucination        |
| AUROC < 0.5              | Metric is inverted on that dataset / setting          |
| AUROC ~= 0.5             | Metric is not separating the two classes              |

### Why AUROC matters more than the raw scale

The assignment is primarily scored by **AUROC**, not by the absolute value of the raw score.

That means:

- the **ordering** of hallucinated vs faithful samples matters most
- two metrics can have very different magnitudes and still be comparable via AUROC
- a metric with huge raw values can still be bad if it ranks samples poorly

This is why `composite_sample_score` is meaningful mainly as a **ranking signal**, not as an interpretable calibrated probability.

---

## 3. Core Concepts You Must Be Able to Explain

### Definitions table

| Term                  | What it means here                                                                                      | Why it matters                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Hidden state          | The transformer representation vector at a layer and token position                                     | Main object used by Track B metrics                                       |
| Attention entropy `H` | Entropy of the attention distribution over keys for an answer token                                     | Baseline for "diffuse vs focused" attention                               |
| KL divergence         | Distribution mismatch between the final token distribution and an intermediate-layer token distribution | Core of `logit_lens` in this repo                                         |
| Mahalanobis distance  | Distance from a reference Gaussian-like cloud after covariance normalization                            | Detects whether a token is off-manifold relative to train faithful tokens |
| PCA deviation         | Norm of the component outside a train-fitted low-dimensional faithful subspace                          | Detects off-subspace behavior                                             |
| AUROC                 | Probability that a randomly chosen positive ranks above a randomly chosen negative                      | Main evaluation criterion                                                 |

### What is entropy `H`?

Formula:

![alt text](image.png)


Interpretation:

- low entropy = concentrated distribution
- high entropy = diffuse distribution

In this repo, `attention_entropy` means:

- take one attention row for one answer token
- treat it as a probability distribution over key positions
- measure how spread out it is

That is why the code computes:
```text
entropy = -(att * log(att)).sum(...)
```

Source:
- `src/inference.py::_compute_attention_entropy (208-226)`

### What is KL divergence?

Formula:

![alt text](image-1.png)

Interpretation:

- `KL(P || Q)` is small if `Q` is close to `P`
- it grows when `Q` puts probability mass in a different shape than `P`

In this repo, `logit_lens` means:

- `P = p_final`, the final-layer token distribution
- `Q = p_layer_l`, the distribution implied by intermediate layer `l`

So the metric is:
```text
KL(p_final || p_layer_l)
```

This is **not** the same thing as:

- cross-entropy
- confidence drop
- ReDeEP PKS

Source:
- `src/inference.py::_compute_logit_lens (171-205)`

---

## 4. Main Metrics: What They Are, Why We Use Them, Where They Live

## 4.1 Baselines

These are included because the assignment explicitly wants two baselines. They are **comparators**, not strict composite inputs.

### `attention_entropy`

#### Why use it?

Because hallucination often coincides with **less anchored attention**:

- if the model is strongly grounded, attention may focus sharply on specific positions
- if the model is uncertain or drifting, attention may become more diffuse

It is a simple, interpretable baseline for "how focused is the model right now?"

#### Formula

```text
entropy[l,t] = -sum_k a[l,t,k] log(a[l,t,k])
score[t] = mean_l entropy[l,t]
```

#### Code path

| Stage                       | File / function                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------- |
| Raw tensor creation         | `src/inference.py::_compute_attention_entropy (208-226)`                            |
| Final raw metric reduction  | `src/metrics.py::attention_entropy (118-120)`                                       |
| Sample-level evaluation use | `pipeline/4-eval.py::_metric_score (56-65)` and `pipeline/4-eval.py::main (68-109)` |

#### Meaningful use in this repo

The meaningful use is:

- compute per-token entropy
- aggregate per sample in `pipeline/4-eval.py`
- compare it against the other metrics as **Baseline 1**

#### Caveat

Do **not** say:

> "This is attention entropy over the whole model"

unless you actually ran `--layers all`.

It is entropy over the **stored layer slice**.

---

### `logit_confidence`

#### Why use it?

Because a model sometimes emits hallucinated tokens with low internal confidence.

This is the simplest "how surprised is the model by its own token?" baseline.

#### Formula

```text
score[t] = -log p(y_t | prefix up to t-1)
```

#### Code path

| Stage                       | File / function                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------- |
| Raw tensor creation         | `src/inference.py::_compute_logit_confidence (229-248)`                             |
| Final raw metric reduction  | `src/metrics.py::logit_confidence (123-125)`                                        |
| Sample-level evaluation use | `pipeline/4-eval.py::_metric_score (56-65)` and `pipeline/4-eval.py::main (68-109)` |

#### Meaningful use in this repo

The meaningful use is as **Baseline 2** only.

#### Caveat

It is called `logit_confidence` in the repo, but mathematically it is a **token NLL**.

So in viva, say:

> "This implementation uses negative log-likelihood as a confidence proxy."

not:

> "This is a confidence-drop metric relative to retrieved evidence."

---

## 4.2 Representation Metrics

These are the main Track B detector family. They are meant to operate on the **representation stream**.

### `cosine_drift`

#### Why use it?

A grounded continuation should evolve relatively smoothly in representation space.
A hallucinated continuation may show a sharper directional jump as the model moves
away from the grounded trajectory.

So `cosine_drift` tries to detect **trajectory instability**.

#### Formula

```text
drift[l,t] = 1 - cos(h[l,t], h[l,t-1])     for t > 0
drift[l,0] = 0
score[t] = mean_l drift[l,t]
```

#### Code path

| Stage                                       | File / function                                        |
| ------------------------------------------- | ------------------------------------------------------ |
| Raw token metric                            | `src/metrics.py::cosine_drift (69-84)`                 |
| Per-layer feature path for strict composite | `pipeline/3-score.py::_cosine_drift_per_layer (40-48)` |
| Sample feature construction                 | `pipeline/3-score.py::_sample_feature (79-97)`         |
| Sample-level evaluation use                 | `pipeline/4-eval.py::_metric_score (56-65)`            |

#### Meaningful use in this repo

There are two meaningful uses:

1. raw row in the evaluation table
2. strict composite input after validation-selected layer/pooling choices

#### Caveat

The first token is always zero. This is deliberate, not a bug.

---

### `mahalanobis`

#### Why use it?

This is the most classical "distance from normal" detector in representation space.

If faithful tokens form a cloud in hidden-state space, then a hallucinated token
may sit unusually far from that cloud. Mahalanobis is stronger than plain Euclidean
distance because it respects the covariance geometry of the faithful distribution.

That means:

- moving in a high-variance direction is less surprising
- moving in a low-variance direction is more surprising

#### Formula

```text
Mahalanobis[l,t] = sqrt((h - mu)^T Sigma^-1 (h - mu))
score[t] = mean_l Mahalanobis[l,t]
```

#### Code path

| Stage                                    | File / function                                  |
| ---------------------------------------- | ------------------------------------------------ |
| Train faithful-token accumulation        | `src/metrics.py::_accumulate_moments (178-203)`  |
| Fit `mu` and `Sigma^-1`                  | `src/metrics.py::fit_mahalanobis (206-221)`      |
| Raw per-layer distance                   | `src/metrics.py::mahalanobis_per_layer (92-104)` |
| Raw token metric                         | `src/metrics.py::mahalanobis (87-89)`            |
| Validation tuning / regularization sweep | `pipeline/2-fit.py::main (531-650)`              |
| Strict composite feature use             | `pipeline/3-score.py::main (116-237)`            |

#### Meaningful use in this repo

This metric is meaningful in **three** places:

1. as an individual row in the final metric table
2. as part of the strict composite
3. as the base for the surrogate `cie_top3` metric

#### Caveat

The assignment is strict here:

- `mu_l` and `Sigma_l` must be estimated on **train only**

The repo's current implementation is aligned with that requirement.

---

### `pca_deviation`

#### Why use it?

The idea is not just "distance from center". It is:

> faithful tokens may lie near a lower-dimensional subspace, and hallucinated tokens may deviate outside that subspace.

So this metric tests **off-subspace behavior**, not just overall displacement.

#### Formula

```text
diff = h - mu
score[l,t] = ||diff - Proj_PCA(diff)||
score[t] = mean_l score[l,t]
```

#### Code path

| Stage                               | File / function                                         |
| ----------------------------------- | ------------------------------------------------------- |
| Train faithful-token accumulation   | `src/metrics.py::_accumulate_moments (178-203)`         |
| Fit PCA basis                       | `src/metrics.py::fit_pca (224-245)`                     |
| Raw token metric                    | `src/metrics.py::pca_deviation (128-138)`               |
| Per-layer feature path              | `pipeline/3-score.py::_pca_deviation_per_layer (51-61)` |
| Validation tuning / PCA rank choice | `pipeline/2-fit.py::main (531-650)`                     |

#### Meaningful use in this repo

Two meaningful uses:

1. raw evaluation row
2. strict composite input with validation-selected PCA rank

#### Caveat

The basis is fit once on train data. The rank is selected on validation.
Do not say the PCA basis is re-fit separately per rank on validation/test.

---

### `logit_lens`

#### Why use it?

This metric is trying to detect **internal disagreement across depth**.

If an intermediate layer already "wants" a very different token distribution from
what the final layer eventually emits, that token may lie on an unstable internal
trajectory.

This is why it is a representation-side analogue of a mechanistic instability probe.

#### Formula

```text
KL_l[t] = KL(p_final[t] || p_layer_l[t])
score[t] = mean_l KL_l[t]
```

#### Code path

| Stage                                             | File / function                                   |
| ------------------------------------------------- | ------------------------------------------------- |
| Project intermediate layer through norm + LM head | `src/inference.py::_compute_logit_lens (171-205)` |
| Raw token metric                                  | `src/metrics.py::logit_lens (141-148)`            |
| Strict composite feature use                      | `pipeline/3-score.py::main (116-237)`             |

#### Meaningful use in this repo

The meaningful use is the **saved per-layer KL tensor** produced at inference time.

That saved tensor is then used in two ways:

1. raw `logit_lens` metric row
2. strict composite sample feature

#### Caveat

This implementation is **not ReDeEP PKS**.

That matters because if someone asks:

> "Are you implementing ReDeEP's knowledge score?"

the correct answer is:

> "No. Our current `logit_lens` is a layerwise KL-to-final distribution metric."

---

### `cie_top3`

#### Why use it here?

Because the rubric wants a CIE-style row, but the main E1/E2 pipeline is not actually
running activation patching during score generation.

So this repo uses a transparent surrogate:

- rank layers by validation AUROC using Mahalanobis
- keep the top 3
- average Mahalanobis over those layers

This is trying to capture:

> "which layers appear most informative according to the validation signal?"

#### Formula

```text
cie_top3[t] = mean_{l in top3} Mahalanobis[l,t]
```

#### Code path

| Stage                             | File / function                                      |
| --------------------------------- | ---------------------------------------------------- |
| Choose top-3 layers on validation | `pipeline/2-fit.py::_pick_cie_top3_layers (172-186)` |
| Score tokens using those layers   | `src/metrics.py::cie_top3 (107-115)`                 |

#### Meaningful use in this repo

Only as an **auxiliary reported row**.

#### Caveat

Do **not** call this "real causal effect" in your report.

Correct phrasing:

> "CIE top-3 layers is currently a Mahalanobis-based surrogate pending full activation-patching integration."

---

## 4.3 The Strict Composite

This is the most important metric in the repo because the assignment's main mark band depends on it.

### Why use a composite at all?

Because each representation metric captures a different failure mode:

- `cosine_drift`: trajectory instability
- `mahalanobis`: off-manifold distance
- `logit_lens`: cross-depth disagreement
- `pca_deviation`: off-subspace deviation

The composite assumes these signals are **complementary**.

### What the strict composite is **not**

It is **not**:

- a learned classifier
- logistic regression
- a token-level average of every metric in `src/metrics.py`
- a composite that includes baselines
- a composite that includes `cie_top3`

### Which features go into it?

| Feature             | Included? | Why                                       |
| ------------------- | --------- | ----------------------------------------- |
| `cosine_drift`      | Yes       | Complementary representation-drift signal |
| `mahalanobis`       | Yes       | Off-manifold signal                       |
| `logit_lens`        | Yes       | Internal depth disagreement               |
| `pca_deviation`     | Yes       | Off-subspace signal                       |
| `attention_entropy` | No        | Baseline comparator only                  |
| `logit_confidence`  | No        | Baseline comparator only                  |
| `cie_top3`          | No        | Auxiliary surrogate only                  |

Source:
- `pipeline/2-fit.py (46-47)`

### How one sample feature is built

For each composite feature:

1. pick the validation-selected layers
2. optionally weight those layers
3. reduce the per-layer tensor to one token-level series
4. pool tokens with `max` or `mean`

Formula:
```text
feature_i(sample) = pool_tokens(sum_l alpha_l * per_layer_metric[l,:])
```

Source:
- `pipeline/3-score.py::_sample_feature (79-97)`

### How the final composite score is built

Formula:
```text
z_i = (feature_i - median_i) / max(IQR_i, eps)
signed_z_i = sign_i * z_i
composite = sum_i weight_i * signed_z_i
```

Where each term comes from:

| Quantity   | Meaning                           | Source                                                          |
| ---------- | --------------------------------- | --------------------------------------------------------------- |
| `median_i` | Train-only robust center          | `pipeline/2-fit.py::_train_feature_robust_stats (254-314)`      |
| `IQR_i`    | Train-only robust scale           | `pipeline/2-fit.py::_train_feature_robust_stats (254-314)`      |
| `sign_i`   | Validation-derived direction fix  | `pipeline/2-fit.py::_build_weighted_zscore_composite (317-378)` |
| `weight_i` | Validation-derived simplex weight | `pipeline/2-fit.py::_build_weighted_zscore_composite (317-378)` |

### Which code path is the real one?

This is important.

The real strict composite path is:

- fit/tune: `pipeline/2-fit.py`
- score: `pipeline/3-score.py`
- evaluate: `pipeline/4-eval.py`

The function:

- `src/metrics.py::composite (151-173)`

is **not** the strict composite that the current pipeline evaluates.

If you say otherwise in viva, you will be wrong.

---

## 5. What Is Correct to Claim About the Pipeline

### Main path vs analysis path

| Path                                                         | Uses                 | Proper role                                |
| ------------------------------------------------------------ | -------------------- | ------------------------------------------ |
| `hidden_states` + saved per-layer tensors                    | E1/E2/E5 + live demo | Main Track B detector path                 |
| Activation patching (`scripts/e3_patching.py`)               | E3                   | Causal analysis path                       |
| `self_attn` / `mlp` replay (`scripts/e6_component_drift.py`) | E6                   | Decomposition / localization analysis path |

### So did we take the wrong path?

For the **main detector**, no.

Using `hidden_states` was the right path because the assignment explicitly asks for:

- cosine drift
- Mahalanobis distance
- logit lens divergence
- PCA deviation
- CIE top-3 layers

Those are all naturally implemented on the **representation stream**.

For **E6**, hidden states alone are not enough.

That is why the repo adds:

- `src/component_outputs.py`
- `scripts/e6_component_drift.py`

to replay and capture:

- `self_attn` outputs
- `mlp` outputs

So the correct interpretation is:

- **main detector** = hidden-state path
- **mechanistic decomposition** = self-attn / FFN replay path

### Are we using all layers or not?

For the main pipeline:

- **No**, not necessarily.
- It uses the layer slice resolved by `src/inference.py::resolve_layers (40-53)`.
- In practice that is usually `last8` or `last16`, depending on the run.

For E6:

- **Yes**, it replays the full model depth.
- That is deliberate because E6 asks for localization by layer range.

Source:
- Main slice: `src/inference.py::resolve_layers (40-53)` and `src/inference.py::InferenceRunner.__init__ (59-91)`
- Full-depth replay: `src/component_outputs.py::capture_component_outputs (98-136)`

Important caveat:
- `pipeline/1-infer.py` has a stale help/doc mismatch in some places: text may still say `last4`, but the actual parser default is `last16`. In viva, trust the code path, not old help text.

---

## 6. Formula Correctness Check

### Verdict

For the main metric family, the formulas in the code are **mathematically correct for the current implementation**.

### Important caveats

| Item                        | Verdict                                                |         |                       |
| ---                         | ---                                                    |         |                       |
| `attention_entropy` formula | Correct                                                |         |                       |
| `logit_confidence` as NLL   | Correct                                                |         |                       |
| `cosine_drift` formula      | Correct                                                |         |                       |
| `mahalanobis` formula       | Correct                                                |         |                       |
| `pca_deviation` formula     | Correct                                                |         |                       |
| `logit_lens = KL(final      |                                                        | layer)` | Correct for this repo |
| `cie_top3` as surrogate     | Correct description if explicitly called a surrogate   |         |                       |
| Strict composite formula    | Correct in `pipeline/2-fit.py` + `pipeline/3-score.py` |         |                       |

### Implementation mismatches you must not ignore

These are not formula errors, but they **do matter** for viva/report accuracy:

1. `src/metrics.py::composite` is legacy and not the evaluated strict composite.
2. `cie_top3` is not real causal effect.
3. `pipeline/4-eval.py` computes **sample-level F1**, not token span F1.
4. `src/evaluate.py::ece (187-213)` min-max normalizes scores on the evaluation set before binning, so this ECE is a **relative anomaly-score calibration proxy**, not true probability calibration.
5. `scripts/e3_patching.py` is currently one-directional, while the assignment asks for both directions.
6. `scripts/e4_temporal.py` currently includes baselines too, but Track B E4 is really about the five representation metrics.
7. `src/component_outputs.py::tokenize_sample (55-84)` still uses the older `answer_start >= len(offsets) - 1` truncation guard, while main inference uses the fixed `answer_start >= len(offsets)`. That can cause replay-only skipping differences on one-token answers.
8. The strict composite is best described as **validation-tuned and then frozen**, not purely unsupervised, because validation labels/AUROC are used to choose pooling, layers, signs, weights, and Mahalanobis regularization.

Items 7 and 8 are especially important: one is a real replay inconsistency, the other is a phrasing trap.

---

## 7. Can You Use This Directly in the Report?

### Short answer

You can use this as a **viva guide and report drafting base**, but not as a blind copy-paste source.

### Safe to use directly

| Content                                    | Safe? | Why                                        |
| ------------------------------------------ | ----- | ------------------------------------------ |
| Main metric definitions                    | Yes   | They now match the current code path       |
| Composite explanation                      | Yes   | It now reflects the real strict pipeline   |
| Hidden-state vs FFN/attention distinction  | Yes   | This is exactly how the repo is structured |
| KL / entropy / Mahalanobis / PCA rationale | Yes   | These are proper technical definitions     |

### Not safe to state casually without caveat

| Content                              | Why not                                                            |
| ------------------------------------ | ------------------------------------------------------------------ |
| "`cie_top3` is causal effect"        | False; it is a surrogate                                           |
| "Track B F1 in our table is span F1" | False; current script prints sample-level F1                       |
| "E6 proves FFN localization"         | Not unless the full run, not a smoke test, supports it             |
| "The detector uses all layers"       | False for the main pipeline unless you actually ran `--layers all` |

---

## 8. Viva Quick Answers

### What does each metric score indicate?

- Higher raw token score = more hallucination-like token
- Higher sample score = more hallucination-like response
- Higher AUROC = better ranking of hallucinated samples above faithful ones
- AUROC below `0.5` means the metric direction is effectively inverted

### Which file computes the metric, and which file actually uses it?

That distinction is critical:

| Metric              | Raw compute file                            | Meaningful use file                                              |
| ------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| `attention_entropy` | `src/inference.py` + `src/metrics.py`       | `pipeline/4-eval.py`                                             |
| `logit_confidence`  | `src/inference.py` + `src/metrics.py`       | `pipeline/4-eval.py`                                             |
| `cosine_drift`      | `src/metrics.py`                            | `pipeline/4-eval.py` and `pipeline/3-score.py`                   |
| `mahalanobis`       | `src/metrics.py`                            | `pipeline/4-eval.py`, `pipeline/2-fit.py`, `pipeline/3-score.py` |
| `pca_deviation`     | `src/metrics.py`                            | `pipeline/4-eval.py`, `pipeline/2-fit.py`, `pipeline/3-score.py` |
| `logit_lens`        | `src/inference.py` + `src/metrics.py`       | `pipeline/4-eval.py`, `pipeline/2-fit.py`, `pipeline/3-score.py` |
| `cie_top3`          | `pipeline/2-fit.py` + `src/metrics.py`      | `pipeline/4-eval.py`                                             |
| `composite`         | `pipeline/2-fit.py` + `pipeline/3-score.py` | `pipeline/4-eval.py`                                             |

