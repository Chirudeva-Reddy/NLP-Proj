# Pipeline Runbook (Qwen Primary)

This runbook provides component-level execution instructions for the current NLP Track B repository.

Primary profile in this runbook:
- model family: `Qwen/Qwen2.5-1.5B` for analysis scripts (`e3`-`e7`)
- core pipeline defaults: scripts support `distilgpt2` by default in `pipeline/1-infer.py`, but this runbook standardizes commands to Qwen for consistency with current experiment outputs.

## 1) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick sanity checks:

```bash
python pipeline/1-infer.py --help
python pipeline/2-fit.py --help
python pipeline/3-score.py --help
python pipeline/4-eval.py --help
python scripts/e3_patching.py --help
python scripts/e5_halueval.py --help
python scripts/e6_component_drift.py --help
python scripts/e7_failures.py --help
python scripts/e8_sota_gap.py --help
```

Note:
- `scripts/e4_temporal.py` currently has no argparse interface and runs immediately.

## 2) Dataset Placement

Required inputs:
- `dataset/ragtruth/response.jsonl`
- `dataset/ragtruth/source_info.jsonl`

For E5 HaluEval transfer script:
- `dataset/halueval/qa_data.json`

## 3) Canonical Component Order

### Step A - Person-1 style inference artifacts

```bash
python pipeline/1-infer.py \
  --model Qwen/Qwen2.5-1.5B \
  --layers last8 \
  --device auto \
  --output-dir outputs/artifacts
```

Expected outputs:
- `outputs/artifacts/train/*.pt`
- `outputs/artifacts/val/*.pt`
- `outputs/artifacts/test/*.pt`

### Step B - Fit train stats and val-tuned strict composite

```bash
python pipeline/2-fit.py \
  --artifacts-dir outputs/artifacts \
  --output outputs/stats.pt \
  --pca-components 16
```

Expected output:
- `outputs/stats.pt`

### Step C - Score test split

```bash
python pipeline/3-score.py \
  --artifacts-dir outputs/artifacts \
  --stats outputs/stats.pt \
  --output-dir outputs/scores_test \
  --split test
```

Expected output:
- `outputs/scores_test/*.pt`

### Step D - Evaluate test metrics

```bash
python pipeline/4-eval.py \
  --scores-dir outputs/scores_test \
  --aggregate max \
  --n-boot 1000
```

Expected output:
- console table (AUROC, 95% CI, Spearman, F1, ECE)

### Step E - E2 layer profile plot

```bash
python pipeline/plot.py \
  --artifacts-dir outputs/artifacts/test \
  --stats outputs/stats.pt \
  --output outputs/e2/layer_profile_verified.png
```

Expected output:
- `outputs/e2/layer_profile_verified.png`

## 4) Experiment Script Matrix (E3-E8)

### E3 - Causal intervention

```bash
python scripts/e3_patching.py \
  --model Qwen/Qwen2.5-1.5B \
  --device auto \
  --output-dir outputs/e3
```

Primary outputs:
- `outputs/e3/cie_bidirectional.md`
- `outputs/e3/cie_bidirectional.csv`
- `outputs/e3/cie_bidirectional.json`
- `outputs/e3/cie_bidirectional.png`
- `outputs/e3/run.log`

### E4 - Temporal precedence

```bash
python scripts/e4_temporal.py
```

Primary outputs:
- `outputs/e4/temporal.csv`
- `outputs/e4/temporal.json`
- `outputs/e4/temporal.png`

Important:
- script reads from hardcoded `outputs/scores` path. If your latest scores are in `outputs/scores_test`, align paths before running.

### E5 - Zero-shot transfer

```bash
python scripts/e5_halueval.py \
  --model Qwen/Qwen2.5-1.5B \
  --layers last8 \
  --device auto \
  --stats outputs/stats.pt \
  --qa-json dataset/halueval/qa_data.json \
  --artifacts-dir outputs/halueval_artifacts \
  --scores-dir outputs/halueval_scores \
  --ragtruth-scores-dir outputs/scores
```

Primary output used in report:
- `outputs/e5/E5_TRANSFER_EXPLANATION.md`

### E6 - FFN vs attention decomposition

```bash
python scripts/e6_component_drift.py \
  --model Qwen/Qwen2.5-1.5B \
  --device auto \
  --artifacts-dir outputs/artifacts/test \
  --output-dir outputs/e6
```

Primary outputs:
- `outputs/e6/component_drift.csv`
- `outputs/e6/component_drift.json`
- `outputs/e6/component_drift.png`
- `outputs/e6/run.log`

### E7 - Failure cases

```bash
python scripts/e7_failures.py \
  --model Qwen/Qwen2.5-1.5B \
  --device auto \
  --scores-dir outputs/scores_test \
  --stats outputs/stats.pt \
  --e6-json outputs/e6/component_drift.json \
  --output-dir outputs/e7
```

Primary outputs:
- `outputs/e7/failures.md`
- `outputs/e7/failures.json`
- `outputs/e7/failure_traces/*.png`

### E8 - SOTA gap

```bash
python scripts/e8_sota_gap.py \
  --scores-dir outputs/scores_test \
  --aggregate max \
  --output-dir outputs/e8
```

Primary outputs:
- `outputs/e8/sota_gap.csv`
- `outputs/e8/sota_gap.json`
- `outputs/e8/sota_gap.md`

## 5) Reproduction Profiles

### Quick run (smoke)

```bash
python pipeline/1-infer.py --model Qwen/Qwen2.5-1.5B --layers last8 --output-dir outputs/artifacts --limit 20
python pipeline/2-fit.py --artifacts-dir outputs/artifacts --output outputs/stats.pt --pca-components 16
python pipeline/3-score.py --artifacts-dir outputs/artifacts --stats outputs/stats.pt --output-dir outputs/scores_test --split test
python pipeline/4-eval.py --scores-dir outputs/scores_test --aggregate max --n-boot 200
python pipeline/plot.py --artifacts-dir outputs/artifacts/test --stats outputs/stats.pt --output outputs/e2/layer_profile_verified.png
```

### Full run (core + E3-E8)

```bash
python pipeline/1-infer.py --model Qwen/Qwen2.5-1.5B --layers last8 --device auto --output-dir outputs/artifacts
python pipeline/2-fit.py --artifacts-dir outputs/artifacts --output outputs/stats.pt --pca-components 16
python pipeline/3-score.py --artifacts-dir outputs/artifacts --stats outputs/stats.pt --output-dir outputs/scores_test --split test
python pipeline/4-eval.py --scores-dir outputs/scores_test --aggregate max --n-boot 1000
python pipeline/plot.py --artifacts-dir outputs/artifacts/test --stats outputs/stats.pt --output outputs/e2/layer_profile_verified.png
python scripts/e3_patching.py --model Qwen/Qwen2.5-1.5B --device auto --output-dir outputs/e3
python scripts/e4_temporal.py
python scripts/e5_halueval.py --model Qwen/Qwen2.5-1.5B --layers last8 --device auto --stats outputs/stats.pt --qa-json dataset/halueval/qa_data.json --artifacts-dir outputs/halueval_artifacts --scores-dir outputs/halueval_scores --ragtruth-scores-dir outputs/scores
python scripts/e6_component_drift.py --model Qwen/Qwen2.5-1.5B --device auto --artifacts-dir outputs/artifacts/test --output-dir outputs/e6
python scripts/e7_failures.py --model Qwen/Qwen2.5-1.5B --device auto --scores-dir outputs/scores_test --stats outputs/stats.pt --e6-json outputs/e6/component_drift.json --output-dir outputs/e7
python scripts/e8_sota_gap.py --scores-dir outputs/scores_test --aggregate max --output-dir outputs/e8
```

## 6) Troubleshooting

### Problem: unreadable `.pt` artifacts on macOS external drives

Symptom:
- filenames prefixed by `._` appear and break loading in some scripts.

Action:
- keep sidecar-safe filters where available (E5 already ignores `._*`).
- remove AppleDouble sidecars before runs when needed.

### Problem: missing score files

Symptom:
- `No score files in <dir>`.

Action:
- run `pipeline/3-score.py` first and confirm split path (`outputs/scores` vs `outputs/scores_test`).

### Problem: E4 uses old score directory

Symptom:
- E4 output does not reflect latest scoring run.

Action:
- align latest scores into `outputs/scores` before `scripts/e4_temporal.py`, or patch script constants.

### Problem: transfer run has low/zero valid files

Symptom:
- E5 counts look wrong.

Action:
- verify `dataset/halueval/qa_data.json` exists.
- verify both score dirs and artifact dirs are non-empty.

### Problem: path assumptions across scripts

Action:
- prefer explicit flags in scripts that support them.
- for hardcoded scripts, verify constants at top of file before execution.



### 1. Result report with test numbers, bootstrap CIs, and test logs

Include:
- `docs/TRACK_B_REPORT.md`
- `pipeline/4-eval.py` stdout table (contains bootstrap CI)
- logs:
  - `outputs/e3/run.log`
  - `outputs/e4/run_last8.log` and/or `outputs/e4/run_last18.log`
  - `outputs/e6/run.log`

### 2. PowerPoint with architecture, formulas, and result tables

Build from:
- architecture and methodology sections in `docs/TRACK_B_REPORT.md`
- formula/source files: `src/metrics.py`, `scripts/e3_patching.py`, `scripts/e4_temporal.py`
- result tables/figures in `outputs/e2` to `outputs/e8`
