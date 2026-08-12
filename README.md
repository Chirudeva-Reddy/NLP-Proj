# Pre-Generation Hallucination Detection via Internal Representation Drift

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Project Report](https://img.shields.io/badge/PDF_Report-CS_F429-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](docs/CS_F429_Project_Report.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **CS F429 — Natural Language Processing Project**  
> **BITS Pilani, Dubai Campus** (May 2026)  
> **Authors**: Sanya Wadhawan, Chirudeva Reddy, Yusra Hakim, Joseph Cijo  
> **Supervisor**: Prof. Elakkiya Rajasekar  
> 📄 **Official Project Report PDF**: [`docs/CS_F429_Project_Report.pdf`](docs/CS_F429_Project_Report.pdf)

---

## 📌 Executive Summary

Hallucinations in Retrieval-Augmented Generation (RAG) present a critical failure mode where Large Language Models (LLMs) emit fluent but factually ungrounded text despite having access to retrieved documents. Standard hallucination detectors rely heavily on post-generation heuristics, logit sampling, or external LLM-as-a-judge evaluators—approaches that are computationally expensive, slow, and incapable of explaining *why* the model drifted from retrieved context.

This project introduces a **pre-generation hallucination detection framework** based on **Internal Representation Drift**. By extracting hidden-state dynamics across transformer layers prior to token emission, we unify five complementary internal drift signals into a frozen validation-tuned composite detector.

```
       +-----------------------------------------------------------------------------------+
       |                                   INPUT PROMPT                                    |
       |  Query (q) + Retrieved Evidence (D) vs. Query (q) + Empty Context (∅)            |
       +-----------------------------------------------------------------------------------+
                                                 |
                                                 v
       +-----------------------------------------------------------------------------------+
       |                       INTERNAL HIDDEN STATE DYNAMICS (L*)                         |
       |  Extract token-level hidden representations h_t^(l)(D) across Qwen-2.5 depth      |
       +-----------------------------------------------------------------------------------+
                                                 |
          +-------------------+------------------+-------------------+------------------+
          |                   |                     |                   |                 |
          v                   v                     v                   v                 v
   +--------------+    +--------------+      +--------------+    +--------------+  +--------------+
   | Cosine Drift |    | Mahalanobis  |      |  Logit Lens  |    |PCA Residual  |  |    Causal    |
   | δ_t^(l)      |    | Distance m_t |      |  KL Div Λ_t  |    |Dev. ρ_t^(l)  |  | Patching CIE |
   +--------------+    +--------------+      +--------------+    +--------------+  +--------------+
          |                   |                     |                   |                 |
          +-------------------+------------------+-------------------+------------------+
                                                 |
                                                 v
       +-----------------------------------------------------------------------------------+
       |                   VALIDATION-TUNED ROBUST Z-SCORE COMPOSITE FUSION                |
       |  s_t = ∑ w_i * z_i(t)  -->  Frozen non-supervised sample-level score              |
       +-----------------------------------------------------------------------------------+
                                                 |
                                                 v
       +-----------------------------------------------------------------------------------+
       |                        PRE-GENERATION HALLUCINATION SIGNAL                        |
       |  • RAGTruth Test AUROC: 0.6511 (95% CI: [0.6307, 0.6614])                         |
       |  • Pre-onset Peak: Position t-2 before hallucinated token (p = 2.86e-5)           |
       |  • Zero-shot HaluEval Transfer: 0.6450 AUROC (No re-fitting)                      |
       +-----------------------------------------------------------------------------------+
```

### 🌟 Key Research Discoveries

1. **Composite Detection Performance**: Our 5-signal composite achieves an **AUROC of 0.6511** (95% Bootstrap CI: `[0.6307, 0.6614]`) on the RAGTruth held-out test set, outperforming the Attention Entropy baseline (**0.6106**) by **+0.0405 AUROC**.
2. **Pre-Onset Temporal Precedence**: Internal representation drift (Cosine Drift) displays a statistically significant pre-hallucination peak at position **$t-2$** before the first hallucinated token is emitted ($p = 2.86 \times 10^{-5}$, Mann-Whitney U test).
3. **Causal Localization**: Bidirectional activation patching across 50+ paired examples confirms that early attention (layers 1–25%), mid FFN (26–75%), and late FFN (76–100%) layers causally drive hallucinated outputs.
4. **Zero-Shot Cross-Domain Robustness**: Applying the frozen composite parameters directly to HaluEval-QA without re-estimating training statistics yields **0.6450 AUROC**, demonstrating cross-domain stability.
5. **SOTA Gap Closure**: The framework closes **19.51%** of the gap to ReDeEP (0.82 AUROC) and **16.44%** of the gap to LUMINA (0.87 AUROC) over standard attention-entropy baselines.

---

## 📄 Project Documentation & Paper

* 📑 **Project Report (PDF)**: [`docs/CS_F429_Project_Report.pdf`](docs/CS_F429_Project_Report.pdf) — Complete 14-page formal manuscript including theoretical formulation, experimental protocol, and individual contribution breakdown.
* 📖 **Pipeline Runbook**: [`docs/PIPELINE_RUNBOOK.md`](docs/PIPELINE_RUNBOOK.md) — Step-by-step shell commands, flags, and CLI arguments for pipeline execution.
* 🛡️ **Metrics Defense & Viva Guide**: [`METRICS_VIVA.md`](METRICS_VIVA.md) — Deep-dive viva defense documentation covering mathematical derivations and implementation logic.
* 📚 **Comprehensive Technical Reference**: [`NLP-sub/COMPREHENSIVE.md`](NLP-sub/COMPREHENSIVE.md) — Detailed metric definitions and architecture breakdown.

---

## 🧮 Theoretical Formulation & 5 Internal Drift Signals

Let $\mathcal{M}$ be a transformer language model with $L$ layers. Given query $q$ and retrieved context $\mathcal{D}$, let $h_t^{(\ell)}(\mathcal{D}) \in \mathbb{R}^d$ represent the hidden state at layer $\ell$ and token position $t$.

### 1. Cosine Drift ($\delta_t^{(\ell)}$)
Measures continuous representational trajectory instability between consecutive answer tokens:
$$\delta_t^{(\ell)} = 1 - \frac{h_t^{(\ell)}(\mathcal{D}) \cdot h_{t-1}^{(\ell)}(\mathcal{D})}{\|h_t^{(\ell)}(\mathcal{D})\|_2 \|h_{t-1}^{(\ell)}(\mathcal{D})\|_2}$$

### 2. Mahalanobis Distance ($m_t^{(\ell)}$)
Quantifies off-manifold displacement from the training distribution of faithful tokens:
$$m_t^{(\ell)} = \sqrt{(h_t^{(\ell)} - \mu_\ell)^\top \Sigma_\ell^{-1} (h_t^{(\ell)} - \mu_\ell)}$$
*Note: $\mu_\ell$ and $\Sigma_\ell$ are estimated strictly on training-split faithful tokens.*

### 3. Logit Lens Divergence ($\Lambda_t^{(\ell)}$)
Measures cross-depth internal prediction disagreement by projecting intermediate representations through the unembedding matrix $W_U$:
$$\hat{P}_t^{(\ell)}(\cdot) = \text{softmax}(W_U h_t^{(\ell)}(\cdot))$$
$$\Lambda_t^{(\ell)} = D_{\text{KL}}\left(\hat{P}_t^{(\ell)}(\mathcal{D}) \,\|\, \hat{P}_t^{(\ell)}(\emptyset)\right)$$

### 4. PCA Residual Deviation ($\rho_t^{(\ell)}$)
Measures activation deviation outside the low-dimensional faithful subspace $V_\ell \in \mathbb{R}^{d \times r}$:
$$\rho_t^{(\ell)} = \left\| h_t^{(\ell)}(\mathcal{D}) - V_\ell V_\ell^\top h_t^{(\ell)}(\mathcal{D}) \right\|_2$$

### 5. Causal Indirect Effect ($\text{CIE}_c$)
Evaluates the change in target token probability under bidirectional activation patching:
$$\text{CIE}_c = P_{\mathcal{M}}^{\text{patch}(c)}(y_t^*) - P_{\mathcal{M}}^{\text{corrupt}}(y_t^*)$$

---

## 📊 Experimental Results & Key Figures

### 1. Main Detection Performance on RAGTruth Test Set

Evaluated on 2,655 test responses (38.9% hallucination rate) using `Qwen/Qwen2.5-1.5B`.

| Metric / Composite | AUROC ↑ | 95% Bootstrap CI | F1 Span ↑ | Spearman $\rho$ ↑ | ECE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Attention Entropy (Baseline B1)** | 0.6106 | [0.5891, 0.6318] | 0.5351 | 0.1869 | 0.2230 |
| **Logit Confidence (Baseline B2)** | 0.5497 | [0.5274, 0.5716] | 0.4737 | 0.0839 | 0.1249 |
| **Cosine Drift** | 0.5989 | [0.5772, 0.6201] | 0.5181 | 0.1670 | 0.0472 |
| **Mahalanobis Distance** | 0.5389 | [0.5168, 0.5612] | 0.4528 | 0.0657 | 0.0814 |
| **Logit Lens Divergence** | 0.5732 | [0.5510, 0.5954] | 0.4836 | 0.1237 | 0.1112 |
| **PCA Deviation** | 0.5490 | [0.5269, 0.5711] | 0.4437 | 0.0828 | 0.0778 |
| **CIE (Top-3 Layers)** | 0.5138 | [0.4916, 0.5360] | 0.4286 | 0.0233 | 0.0824 |
| **✨ Full Composite Detector** | **0.6511** | **[0.6307, 0.6614]** | **0.5548** | **0.2553** | **0.0678** |

---

### 2. Layer-Profile Analysis ($\mathcal{L}^*$) & Layer Localization

Point-biserial correlation across Qwen-2.5 layers reveals that discriminative hallucination signals concentrate in middle-to-late transformer layers (saved indices 5–15 $\rightarrow$ Qwen layers 15–25). The top-3 selected layers are **21, 23, and 22**.

![Layer Profile](docs/images/layer_profile.png)

---

### 3. Bidirectional Activation Patching (E3)

Causal intervention across 50 paired examples confirms that activations in early attention and FFN layers exert a direct causal influence on token generation probabilities.

| Component Group | Layers | CIE ($f \rightarrow h$) | CIE ($h \rightarrow f$) | Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **Early Attention** | 1–25% | -1.0991 | -1.0782 | **Yes** ($p < 0.05$) |
| **Mid FFN** | 26–75% | -0.0550 | -0.0299 | **Yes** ($p < 0.05$) |
| **Late FFN** | 76–100% | -0.2732 | -0.1608 | **Yes** ($p < 0.05$) |
| **Copying Heads** | Last 25% | -0.0807 | -0.0358 | **Yes** ($p < 0.05$) |

![Bidirectional CIE](docs/images/cie_bidirectional.png)

---

### 4. Pre-Onset Temporal Precedence Analysis (E4)

Tracking drift signals relative to the first hallucinated token ($t=0$) shows that internal Cosine Drift peaks at position **$t-2$** ($p = 2.86 \times 10^{-5}$), proving that hidden representations drift *before* the hallucination manifests in generated text.

| Token Offset | Cosine Drift ($\delta$) | Mahalanobis ($m$) | Logit Lens ($\Lambda$) | Spearman $\rho$ | CIE Signal |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$t-3$** | 0.2668 | 37.9664 | 3.6684 | 87.4574 | **37.7385** |
| **$t-2$ (Peak)** | **0.2872** | 34.0917 | 3.5820 | 80.0704 | 35.6801 |
| **$t-1$** | 0.2821 | 31.4590 | **4.2375** | 72.2008 | 31.7477 |
| **$t$ (Onset)** | 0.2729 | 36.5722 | 3.8713 | 85.6095 | 36.3971 |
| **$t+1$** | 0.2708 | **38.1772** | 3.5158 | **88.1041** | 38.2783 |

![Temporal Precedence](docs/images/temporal_precedence.png)

---

### 5. Zero-Shot Cross-Domain Transfer to HaluEval (E5)

Evaluating the frozen composite on HaluEval-QA without re-estimating training statistics demonstrates robust cross-domain generalization.

| Metric / Composite | RAGTruth AUROC | HaluEval AUROC | Performance Drop |
| :--- | :---: | :---: | :---: |
| **Full Composite** | **0.6508** | **0.6450** | **+0.0058** (Robust) |
| **Cosine Drift** | 0.6244 | 0.8345 | -0.2102 |
| **Mahalanobis Distance** | 0.5190 | 0.7006 | -0.1815 |
| **Logit Lens Divergence** | 0.5917 | 0.4337 | +0.1580 (Brittle) |
| **PCA Residual Deviation** | 0.5764 | 0.7186 | -0.1421 |

---

### 6. Component Drift & Failure Trace Visualizations (E6 & E7)

Decomposing representation updates into Self-Attention vs. Feed-Forward Network (FFN) streams confirms that FFN key-value memories contribute significantly to mid-to-late layer drift.

![Component Drift](docs/images/component_drift.png)

#### Qualitative Failure Analysis
Below are representative trace plots analyzing edge cases (False Negatives, False Positives, and Metric Disagreements):

````carousel
![False Negative Trace](docs/images/failure_fn_1289.png)
<!-- slide -->
![False Positive Trace](docs/images/failure_fp_12310.png)
<!-- slide -->
![Metric Disagreement Trace](docs/images/failure_disagreement_3574.png)
````

---

## 🛠️ Reproduction & Installation Guide

This project is engineered for full end-to-end reproducibility. Follow the steps below to replicate all results from scratch.

### 1. Prerequisites & Environment Setup

* **Python**: 3.10 or higher
* **Hardware**: GPU with $\ge$ 16GB VRAM (or Apple Silicon M-series with Unified Memory)
* **Storage**: $\ge$ 15GB free disk space for dataset and hidden-state tensor artifacts

```bash
# Clone the repository
git clone https://github.com/Chirudeva-Reddy/NLP-Proj.git
cd NLP-Proj

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r Requirements.txt
```

### 2. Dataset Preparation

Place the primary RAGTruth JSONL files under `dataset/ragtruth/`:

```
dataset/
├── ragtruth/
│   ├── response.jsonl        # Model generated responses & character hallucination spans
│   └── source_info.jsonl     # Source documents & retrieval context
└── halueval/
    └── qa_data.json          # HaluEval QA evaluation set (auto-downloaded if missing)
```

---

### 3. Canonical Pipeline Execution (4-Step Sequential Run)

```bash
# Step 1: Run inference & save float16 hidden-state artifacts
python NLP-sub/pipeline/1-infer.py \
  --model Qwen/Qwen2.5-1.5B \
  --layers last18 \
  --device auto \
  --output-dir NLP-sub/outputs/artifacts

# Step 2: Fit train-only statistics (μ, Σ, V) & freeze composite parameters
python NLP-sub/pipeline/2-fit.py \
  --artifacts-dir NLP-sub/outputs/artifacts \
  --output NLP-sub/outputs/stats.pt \
  --pca-components 16

# Step 3: Score held-out test split using frozen stats.pt
python NLP-sub/pipeline/3-score.py \
  --artifacts-dir NLP-sub/outputs/artifacts \
  --stats NLP-sub/outputs/stats.pt \
  --output-dir NLP-sub/outputs/scores_test \
  --split test

# Step 4: Evaluate test performance & compute 1000-iteration bootstrap CIs
python NLP-sub/pipeline/4-eval.py \
  --scores-dir NLP-sub/outputs/scores_test \
  --aggregate max \
  --n-boot 1000
```

---

### 4. Running Extended Analysis Experiments (E2–E8)

```bash
# E2: Generate per-layer AUROC profile plot
python NLP-sub/pipeline/plot.py \
  --artifacts-dir NLP-sub/outputs/artifacts/test \
  --stats NLP-sub/outputs/stats.pt \
  --output docs/images/layer_profile.png

# E3: Causal activation patching
python NLP-sub/scripts/e3_patching.py \
  --model Qwen/Qwen2.5-1.5B \
  --device auto \
  --output-dir NLP-sub/outputs/e3

# E4: Temporal precedence pre-onset analysis
python NLP-sub/scripts/e4_temporal.py

# E5: Zero-shot HaluEval cross-domain transfer
python NLP-sub/scripts/e5_halueval.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --layers last18 \
  --device auto \
  --stats NLP-sub/outputs/stats.pt \
  --qa-json dataset/halueval/qa_data.json \
  --artifacts-dir NLP-sub/outputs/halueval_artifacts \
  --scores-dir NLP-sub/outputs/halueval_scores \
  --ragtruth-scores-dir NLP-sub/outputs/scores_test

# E6: Component update direction drift (FFN vs. Self-Attention)
python NLP-sub/scripts/e6_component_drift.py \
  --model Qwen/Qwen2.5-1.5B \
  --device auto \
  --artifacts-dir NLP-sub/outputs/artifacts/test \
  --output-dir NLP-sub/outputs/e6

# E7: Deterministic failure case analysis
python NLP-sub/scripts/e7_failures.py \
  --model Qwen/Qwen2.5-1.5B \
  --device auto \
  --scores-dir NLP-sub/outputs/scores_test \
  --stats NLP-sub/outputs/stats.pt \
  --e6-json NLP-sub/outputs/e6/component_drift.json \
  --output-dir NLP-sub/outputs/e7

# E8: SOTA gap analysis (ReDeEP & LUMINA comparison)
python NLP-sub/scripts/e8_sota_gap.py \
  --scores-dir NLP-sub/outputs/scores_test \
  --aggregate max \
  --output-dir NLP-sub/outputs/e8
```

---

### 💻 Interactive Live Demo CLI

To score any arbitrary prompt-response pair on-the-fly:

```bash
python NLP-sub/scripts/live_demo.py \
  --profile local \
  --input-file NLP-sub/live_demo_input.example.json
```

Or pass direct text files:
```bash
python NLP-sub/scripts/live_demo.py \
  --profile local \
  --context-file /path/to/retrieved_context.txt \
  --passage-file /path/to/generated_answer.txt
```

---

## 🗂️ Repository Directory Architecture

```
NLP-Proj/
├── README.md                           # Master Project Documentation
├── METRICS_VIVA.md                     # Comprehensive Viva Defense Guide
├── Requirements.txt                    # Project Python Dependencies
├── docs/                               # Documentation & Assets
│   ├── CS_F429_Project_Report.pdf      # Formal Project Report PDF
│   ├── PIPELINE_RUNBOOK.md             # Detailed Execution Runbook
│   └── images/                         # Generated Result Plots & Traces
│       ├── layer_profile.png
│       ├── cie_bidirectional.png
│       ├── temporal_precedence.png
│       ├── component_drift.png
│       ├── failure_fn_1289.png
│       ├── failure_fp_12310.png
│       └── failure_disagreement_3574.png
├── dataset/                            # Dataset Directories (Git-ignored)
│   ├── ragtruth/                       # RAGTruth dataset
│   └── halueval/                       # HaluEval QA dataset
└── NLP-sub/                            # Primary Codebase & Pipeline
    ├── COMPREHENSIVE.md                # In-depth Codebase Guide
    ├── pipeline/                       # 4-Step Pipeline Core Scripts
    │   ├── 1-infer.py                  # Step 1: Inference & Tensor Artifacts
    │   ├── 2-fit.py                    # Step 2: Fitting Train Stats & Tuning Composite
    │   ├── 3-score.py                  # Step 3: Scoring Test Split
    │   ├── 4-eval.py                   # Step 4: Metric Evaluation & Bootstrap CI
    │   └── plot.py                     # E2 Layer Profile Generator
    ├── src/                            # Modular Python Utilities
    │   ├── dataset.py                  # Grouped Train/Val/Test Splitter
    │   ├── inference.py                # PyTorch Forward Hooks & Tensor Extractor
    │   ├── metrics.py                  # Mathematical Implementations of 5 Signals
    │   ├── evaluate.py                 # AUROC, F1, Spearman, ECE Metrics
    │   ├── component_outputs.py        # FFN/Attn Replay Hooks for E6/E7
    │   └── halueval.py                 # HaluEval Dataset Loader
    ├── scripts/                        # Extended Analysis Scripts (E3-E8)
    │   ├── e3_patching.py              # Activation Patching Engine
    │   ├── e4_temporal.py              # Temporal Precedence Analysis
    │   ├── e5_halueval.py              # HaluEval Transfer Evaluator
    │   ├── e6_component_drift.py       # FFN vs. Attention Drift
    │   ├── e7_failures.py              # Failure Trace Generator
    │   ├── e8_sota_gap.py              # SOTA Gap Metric Computer
    │   └── live_demo.py                # Single-Sample Live Scoring CLI
    └── outputs/                        # Saved Artifacts & Results (Git-ignored)
```

---

## 👥 Authors & Academic Attribution

This project was conducted as part of **CS F429: Natural Language Processing Project** at **BITS Pilani, Dubai Campus** under the supervision of **Prof. Elakkiya Rajasekar**.

| Student Name | Student ID | Primary Responsibilities |
| :--- | :---: | :--- |
| **Sanya Wadhawan** | `2023A7PS0296U` | Methodology framing, dataset preprocessing, RAGTruth pipeline integration |
| **Chirudeva Reddy** | `2023A7PS0331U` | Experimental setup, composite metric fusion, causal patching analysis |
| **Yusra Hakim** | `2022A7PS0004U` | Related work survey, mechanistic interpretability literature review |
| **Joseph Cijo** | `2022A7PS0019U` | Introduction, result tables, baseline evaluation, final report documentation |

---

## 📜 Citation

If you find this repository or project report useful in your research, please cite our work:

```bibtex
@techreport{wadhawan2026pregen,
  title     = {Pre-Generation Hallucination Detection via Internal Representation Drift},
  author    = {Wadhawan, Sanya and Reddy, Chirudeva and Hakim, Yusra and Cijo, Joseph},
  institution = {BITS Pilani, Dubai Campus},
  course    = {CS F429 - Natural Language Processing},
  year      = {2026},
  month     = {May},
  url       = {https://github.com/Chirudeva-Reddy/NLP-Proj}
}
```

---

<div align="center">
  <sub>Built with ❤️ for Natural Language Processing Research at BITS Pilani, Dubai Campus.</sub>
</div>
