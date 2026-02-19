# Uncertainty Evaluation for SOTIF Analysis of LiDAR Object Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen)](tests/test_pipeline.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milinpatel07/SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection/blob/main/notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb)

This repository provides the evaluation methodology and experimental code for the following research:

> **Uncertainty Evaluation to Support Safety of the Intended Functionality Analysis for Identifying Performance Insufficiencies in ML-Based LiDAR Object Detection**
>
> Milin Patel and Rolf Jung, Kempten University of Applied Sciences (2026)

> **Uncertainty Representation in a SOTIF-Related Use Case with Dempster-Shafer Theory for LiDAR Sensor-Based Object Detection**
>
> Milin Patel and Rolf Jung, Kempten University of Applied Sciences (2025), [arXiv:2503.02087](https://arxiv.org/abs/2503.02087)

---

## Motivation

ISO 21448 (SOTIF) requires systematic identification of performance insufficiencies and triggering conditions in automated driving perception. However, its analysis techniques assume explicitly specified system behaviour and cannot directly address ML-based systems whose decision logic is learned from data.

This work bridges the gap by using **prediction uncertainty from deep ensembles** (K=6 SECOND detectors) as a proxy for detection reliability, enabling three ISO 21448 activities:

| SOTIF Activity | ISO 21448 Clause | Output |
|---|---|---|
| Performance insufficiency identification | Clause 7 | Per-detection AUROC separating TP from FP |
| Triggering condition ranking | Clause 7 | Conditions ranked by FP share and uncertainty |
| Acceptance criteria documentation | Clause 11 | Operating points with coverage and FAR |

---

## Experimental Results

The methodology is evaluated on two datasets: real-world KITTI and synthetic CARLA (SOTIF-PCOD, 547 frames across 22 weather configurations).

### Discrimination (AUROC)

| Indicator | KITTI (Real-World) | CARLA (Synthetic) |
|---|---|---|
| Mean confidence | 0.999 | 0.903 |
| Confidence variance | 0.889 | 0.722 |
| Geometric disagreement | 0.912 | 0.982 |

### Calibration

| Metric | KITTI | CARLA |
|---|---|---|
| ECE | 0.202 | 0.231 |
| NLL | 0.235 | 0.554 |
| Brier Score | 0.049 | 0.193 |

### Operating Points

At confidence threshold s >= 0.70 on KITTI: **25.8% coverage with zero false positives**.

Combined gate s >= 0.35 & d <= 0.30 on CARLA: **40.2% coverage with FAR = 2.8%** (22 FP out of 773 retained).

Full results, figures, and cross-dataset analysis are in [`results/`](results/).

---

## Repository Structure

```
.
├── paper/                                 # Research paper sources
│   ├── SOTIF_Uncertainty_Conference_Paper.tex
│   └── SOTIF_Uncertainty_Conference_Paper.md
│
├── sotif_uncertainty/                     # Core Python package
│   ├── __init__.py                        # Public API
│   ├── uncertainty.py                     # Uncertainty indicators (Eqs. 1-3)
│   ├── ensemble.py                        # DBSCAN clustering + uncertainty decomposition
│   ├── matching.py                        # TP/FP/FN matching (BEV IoU >= 0.5)
│   ├── metrics.py                         # AUROC, AURC, ECE, NLL, Brier Score
│   ├── sotif_analysis.py                  # TC ranking, frame flags, acceptance gates
│   ├── visualization.py                   # Publication-quality figures (13 plot types)
│   ├── carla_case_study.py                # CARLA case study data generation
│   ├── demo_data.py                       # Lightweight synthetic data for demos
│   ├── dst_uncertainty.py                 # Dempster-Shafer Theory decomposition
│   ├── mc_dropout.py                      # MC Dropout alternative to ensembles
│   ├── kitti_utils.py                     # KITTI calibration, label I/O
│   ├── weather_augmentation.py            # Physics-based weather effects
│   └── dataset_adapter.py                 # Unified adapter for KITTI/CARLA formats
│
├── scripts/                               # Execution scripts
│   ├── evaluate.py                        # Evaluation (demo / carla_study / real modes)
│   ├── run_pipeline.py                    # End-to-end pipeline orchestrator
│   ├── execute_evaluation.py              # Cross-dataset evaluation with report
│   ├── run_inference.py                   # Ensemble inference + DBSCAN + evaluation
│   ├── train_ensemble.sh                  # Train K SECOND models via OpenPCDet
│   ├── prepare_kitti.py                   # Download and prepare KITTI dataset
│   └── generate_carla_data.py             # Generate CARLA simulation data
│
├── results/                               # Evaluation outputs
│   ├── evaluation_report.md               # Full cross-dataset analysis report
│   ├── comparison_summary.json            # Machine-readable cross-dataset metrics
│   ├── results_summary.json               # CARLA case study detailed metrics
│   ├── evaluation_results.json            # CARLA case study summary
│   └── figures/
│       ├── carla_case_study/              # 13 CARLA case study figures
│       ├── carla_synthetic/               # 13 CARLA cross-dataset figures
│       ├── kitti_real_world/              # 13 KITTI figures
│       └── comparison/                    # 5 cross-dataset comparison figures
│
├── notebooks/
│   └── SOTIF_Uncertainty_Evaluation_Demo.ipynb   # Interactive Colab demo
│
├── configs/
│   └── second_sotif_ensemble.yaml         # OpenPCDet SECOND config for K=6 ensemble
│
├── tests/
│   └── test_pipeline.py                   # 58 tests across all pipeline stages
│
├── requirements.txt
├── pyproject.toml
├── setup.py
└── LICENSE
```

---

## Methodology

The evaluation pipeline comprises five stages:

```
LiDAR Frame + K Ensemble Members
         │
   ┌─────▼─────┐
   │  Stage 1   │  Ensemble Inference (K independent forward passes)
   └─────┬─────┘
         │  K × D^(k) detections
   ┌─────▼─────┐
   │  Stage 2   │  DBSCAN Clustering + Uncertainty Indicators
   └─────┬─────┘   ├─ Mean confidence (s̄_j): existence uncertainty
         │         ├─ Confidence variance (σ²_s,j): epistemic uncertainty
         │         └─ Geometric disagreement (d_iou,j): localisation uncertainty
   ┌─────▼─────┐
   │  Stage 3   │  Correctness Determination (greedy matching, BEV IoU ≥ 0.5)
   └─────┬─────┘   └─ TP / FP / FN labels per proposal
         │
   ┌─────▼─────┐
   │  Stage 4   │  Metric Computation
   └─────┬─────┘   ├─ Discrimination: AUROC, AURC
         │         ├─ Calibration: ECE, NLL, Brier
         │         └─ Operating characteristics: Coverage, FAR at thresholds
   ┌─────▼─────┐
   │  Stage 5   │  SOTIF Analysis Artefacts
   └───────────┘   ├─ TC ranking (Clause 7)
                   ├─ Frame-level triage (Clause 7, Area 3 → Area 2)
                   ├─ Acceptance gates (Clause 11)
                   └─ Confidence interpretation (Clause 10)
```

### Uncertainty Indicators

| Indicator | Formula | Uncertainty Type | Safety Concern |
|---|---|---|---|
| Mean confidence | (1/K) Σ s_j^(k) | Existence | False or missed detection |
| Confidence variance | (1/(K−1)) Σ (s_j^(k) − s̄)² | Epistemic | Unknown operating condition |
| Geometric disagreement | 1 − mean pairwise BEV IoU | Localisation | Incorrect distance estimate |

### Acceptance Gate

```
G(s̄, σ²_s, d_iou) = [s̄ ≥ τ_s] ∧ [σ²_s ≤ τ_v] ∧ [d_iou ≤ τ_d]
```

The multi-indicator gate filters detections using all three uncertainty dimensions, achieving lower FAR than confidence alone at comparable coverage.

### DBSCAN Detection Association

Detections from K ensemble members are clustered using DBSCAN on a BEV IoU distance matrix:

1. Collect all detections from K members for each frame
2. Compute pairwise `(1 − BEV IoU)` distance matrix
3. Run DBSCAN with `eps = 1 − iou_threshold` (default 0.5)
4. Voting strategies control `min_samples`:
   - **Affirmative** (`min_samples=1`): keep all detections
   - **Consensus** (`min_samples=K//2+1`): majority agreement
   - **Unanimous** (`min_samples=K`): all members agree
5. Aggregate: mean box position, yaw from highest-confidence member

### Dempster-Shafer Theory Decomposition

Ensemble member scores are converted to mass functions and combined using Dempster's rule, decomposing total uncertainty into:

| Component | Interpretation |
|---|---|
| Aleatoric | Irreducible sensor noise (Shannon entropy of pignistic probability) |
| Epistemic | Model ignorance (Plausibility − Belief interval width) |
| Ontological | Evidence for unknown unknowns (combined conflict mass) |

---

## Quick Start

### Google Colab (No Setup)

Click the Colab badge above or open [`notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb`](notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb). Runs end-to-end with synthetic data — no GPU or dataset download needed.

### Local Installation

```bash
git clone https://github.com/milinpatel07/SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection.git
cd SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection
pip install -e .

# Quick demo (synthetic data, generates figures)
python scripts/evaluate.py

# CARLA case study (547 frames, 22 weather configs, full analysis)
python scripts/evaluate.py --mode carla_study

# End-to-end pipeline with figure generation
python scripts/run_pipeline.py --mode carla_study --output_dir results

# Run tests (58 tests)
pytest tests/ -v
```

### Cross-Dataset Evaluation (CARLA + KITTI)

```bash
# Clone the SOTIF-PCOD dataset
git clone https://github.com/milinpatel07/SOTIF-PCOD.git

# Run cross-dataset evaluation with report generation
python scripts/execute_evaluation.py \
    --carla_root SOTIF-PCOD/SOTIF_Scenario_Dataset \
    --output_dir results

# Outputs:
#   results/evaluation_report.md         — full analysis with tables
#   results/comparison_summary.json      — machine-readable metrics
#   results/figures/carla_synthetic/     — 13 CARLA figures
#   results/figures/kitti_real_world/    — 13 KITTI figures
#   results/figures/comparison/          — 5 cross-dataset comparison figures
```

---

## Real Data Pipeline

For reproducing results with actual LiDAR data and trained ensemble models.

### Prerequisites

| Component | Version | Purpose |
|---|---|---|
| Python | >= 3.8 | Runtime |
| PyTorch | >= 1.10 | Training and inference |
| CUDA | 11.x | GPU acceleration |
| spconv | v2.x | Sparse convolutions |
| [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | 0.6+ | SECOND detector framework |

### Step 1: Prepare Dataset

**KITTI:**
```bash
python scripts/prepare_kitti.py --data_root data/kitti
```

**CARLA:**
```bash
python scripts/generate_carla_data.py --output_dir data/carla --frames_per_config 10
```

Both produce KITTI-format output:
```
data/{kitti,carla}/
├── training/
│   ├── velodyne/       # .bin point cloud files
│   ├── label_2/        # .txt label files
│   ├── calib/          # .txt calibration files
│   └── image_2/        # .png images (optional)
├── ImageSets/
│   ├── train.txt
│   └── val.txt
└── conditions.json     # Per-frame weather metadata (CARLA only)
```

### Step 2: Train Ensemble (K=6)

```bash
bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5

# With custom parameters
bash scripts/train_ensemble.sh \
    --seeds 0 1 2 3 4 5 \
    --epochs 80 \
    --batch_size 4 \
    --openpcdet_root /path/to/OpenPCDet
```

Each member trains the SECOND architecture (`MeanVFE → VoxelBackBone8x → HeightCompression → BaseBEVBackbone → AnchorHeadSingle`) with identical hyperparameters, differing only in random seed.

### Step 3: Ensemble Inference

```bash
python scripts/run_inference.py \
    --ckpt_dirs output/ensemble/seed_0 output/ensemble/seed_1 \
                output/ensemble/seed_2 output/ensemble/seed_3 \
                output/ensemble/seed_4 output/ensemble/seed_5 \
    --data_path data/kitti \
    --gt_path data/kitti/training/label_2 \
    --voting consensus
```

### Step 4: Evaluate

```bash
python scripts/evaluate.py \
    --input results/ensemble_results.pkl \
    --gt_path data/kitti/training/label_2 \
    --calib_path data/kitti/training/calib
```

---

## Generated Figures

The pipeline produces 13 publication-quality figures per dataset:

| Figure | Description | Paper Reference |
|---|---|---|
| Reliability diagram | Calibration: predicted confidence vs. actual accuracy | Section 5.3 |
| Risk-coverage curve | Selective prediction: risk reduction with coverage | Section 5.3 |
| Confidence-variance scatter | TP/FP separation in uncertainty space | Section 5.2 |
| ROC curves | Discrimination for all three indicators | Table 3 |
| Frame risk scatter | Per-frame mean confidence vs. variance | Section 5.5 |
| TC ranking bar chart | Triggering condition FP share ranking | Table 7 |
| Operating points comparison | Coverage and FAR at multiple thresholds | Table 6 |
| ISO 21448 scenario grid | SOTIF Area 1–4 mapping | Figure 1 |
| Indicator distributions | Histogram of each indicator by TP/FP | — |
| Condition boxplots | Per-condition score and variance distributions | — |
| Member agreement heatmap | Score correlation across ensemble members | — |
| Condition breakdown | Stacked bar of TP/FP by environment | — |
| Operating point heatmap | 2D threshold sweep (confidence × variance) | — |

---

## Testing

```bash
pytest tests/ -v          # 58 tests
python tests/test_pipeline.py   # alternative without pytest
```

Coverage includes: uncertainty indicators, DBSCAN clustering, TP/FP matching, all metrics (AUROC, ECE, NLL, Brier, AURC), SOTIF analysis, weather augmentation, DST decomposition, dataset adapter, and end-to-end pipeline.

---

## Dependencies

**Demo / Colab** (no GPU needed):
- numpy >= 1.20
- matplotlib >= 3.4
- scikit-learn >= 0.24

**Full pipeline** (real data, GPU required):
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) >= 0.6
- PyTorch >= 1.10
- spconv v2.x
- CUDA 11.x

---

## Available Datasets

| Dataset | Size | Weather Conditions | Access |
|---|---|---|---|
| [SOTIF-PCOD](https://github.com/milinpatel07/SOTIF-PCOD) | 547 frames | 22 CARLA configs | Free |
| [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) | ~29 GB | Clear | Free |
| [CARLA](https://carla.org/) | Custom | Any (simulated) | Free |

---

## Citation

```bibtex
@inproceedings{patel2026uncertainty,
  title={Uncertainty Evaluation to Support Safety of the Intended Functionality
         Analysis for Identifying Performance Insufficiencies in ML-Based
         LiDAR Object Detection},
  author={Patel, Milin and Jung, Rolf},
  year={2026},
  institution={Kempten University of Applied Sciences}
}

@article{patel2025dst,
  title={Uncertainty Representation in a SOTIF-Related Use Case with
         Dempster-Shafer Theory for LiDAR Sensor-Based Object Detection},
  author={Patel, Milin and Jung, Rolf},
  journal={arXiv preprint arXiv:2503.02087},
  year={2025}
}
```

## References

- ISO 21448:2022 — Safety of the Intended Functionality (SOTIF)
- ISO/PAS 8800:2024 — Safety for AI-based systems in road vehicles
- Lakshminarayanan et al. (2017) — Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
- Feng et al. (2018, 2021) — Uncertainty estimation for LiDAR 3D detection
- Yan et al. (2018) — SECOND: Sparsely Embedded Convolutional Detection
- Pitropov et al. (2022) — LiDAR-MIMO: Efficient Uncertainty Estimation for LiDAR 3D Object Detection
- Shafer (1976) — A Mathematical Theory of Evidence

## License

MIT License — see [LICENSE](LICENSE) for details.
