# Uncertainty Evaluation for SOTIF Analysis of LiDAR Object Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milinpatel07/SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection/blob/main/notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Evaluation methodology for determining whether prediction uncertainty from ensemble-based LiDAR object detection supports **ISO 21448 (SOTIF)** analysis. This repository provides the complete pipeline from the paper:

> **Uncertainty Evaluation to Support Safety of the Intended Functionality Analysis for Identifying Performance Insufficiencies in ML-Based LiDAR Object Detection**
>
> Milin Patel and Rolf Jung, Kempten University of Applied Sciences

## Overview

ISO 21448 requires identification of performance insufficiencies and triggering conditions in automated driving perception, but its analysis techniques assume explicitly specified system behaviour and cannot address neural networks whose decision logic is learned from data.

This methodology uses prediction uncertainty from deep ensembles to support three SOTIF activities:

| SOTIF Activity | ISO 21448 Clause | What It Produces |
|---|---|---|
| Performance insufficiency identification | Clause 7 | Per-detection AUROC separating TP from FP |
| Triggering condition ranking | Clause 7 | Conditions ranked by FP share and uncertainty |
| Acceptance criteria documentation | Clause 11 | Operating points with coverage and FAR |

### Key Results (Case Study)

| Indicator | AUROC |
|---|---|
| Mean confidence | 0.999 |
| Confidence variance | 0.984 |
| Geometric disagreement | 0.891 |

At confidence threshold 0.70: **26.2% coverage with zero false positives**.

## Quick Start

### Option 1: Google Colab (No Setup Required)

Click the badge above or open [`notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb`](notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb) in Colab. The notebook runs end-to-end with synthetic data matching the paper's statistics -- no GPU or dataset download needed.

### Option 2: Local Installation

```bash
git clone https://github.com/milinpatel07/SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection.git
cd SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection

pip install -e .

# Run the evaluation with synthetic demo data
python scripts/evaluate.py

# Or open the notebook locally
jupyter notebook notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb
```

### Option 3: Real Data Pipeline (Requires GPU + KITTI)

```bash
# 1. Install OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# 2. Download KITTI 3D Object Detection dataset
# 3. Train ensemble
bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5

# 4. Run inference
python scripts/run_inference.py \
    --ckpt_dirs output/ensemble/seed_*

# 5. Evaluate
python scripts/evaluate.py --input results/ensemble_results.pkl
```

## Repository Structure

```
.
├── sotif_uncertainty/          # Core Python package
│   ├── __init__.py
│   ├── uncertainty.py          # Stage 2: Uncertainty indicators (Eqs. 1-3)
│   ├── matching.py             # Stage 3: TP/FP/FN greedy matching
│   ├── metrics.py              # Stage 4: AUROC, AURC, ECE, NLL, Brier
│   ├── sotif_analysis.py       # Stage 5: TC ranking, frame flags, gates
│   ├── visualization.py        # All paper figures
│   └── demo_data.py            # Synthetic data generator
├── notebooks/
│   └── SOTIF_Uncertainty_Evaluation_Demo.ipynb  # Main demo notebook
├── scripts/
│   ├── evaluate.py             # Standalone evaluation script
│   ├── train_ensemble.sh       # Train K SECOND models
│   └── run_inference.py        # Run ensemble inference
├── Analysis/                   # Generated figures
├── data/                       # Dataset directory (not tracked)
├── requirements.txt
├── setup.py
└── LICENSE
```

## Methodology

The methodology comprises five stages:

```
LiDAR Frame + K Ensemble Members
         │
    ┌────▼────┐
    │ Stage 1  │  Ensemble Inference (K independent forward passes)
    └────┬────┘
         │  K × D^(k) detections
    ┌────▼────┐
    │ Stage 2  │  Cross-Member Association + Uncertainty Indicators
    └────┬────┘    ├─ Mean confidence (s̄ⱼ): existence uncertainty
         │         ├─ Confidence variance (σ²ₛ,ⱼ): epistemic uncertainty
         │         └─ Geometric disagreement (d_iou,j): localisation uncertainty
    ┌────▼────┐
    │ Stage 3  │  Correctness Determination (greedy matching, BEV IoU ≥ 0.5)
    └────┬────┘    └─ TP / FP / FN labels per proposal
         │
    ┌────▼────┐
    │ Stage 4  │  Metric Computation
    └────┬────┘    ├─ Discrimination: AUROC, AURC
         │         ├─ Calibration: ECE, NLL, Brier
         │         └─ Operating characteristics: Coverage, FAR at thresholds
    ┌────▼────┐
    │ Stage 5  │  SOTIF Analysis Artefacts
    └─────────┘    ├─ TC ranking (Clause 7)
                   ├─ Frame flags (Clause 7, Area 3 → Area 2)
                   ├─ Acceptance gates (Clause 11)
                   └─ Confidence interpretation (Clause 10)
```

### Three Uncertainty Indicators

| Indicator | Formula | Uncertainty Type | Safety Concern |
|---|---|---|---|
| Mean confidence s̄ⱼ | (1/K) Σ s_j^(k) | Existence | False/missed detection |
| Confidence variance σ²ₛ,ⱼ | (1/(K-1)) Σ (s_j^(k) - s̄ⱼ)² | Epistemic | Unknown operating condition |
| Geometric disagreement d_iou,j | 1 - mean pairwise BEV IoU | Localisation | Incorrect distance estimate |

### Acceptance Gate

```
G(s̄ⱼ, σ²ₛ,ⱼ, d_iou,j) = [s̄ⱼ ≥ τₛ] ∧ [σ²ₛ,ⱼ ≤ τᵥ] ∧ [d_iou,j ≤ τ_d]
```

The multi-indicator gate achieves zero FAR at a lower confidence threshold by using variance as an additional filter.

## Available Datasets for Reproduction

| Dataset | Size | Format | Weather Conditions | Access |
|---|---|---|---|---|
| **KITTI** | ~29 GB | Native KITTI | Clear only | Free ([cvlibs.net](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)) |
| **nuScenes mini** | ~4 GB | Convertible | Rain, night (19%/12%) | Free ([nuscenes.org](https://www.nuscenes.org/nuscenes)) |
| **MultiFog KITTI** | Varies | KITTI | Synthetic fog | Free ([link](https://maiminh1996.github.io/multifogkitti/)) |
| **CADC** | ~7K frames | Custom | Snow/winter | Free ([cadcd.uwaterloo.ca](http://cadcd.uwaterloo.ca/)) |
| **CARLA** | Custom | KITTI | Any (simulated) | Free ([carla.org](https://carla.org/)) |

For the paper's case study, data was generated using CARLA with 22 environmental configurations (Table 2 in paper).

## Dependencies

**Demo mode** (synthetic data, Colab-compatible):
- numpy >= 1.20
- matplotlib >= 3.4

**Full pipeline** (real data, requires GPU):
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- PyTorch >= 1.10
- spconv v2.x
- CUDA 11.x

## Citation

If you use this methodology or code, please cite:

```bibtex
@inproceedings{patel2026uncertainty,
  title={Uncertainty Evaluation to Support Safety of the Intended Functionality
         Analysis for Identifying Performance Insufficiencies in ML-Based
         LiDAR Object Detection},
  author={Patel, Milin and Jung, Rolf},
  year={2026},
  institution={Kempten University of Applied Sciences}
}
```

## References

- ISO 21448:2022 - Safety of the Intended Functionality (SOTIF)
- ISO/PAS 8800:2024 - Safety for AI-based systems in road vehicles
- Lakshminarayanan et al. (2017) - Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
- Feng et al. (2018, 2021) - Uncertainty estimation for LiDAR 3D detection
- Yan et al. (2018) - SECOND: Sparsely Embedded Convolutional Detection
- Patel and Jung (2024) - SOTIF-relevant evaluation of LiDAR detectors in CARLA
- OpenPCDet (2020) - Open-source 3D object detection codebase

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
