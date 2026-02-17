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

See [Real Data Pipeline](#real-data-pipeline) below for the full walkthrough.

```bash
# 1. Prepare KITTI dataset
python scripts/prepare_kitti.py --data_root data/kitti

# 2. Train ensemble (K=6 SECOND detectors with different seeds)
bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5

# 3. Run inference with DBSCAN clustering
python scripts/run_inference.py \
    --ckpt_dirs output/ensemble/seed_* \
    --gt_path data/kitti/training/label_2

# 4. Full evaluation (or use pre-computed pickle from step 3)
python scripts/evaluate.py --input results/ensemble_results.pkl
```

## Repository Structure

```
.
├── sotif_uncertainty/          # Core Python package
│   ├── __init__.py             # Public API exports
│   ├── uncertainty.py          # Stage 2: Uncertainty indicators (Eqs. 1-3)
│   ├── ensemble.py             # DBSCAN clustering + uncertainty decomposition
│   ├── matching.py             # Stage 3: TP/FP/FN greedy matching (BEV IoU ≥ 0.5)
│   ├── metrics.py              # Stage 4: AUROC, AURC, ECE, NLL, Brier Score
│   ├── sotif_analysis.py       # Stage 5: TC ranking, frame flags, acceptance gates
│   ├── visualization.py        # 13 publication-quality figures
│   └── demo_data.py            # Synthetic data generator (matches paper stats)
├── notebooks/
│   └── SOTIF_Uncertainty_Evaluation_Demo.ipynb  # Interactive Colab notebook
├── scripts/
│   ├── evaluate.py             # Standalone evaluation (Stages 2-5)
│   ├── train_ensemble.sh       # Train K SECOND models via OpenPCDet
│   ├── run_inference.py        # Ensemble inference + DBSCAN + evaluation
│   └── prepare_kitti.py        # Download and prepare KITTI dataset
├── Analysis/                   # Pre-generated figures (13 plots)
├── data/                       # Dataset directory (not tracked)
├── requirements.txt
├── pyproject.toml
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
    │ Stage 2  │  DBSCAN Clustering + Uncertainty Indicators
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

### DBSCAN Detection Association

Detections from K ensemble members are clustered using DBSCAN on a BEV IoU distance matrix (following [LiDAR-MIMO](https://github.com/mpitropov/LiDAR-MIMO)):

1. For each frame, collect all detections from all K members
2. Compute pairwise `(1 - BEV IoU)` distance matrix
3. Run DBSCAN with `eps = 1 - iou_threshold` (default 0.5)
4. Three voting strategies control `min_samples`:
   - **Affirmative**: `min_samples=1` (keep all detections)
   - **Consensus**: `min_samples=K//2+1` (majority must agree)
   - **Unanimous**: `min_samples=K` (all members must agree)
5. Aggregate cluster: mean box position, yaw from highest-confidence member

## Generated Figures

The pipeline produces 13 publication-quality figures:

| Figure | Description | Paper Reference |
|---|---|---|
| Reliability diagram | Calibration: predicted confidence vs. actual accuracy | Section 5.3 |
| Risk-coverage curve | Selective prediction: risk reduction with coverage | Section 5.3 |
| Confidence-variance scatter | TP/FP separation in uncertainty space | Section 5.2 |
| ROC curves | Discrimination for all three indicators | Table 3 |
| Frame risk scatter | Per-frame mean confidence vs. variance | Section 5.5 |
| TC ranking bar chart | Triggering condition FP share ranking | Table 7 |
| Operating points comparison | Coverage and FAR at multiple thresholds | Table 6 |
| ISO 21448 scenario grid | SOTIF Area 1-4 mapping | Figure 1 |
| Indicator distributions | Histogram of each indicator by TP/FP | Extended |
| Condition boxplots | Per-condition score and variance distributions | Extended |
| Member agreement heatmap | Score correlation across ensemble members | Extended |
| Condition breakdown | Stacked bar chart of TP/FP by environment | Extended |
| Operating point heatmap | 2D threshold sweep (confidence × variance) | Extended |

All figures are saved to `Analysis/` when running `scripts/evaluate.py`.

## Real Data Pipeline

### Prerequisites

| Component | Version | Purpose |
|---|---|---|
| Python | >= 3.8 | Runtime |
| PyTorch | >= 1.10 | Training and inference |
| CUDA | 11.x | GPU acceleration |
| spconv | v2.x | Sparse convolutions for voxel backbone |
| [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | 0.6+ | SECOND detector framework |

### Step 1: Install OpenPCDet

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
pip install -r requirements.txt
python setup.py develop

# Verify installation
python -c "from pcdet.models import build_network; print('OpenPCDet OK')"
```

### Step 2: Prepare KITTI Dataset

```bash
# Automated download and preparation
python scripts/prepare_kitti.py --data_root data/kitti

# Or download manually from:
# https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
# Components needed: velodyne (29 GB), labels (5 MB), calib (16 MB)
```

Expected directory structure:
```
data/kitti/
├── training/
│   ├── velodyne/     # 7481 .bin point cloud files
│   ├── label_2/      # 7481 .txt label files
│   ├── calib/        # 7481 .txt calibration files
│   └── image_2/      # 7481 .png images (optional)
├── testing/
│   ├── velodyne/     # 7518 .bin point cloud files
│   └── calib/        # 7518 .txt calibration files
└── ImageSets/
    ├── train.txt     # 3712 training frame IDs
    ├── val.txt       # 3769 validation frame IDs
    └── test.txt      # 7518 test frame IDs
```

Then create OpenPCDet data info files:
```bash
cd OpenPCDet
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Step 3: Train Ensemble (K=6 SECOND Detectors)

```bash
# Train 6 members with different random seeds
bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5

# Custom configuration
bash scripts/train_ensemble.sh \
    --seeds 0 1 2 3 4 5 \
    --epochs 80 \
    --batch_size 4 \
    --openpcdet_root /path/to/OpenPCDet \
    --num_gpus 1
```

Each member trains the same SECOND architecture (`MeanVFE -> VoxelBackBone8x -> HeightCompression -> BaseBEVBackbone -> AnchorHeadSingle`) with identical hyperparameters, differing only in random seed. Training takes approximately 2-4 hours per member on a single GPU.

**Note on random seeds**: Standard OpenPCDet uses a hardcoded seed of 666. The training script passes `--set OPTIMIZATION.SEED <seed>` which works with LiDAR-MIMO's OpenPCDet fork. For standard OpenPCDet, modify `pcdet/utils/common_utils.py`:

```python
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Step 4: Run Ensemble Inference

```bash
# Full pipeline: inference + DBSCAN clustering + GT matching + evaluation
python scripts/run_inference.py \
    --ckpt_dirs output/ensemble/seed_0 output/ensemble/seed_1 \
                output/ensemble/seed_2 output/ensemble/seed_3 \
                output/ensemble/seed_4 output/ensemble/seed_5 \
    --data_path data/kitti \
    --split val \
    --gt_path data/kitti/training/label_2 \
    --voting consensus \
    --output_dir results/

# Or use pre-computed prediction pickles (from OpenPCDet eval)
python scripts/run_inference.py \
    --pkl_files results/seed_0.pkl results/seed_1.pkl \
               results/seed_2.pkl results/seed_3.pkl \
               results/seed_4.pkl results/seed_5.pkl \
    --gt_path data/kitti/training/label_2
```

**Voting strategies**:
- `--voting affirmative`: Keep any detection from any member (high recall, more FP)
- `--voting consensus`: Require majority agreement (balanced)
- `--voting unanimous`: Require all members to detect (high precision, lower recall)

### Step 5: Evaluate

```bash
# Standalone evaluation with figures
python scripts/evaluate.py --input results/ensemble_results.pkl

# Or open the notebook for interactive analysis
jupyter notebook notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb
```

### Using Pre-Computed Prediction Pickles

If you already have OpenPCDet prediction files (`result.pkl` from `eval_utils.eval_one_epoch()`), you can skip training and inference:

```python
# OpenPCDet result.pkl format: list[dict], one dict per frame
# Each dict: {'name', 'score', 'boxes_lidar', 'pred_labels', ...}

# Load and run through the pipeline:
python scripts/run_inference.py \
    --pkl_files member_0/result.pkl member_1/result.pkl \
                member_2/result.pkl member_3/result.pkl \
                member_4/result.pkl member_5/result.pkl \
    --gt_path data/kitti/training/label_2
```

## Available Datasets for Reproduction

| Dataset | Size | Format | Weather Conditions | Access |
|---|---|---|---|---|
| **KITTI** | ~29 GB | Native KITTI | Clear only | Free ([cvlibs.net](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)) |
| **nuScenes mini** | ~4 GB | Convertible | Rain, night (19%/12%) | Free ([nuscenes.org](https://www.nuscenes.org/nuscenes)) |
| **MultiFog KITTI** | Varies | KITTI | Synthetic fog | Free ([link](https://maiminh1996.github.io/multifogkitti/)) |
| **CADC** | ~7K frames | Custom | Snow/winter | Free ([cadcd.uwaterloo.ca](http://cadcd.uwaterloo.ca/)) |
| **CARLA** | Custom | KITTI | Any (simulated) | Free ([carla.org](https://carla.org/)) |

For the paper's case study, data was generated using CARLA with 22 environmental configurations (Table 2 in paper).

## Related Repositories

| Repository | Relevance |
|---|---|
| [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | SECOND detector training/inference framework |
| [LiDAR-MIMO](https://github.com/mpitropov/LiDAR-MIMO) | Ensemble training + DBSCAN clustering patterns |
| [uncertainty_eval](https://github.com/mpitropov/uncertainty_eval) | Scoring rules (NLL, Brier, Energy Score) for 3D detection |
| [probdet](https://github.com/asharakeh/probdet) | Probabilistic object detection evaluation |
| [pod_compare](https://github.com/asharakeh/pod_compare) | Comparative study of ensemble strategies |

## Dependencies

**Demo mode** (synthetic data, Colab-compatible):
- numpy >= 1.20
- matplotlib >= 3.4

**Full pipeline** (real data, requires GPU):
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) >= 0.6
- PyTorch >= 1.10
- spconv v2.x
- CUDA 11.x
- scikit-learn (for DBSCAN clustering)

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
- Pitropov et al. (2022) - LiDAR-MIMO: Efficient Uncertainty Estimation for LiDAR 3D Object Detection
- Patel and Jung (2024) - SOTIF-relevant evaluation of LiDAR detectors in CARLA
- OpenPCDet (2020) - Open-source 3D object detection codebase

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
