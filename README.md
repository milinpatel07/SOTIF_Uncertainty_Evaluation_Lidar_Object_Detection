# Uncertainty Evaluation for SOTIF Analysis of LiDAR Object Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milinpatel07/SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection/blob/main/notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-58%2F58%20passed-brightgreen)](tests/test_pipeline.py)

Evaluation methodology for determining whether prediction uncertainty from ensemble-based LiDAR object detection supports **ISO 21448 (SOTIF)** analysis. This repository provides the complete pipeline from the papers:

> **Uncertainty Evaluation to Support Safety of the Intended Functionality Analysis for Identifying Performance Insufficiencies in ML-Based LiDAR Object Detection**
>
> Milin Patel and Rolf Jung, Kempten University of Applied Sciences (2026)

> **Uncertainty Representation in a SOTIF-Related Use Case with Dempster-Shafer Theory for LiDAR Sensor-Based Object Detection**
>
> Milin Patel and Rolf Jung, Kempten University of Applied Sciences (2025), [arXiv:2503.02087](https://arxiv.org/abs/2503.02087)

## Overview

ISO 21448 requires identification of performance insufficiencies and triggering conditions in automated driving perception, but its analysis techniques assume explicitly specified system behaviour and cannot address neural networks whose decision logic is learned from data.

This methodology uses prediction uncertainty from deep ensembles to support three SOTIF activities:

| SOTIF Activity | ISO 21448 Clause | What It Produces |
|---|---|---|
| Performance insufficiency identification | Clause 7 | Per-detection AUROC separating TP from FP |
| Triggering condition ranking | Clause 7 | Conditions ranked by FP share and uncertainty |
| Acceptance criteria documentation | Clause 11 | Operating points with coverage and FAR |

### Key Results

**KITTI Real-World (Paper Statistics):**

| Indicator | AUROC |
|---|---|
| Mean confidence | 0.999 |
| Confidence variance | 0.889 |
| Geometric disagreement | 0.912 |

At confidence threshold 0.70: **25.8% coverage with zero false positives**.

**CARLA Synthetic (547 frames, 22 weather configs):**

| Indicator | AUROC |
|---|---|
| Mean confidence | 0.895 |
| Confidence variance | 0.738 |
| Geometric disagreement | 0.974 |

Multi-indicator gate (s>=0.35 & d<=0.49): **38.3% coverage with zero false positives**.

See the full cross-dataset comparison in [`reports/evaluation_report/`](reports/evaluation_report/).

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

# Run end-to-end pipeline (demo mode with figures)
python scripts/run_pipeline.py --mode demo --output_dir results/demo

# Run tests to verify everything works (58 tests)
python tests/test_pipeline.py

# Or open the notebook locally
jupyter notebook notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb
```

### Option 3: Real Data Pipeline (Requires GPU + KITTI or CARLA)

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

# 4. Full evaluation with GT matching and figure generation
python scripts/evaluate.py \
    --input results/ensemble_results.pkl \
    --gt_path data/kitti/training/label_2 \
    --calib_path data/kitti/training/calib
```

### Option 4: Cross-Dataset Evaluation (CARLA + KITTI)

```bash
# Clone the SOTIF-PCOD dataset (547 CARLA frames, 22 weather configs)
git clone https://github.com/milinpatel07/SOTIF-PCOD.git

# Run cross-dataset evaluation with report generation
python scripts/execute_evaluation.py \
    --carla_root SOTIF-PCOD/SOTIF_Scenario_Dataset \
    --output_dir reports/evaluation_report

# This produces:
#   - 13 CARLA figures + 13 KITTI figures + 5 comparison figures
#   - SOTIF_Evaluation_Report.md with full analysis tables
#   - comparison_summary.json with machine-readable metrics
```

### Option 5: CARLA Simulation Data (Custom Generation)

```bash
# Generate synthetic LiDAR data with 22 weather configs (no CARLA needed)
python scripts/generate_carla_data.py --output_dir data/carla --mode synthetic

# Or connect to a running CARLA instance
python scripts/generate_carla_data.py --output_dir data/carla --mode carla

# Evaluate with condition metadata
python scripts/evaluate.py \
    --input results/ensemble_results.pkl \
    --conditions_file data/carla/conditions.json
```

## Repository Structure

```
.
├── sotif_uncertainty/              # Core Python package (v2.0.0)
│   ├── __init__.py                 # Public API exports
│   ├── uncertainty.py              # Stage 2: Uncertainty indicators (Eqs. 1-3)
│   ├── ensemble.py                 # DBSCAN clustering + uncertainty decomposition
│   ├── matching.py                 # Stage 3: TP/FP/FN greedy matching (BEV IoU >= 0.5)
│   ├── metrics.py                  # Stage 4: AUROC, AURC, ECE, NLL, Brier Score
│   ├── sotif_analysis.py           # Stage 5: TC ranking, frame flags, acceptance gates
│   ├── visualization.py            # 13 publication-quality figures
│   ├── demo_data.py                # Synthetic data generator (matches paper stats)
│   ├── kitti_utils.py              # KITTI calibration, label loading, point cloud I/O
│   ├── mc_dropout.py               # MC Dropout uncertainty (alternative to ensembles)
│   ├── weather_augmentation.py     # Physics-based LiDAR weather effects (rain/fog/snow/spray)
│   ├── dst_uncertainty.py          # Dempster-Shafer Theory uncertainty decomposition
│   └── dataset_adapter.py          # Unified adapter for KITTI/CARLA/custom datasets
├── notebooks/
│   └── SOTIF_Uncertainty_Evaluation_Demo.ipynb  # Interactive Colab notebook
├── scripts/
│   ├── evaluate.py                 # Standalone evaluation (Stages 2-5)
│   ├── run_pipeline.py             # End-to-end pipeline orchestrator (demo/kitti/carla modes)
│   ├── execute_evaluation.py       # Cross-dataset SOTIF evaluation with report generation
│   ├── train_ensemble.sh           # Train K SECOND models via OpenPCDet
│   ├── run_inference.py            # Ensemble inference + DBSCAN + evaluation
│   ├── prepare_kitti.py            # Download and prepare KITTI dataset
│   └── generate_carla_data.py      # Generate CARLA simulation data
├── configs/
│   └── second_sotif_ensemble.yaml  # OpenPCDet SECOND detector config for K=6 ensemble
├── tests/
│   └── test_pipeline.py            # 58 tests covering all pipeline stages
├── reports/
│   └── evaluation_report/          # Cross-dataset evaluation results
│       ├── SOTIF_Evaluation_Report.md  # Implementation report with tables and analysis
│       ├── comparison_summary.json     # Machine-readable results summary
│       ├── carla_synthetic/            # 13 CARLA figures
│       ├── kitti_real_world/           # 13 KITTI figures
│       └── comparison/                 # 5 cross-dataset comparison figures
├── Analysis/                       # Pre-generated figures (13 plots)
├── data/                           # Dataset directory (not tracked)
├── requirements.txt
├── pyproject.toml
├── setup.py
└── LICENSE
```

## Methodology

The methodology comprises five stages:

```
LiDAR Frame + K Ensemble Members
         |
    +----v----+
    | Stage 1  |  Ensemble Inference (K independent forward passes)
    +----+----+
         |  K x D^(k) detections
    +----v----+
    | Stage 2  |  DBSCAN Clustering + Uncertainty Indicators
    +----+----+    |-- Mean confidence (s_bar_j): existence uncertainty
         |         |-- Confidence variance (sigma^2_s,j): epistemic uncertainty
         |         +-- Geometric disagreement (d_iou,j): localisation uncertainty
    +----v----+
    | Stage 3  |  Correctness Determination (greedy matching, BEV IoU >= 0.5)
    +----+----+    +-- TP / FP / FN labels per proposal
         |
    +----v----+
    | Stage 4  |  Metric Computation
    +----+----+    |-- Discrimination: AUROC, AURC
         |         |-- Calibration: ECE, NLL, Brier
         |         +-- Operating characteristics: Coverage, FAR at thresholds
    +----v----+
    | Stage 5  |  SOTIF Analysis Artefacts
    +---------+    |-- TC ranking (Clause 7)
                   |-- Frame flags (Clause 7, Area 3 -> Area 2)
                   |-- Acceptance gates (Clause 11)
                   +-- Confidence interpretation (Clause 10)
```

### Three Uncertainty Indicators

| Indicator | Formula | Uncertainty Type | Safety Concern |
|---|---|---|---|
| Mean confidence | (1/K) * sum(s_j^(k)) | Existence | False/missed detection |
| Confidence variance | (1/(K-1)) * sum((s_j^(k) - s_bar)^2) | Epistemic | Unknown operating condition |
| Geometric disagreement | 1 - mean pairwise BEV IoU | Localisation | Incorrect distance estimate |

### Acceptance Gate

```
G(s_bar, sigma^2_s, d_iou) = [s_bar >= tau_s] AND [sigma^2_s <= tau_v] AND [d_iou <= tau_d]
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

### MC Dropout Alternative

As an alternative to deep ensembles, the `mc_dropout` module provides Monte Carlo Dropout uncertainty estimation using a single model with K stochastic forward passes. This requires only one trained model but typically yields lower uncertainty quality (see notebook Section 13 for comparison).

### Dempster-Shafer Theory Uncertainty Decomposition

The `dst_uncertainty` module implements evidence-based uncertainty decomposition following the second paper (Patel & Jung, 2025). Ensemble member scores are converted to Dempster-Shafer mass functions and combined using Dempster's rule to decompose total uncertainty into three components:

| Component | Definition | Interpretation |
|---|---|---|
| Aleatoric | Shannon entropy of pignistic probability | Irreducible sensor noise |
| Epistemic | Plausibility - Belief interval width | Model ignorance / insufficient training data |
| Ontological | Combined conflict mass | Evidence for unknown unknowns |

Key finding: Epistemic uncertainty is the primary discriminator between TP and FP, as ensemble members disagree more on incorrect detections.

### Physics-Based Weather Augmentation

The `weather_augmentation` module applies physically-motivated weather effects to LiDAR point clouds for SOTIF triggering condition analysis:

| Effect | Physical Model | Parameter |
|---|---|---|
| Rain | Beer-Lambert attenuation + random droplet hits | `intensity` (0-1) |
| Fog | Koschmieder's law (visibility-range extinction) | `density` (0-1) |
| Snow | Geometric occlusion + accumulated layer | `intensity` (0-1) |
| Spray | Road-surface water splash from nearby vehicles | `amount` (0-1) |

Seven built-in presets (clear, light_rain, heavy_rain, light_fog, dense_fog, snow, extreme) and severity scoring enable systematic triggering condition evaluation.

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
| Operating point heatmap | 2D threshold sweep (confidence x variance) | Extended |

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

### Step 2: Prepare Dataset

**Option A: KITTI**
```bash
# Automated download and preparation
python scripts/prepare_kitti.py --data_root data/kitti

# Or download manually from:
# https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
# Components needed: velodyne (29 GB), labels (5 MB), calib (16 MB)
```

**Option B: CARLA**
```bash
# Generate synthetic data with all 22 weather configs
python scripts/generate_carla_data.py --output_dir data/carla --frames_per_config 10

# This creates KITTI-format data with condition metadata
# for SOTIF triggering condition analysis
```

Expected directory structure (both KITTI and CARLA outputs):
```
data/{kitti,carla}/
├── training/
│   ├── velodyne/     # .bin point cloud files
│   ├── label_2/      # .txt label files
│   ├── calib/        # .txt calibration files
│   └── image_2/      # .png images (optional)
├── ImageSets/
│   ├── train.txt     # Training frame IDs
│   └── val.txt       # Validation frame IDs
└── conditions.json   # Per-frame weather metadata (CARLA only)
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

Each member trains the same SECOND architecture (`MeanVFE -> VoxelBackBone8x -> HeightCompression -> BaseBEVBackbone -> AnchorHeadSingle`) with identical hyperparameters, differing only in random seed.

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
# Standalone evaluation with figures (labels in pickle from step 4)
python scripts/evaluate.py --input results/ensemble_results.pkl

# With separate GT path and calibration (for proper KITTI label loading)
python scripts/evaluate.py \
    --input results/ensemble_results.pkl \
    --gt_path data/kitti/training/label_2 \
    --calib_path data/kitti/training/calib

# With CARLA condition metadata (for TC analysis)
python scripts/evaluate.py \
    --input results/ensemble_results.pkl \
    --conditions_file data/carla/conditions.json

# Or open the notebook for interactive analysis
jupyter notebook notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb
```

### Using Pre-Computed Prediction Pickles

If you already have OpenPCDet prediction files (`result.pkl` from `eval_utils.eval_one_epoch()`), you can skip training and inference:

```bash
# OpenPCDet result.pkl format: list[dict], one dict per frame
# Each dict: {'name', 'score', 'boxes_lidar', 'pred_labels', ...}

# Load and run through the pipeline:
python scripts/run_inference.py \
    --pkl_files member_0/result.pkl member_1/result.pkl \
                member_2/result.pkl member_3/result.pkl \
                member_4/result.pkl member_5/result.pkl \
    --gt_path data/kitti/training/label_2
```

## KITTI Utilities

The `kitti_utils` module provides proper coordinate transformation between camera and LiDAR frames:

```python
from sotif_uncertainty.kitti_utils import (
    KITTICalibration,
    load_kitti_labels_as_lidar,
    load_point_cloud,
    voxelize_point_cloud,
)

# Load calibration and convert labels from camera to LiDAR frame
gt_boxes_lidar = load_kitti_labels_as_lidar(
    'data/kitti/training/label_2/000000.txt',
    'data/kitti/training/calib/000000.txt',
)

# Load and filter point cloud
points = load_point_cloud(
    'data/kitti/training/velodyne/000000.bin',
    xlim=(0, 70.4), ylim=(-40, 40), zlim=(-3, 1),
)

# Voxelize for SECOND detector input
voxels, coords, num_pts = voxelize_point_cloud(points)
```

## Testing

Run the full test suite (58 tests covering all pipeline stages):

```bash
python tests/test_pipeline.py

# Or with pytest
pip install pytest
pytest tests/ -v
```

Test coverage includes:
- Core uncertainty indicators (Eqs. 1-3)
- DBSCAN clustering and ensemble association
- TP/FP matching at BEV IoU >= 0.5
- AUROC, ECE, NLL, Brier Score, AURC metrics
- SOTIF analysis (TC ranking, acceptance gates, frame triage)
- Weather augmentation (rain, fog, snow, spray, presets, severity)
- Dempster-Shafer Theory (mass functions, combination, decomposition, gates)
- Dataset adapter (KITTI/CARLA format detection, conditions loading)
- End-to-end pipeline with DST and weather augmentation

## Available Datasets for Reproduction

| Dataset | Size | Format | Weather Conditions | Access |
|---|---|---|---|---|
| **SOTIF-PCOD** | 547 frames | KITTI | 22 CARLA weather configs | Free ([GitHub](https://github.com/milinpatel07/SOTIF-PCOD)) |
| **KITTI** | ~29 GB | Native KITTI | Clear only | Free ([cvlibs.net](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)) |
| **nuScenes mini** | ~4 GB | Convertible | Rain, night (19%/12%) | Free ([nuscenes.org](https://www.nuscenes.org/nuscenes)) |
| **MultiFog KITTI** | Varies | KITTI | Synthetic fog | Free ([link](https://maiminh1996.github.io/multifogkitti/)) |
| **CADC** | ~7K frames | Custom | Snow/winter | Free ([cadcd.uwaterloo.ca](http://cadcd.uwaterloo.ca/)) |
| **CARLA** | Custom | KITTI | Any (simulated) | Free ([carla.org](https://carla.org/)) |

The **SOTIF-PCOD** dataset provides 547 CARLA-generated LiDAR frames in KITTI format across 22 weather configurations (7 noon + 7 sunset + 7 night + DustStorm) on a multi-lane highway scenario (Town04). It includes pre-generated OpenPCDet info pickles for immediate evaluation. Use `scripts/execute_evaluation.py` for cross-dataset analysis.

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
- scikit-learn >= 0.24

**Full pipeline** (real data, requires GPU):
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) >= 0.6
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

@article{patel2025dst,
  title={Uncertainty Representation in a SOTIF-Related Use Case with
         Dempster-Shafer Theory for LiDAR Sensor-Based Object Detection},
  author={Patel, Milin and Jung, Rolf},
  journal={arXiv preprint arXiv:2503.02087},
  year={2025}
}
```

## References

- ISO 21448:2022 - Safety of the Intended Functionality (SOTIF)
- ISO/PAS 8800:2024 - Safety for AI-based systems in road vehicles
- Lakshminarayanan et al. (2017) - Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
- Gal and Ghahramani (2016) - Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
- Feng et al. (2018, 2021) - Uncertainty estimation for LiDAR 3D detection
- Yan et al. (2018) - SECOND: Sparsely Embedded Convolutional Detection
- Pitropov et al. (2022) - LiDAR-MIMO: Efficient Uncertainty Estimation for LiDAR 3D Object Detection
- Patel and Jung (2024) - SOTIF-relevant evaluation of LiDAR detectors in CARLA
- Shafer (1976) - A Mathematical Theory of Evidence (Dempster-Shafer Theory)
- OpenPCDet (2020) - Open-source 3D object detection codebase

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
