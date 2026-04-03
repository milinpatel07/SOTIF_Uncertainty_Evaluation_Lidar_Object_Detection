# Uncertainty Evaluation for SOTIF Analysis of LiDAR Object Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the evaluation code for:

> **Uncertainty Evaluation to Support Safety of the Intended Functionality Analysis for Identifying Performance Insufficiencies in ML-Based LiDAR Object Detection**
>
> Milin Patel and Rolf Jung, VEHITS 2026 (Paper #74)

The pipeline generates simulated ensemble detection outputs from ground truth annotations with weather-dependent confidence perturbations, processes them through five evaluation stages, and produces all figures and tables reported in the paper. A single command reproduces the results.

---

## Reproduction

```bash
git clone https://github.com/milinpatel07/SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection.git
cd SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection
pip install -e .
python scripts/evaluate.py
```

This runs the CARLA case study evaluation (547 frames, 22 environmental configurations, K=6 ensemble members) with seed=42 and writes figures and a JSON results file to `results/`. No GPU is needed.

To run the test suite:

```bash
pip install pytest
pytest tests/ -v
```

All numerical results, tables, and ablation comparisons are written to `results/` by the evaluation scripts. See the paper for interpretation.

### Ablation Studies

```bash
python scripts/run_ablations.py
```

Produces DBSCAN vs WBF comparison, deep ensemble vs MC Dropout comparison, and confidence degradation sensitivity analysis. Results saved to `results/ablations/`.

---

## Figures

The pipeline generates 14 figures in the output directory:

| Filename | Description |
|---|---|
| `indicator_distributions.png` | Histogram per indicator, TP vs FP |
| `roc_curves.png` | ROC curves for all three indicators |
| `reliability_diagram_rich.png` | Predicted confidence vs observed accuracy (10 bins) |
| `risk_coverage_curve.png` | Risk-coverage curve |
| `tc_ranking.png` | FP share by triggering condition category |
| `operating_points.png` | Coverage vs FAR for acceptance gate configurations |
| `scatter_score_var_tp_fp.png` | Mean confidence vs variance scatter (TP/FP) |
| `frame_risk_scatter.png` | Per-frame detection count vs high-uncertainty FP count |
| `operating_point_heatmap.png` | FAR and coverage heatmaps over threshold grid |
| `condition_boxplots.png` | Per-condition box plots of indicators |
| `condition_breakdown.png` | TP/FP stacked bars by condition |
| `member_agreement.png` | Ensemble member detection agreement histograms |
| `iso21448_scenario_grid.png` | ISO 21448 Area 1-4 categorisation diagram |
| `summary_dashboard.png` | Multi-panel summary of all results |

---

## Methodology

Five-stage pipeline:

1. **Ensemble inference** (Stage 1): For each of 547 frames, generate K=6 detection sets from ground truth annotations with weather-dependent confidence perturbations, distance-dependent detection probability, and stochastic inter-member disagreement.

2. **Association and uncertainty** (Stage 2): DBSCAN clustering on BEV IoU distance matrix (eps=0.5, min_samples=4) produces unified proposals. Per proposal, compute mean confidence (Eq. 1), confidence variance (Eq. 2), and geometric disagreement (Eq. 3). Non-detecting members are zero-padded.

3. **Ground truth matching** (Stage 3): Greedy matching sorted by mean confidence descending, BEV IoU >= 0.5, one-to-one assignment. Produces TP/FP/FN labels.

4. **Metrics** (Stage 4): AUROC per indicator, ECE (10 bins), NLL, Brier score, AURC. Acceptance gate grid search over threshold combinations; coverage and FAR per configuration.

5. **SOTIF artefacts** (Stage 5): Triggering condition ranking by FP share (Clause 7), frame-level flagging at 80th percentile FP variance (Clause 7), acceptance gate operating point table (Clause 11).

### Uncertainty Indicators

| Indicator | Formula | Uncertainty Type |
|---|---|---|
| Mean confidence | (1/K) * sum of s_j^(k) | Existence |
| Confidence variance | (1/(K-1)) * sum of (s_j^(k) - mean)^2 | Epistemic |
| Geometric disagreement | 1 - mean pairwise BEV IoU | Localisation |

### Acceptance Gate

```
G(s_bar, sigma2, d_iou) = [s_bar >= tau_s] AND [sigma2 <= tau_v] AND [d_iou <= tau_d]
```

Grid search ranges: tau_s in [0.20, 0.80], tau_v in {0.002, 0.005, 0.010}, tau_d in {0.20, 0.30, 0.40, 0.50}.

---

## Repository Structure

```
.
├── sotif_uncertainty/           # Core Python package
│   ├── __init__.py              # Public API
│   ├── uncertainty.py           # Uncertainty indicators (Eqs. 1-3)
│   ├── ensemble.py              # DBSCAN clustering + uncertainty decomposition
│   ├── matching.py              # TP/FP/FN matching (BEV IoU >= 0.5)
│   ├── metrics.py               # AUROC, AURC, ECE, NLL, Brier Score
│   ├── sotif_analysis.py        # TC ranking, frame flags, acceptance gates
│   ├── visualization.py         # Figure generation (14 plot types)
│   ├── carla_case_study.py      # CARLA case study data generation (Section 5)
│   ├── demo_data.py             # Lightweight synthetic data for demos
│   ├── dst_uncertainty.py       # Extension: Dempster-Shafer Theory decomposition
│   ├── mc_dropout.py            # Extension: MC Dropout alternative
│   ├── kitti_utils.py           # Extension: KITTI calibration, label I/O
│   ├── weather_augmentation.py  # Extension: physics-based weather effects
│   └── dataset_adapter.py       # Extension: unified adapter for KITTI/CARLA
│
├── scripts/
│   ├── evaluate.py              # Main entry point (runs CARLA case study)
│   ├── run_ablations.py         # WBF comparison, MC Dropout, sensitivity analysis
│   ├── run_pipeline.py          # End-to-end pipeline with DST analysis
│   ├── execute_evaluation.py    # Cross-dataset evaluation with report
│   ├── run_inference.py         # Ensemble inference (requires OpenPCDet)
│   ├── train_ensemble.sh        # Train K SECOND models (requires OpenPCDet)
│   ├── prepare_kitti.py         # Download and prepare KITTI dataset
│   └── generate_carla_data.py   # Generate CARLA simulation data
│
├── tests/
│   └── test_pipeline.py         # 58 tests across all pipeline stages
│
├── configs/
│   └── second_sotif_ensemble.yaml  # OpenPCDet SECOND config for K=6 ensemble
│
├── notebooks/
│   └── SOTIF_Uncertainty_Evaluation_Demo.ipynb  # Interactive demo
│
├── paper/                       # Paper sources (.tex, .md)
├── requirements.txt
├── pyproject.toml
├── setup.py
└── LICENSE
```

Extension modules (`dst_uncertainty.py`, `mc_dropout.py`, `kitti_utils.py`, `weather_augmentation.py`, `dataset_adapter.py`) are not part of the VEHITS 2026 paper evaluation. They add Dempster-Shafer uncertainty decomposition, MC Dropout inference, KITTI format support, physics-based weather perturbations, and unified dataset handling.

---

## Real Data Pipeline

For use with trained ensemble models on KITTI or CARLA point clouds. Requires OpenPCDet >= 0.6, PyTorch >= 1.10, CUDA 11.x, and spconv v2.x.

```bash
# 1. Prepare dataset
python scripts/prepare_kitti.py --data_root data/kitti

# 2. Train K=6 ensemble
bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5

# 3. Run inference
python scripts/run_inference.py \
    --ckpt_dirs output/ensemble/seed_0 ... output/ensemble/seed_5 \
    --data_path data/kitti \
    --gt_path data/kitti/training/label_2 \
    --voting consensus

# 4. Evaluate
python scripts/evaluate.py \
    --input results/ensemble_results.pkl \
    --gt_path data/kitti/training/label_2 \
    --calib_path data/kitti/training/calib
```

---

## Dependencies

For reproduction (no GPU needed):
- numpy >= 1.20
- matplotlib >= 3.4
- scikit-learn >= 0.24

Install with `pip install -e .`

---

## Citation

```bibtex
@inproceedings{patel2026uncertainty,
  title={Uncertainty Evaluation to Support Safety of the Intended Functionality
         Analysis for Identifying Performance Insufficiencies in ML-Based
         LiDAR Object Detection},
  author={Patel, Milin and Jung, Rolf},
  booktitle={Proceedings of the 12th International Conference on Vehicle
             Technology and Intelligent Transport Systems (VEHITS)},
  year={2026}
}
```

## References

- ISO 21448:2022 -- Safety of the Intended Functionality (SOTIF)
- Lakshminarayanan et al. (2017) -- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
- Yan et al. (2018) -- SECOND: Sparsely Embedded Convolutional Detection

## License

MIT License -- see [LICENSE](LICENSE) for details.
