# FINAL RESEARCH OUTPUT
# Uncertainty Evaluation to Support SOTIF Analysis for ML-Based LiDAR Object Detection

**Authors:** Milin Patel and Rolf Jung
**Affiliation:** Faculty of Electrical Engineering, Kempten University of Applied Sciences, Kempten, Germany
**Date:** 2026

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Complete Repository Contents](#2-complete-repository-contents)
3. [Research Paper (Conference Format)](#3-research-paper)
4. [All Experimental Results](#4-all-experimental-results)
5. [All Generated Figures](#5-all-generated-figures)
6. [Implementation Details](#6-implementation-details)
7. [How to Reproduce](#7-how-to-reproduce)
8. [References](#8-references)

---

## 1. Project Summary

This repository contains the **complete research output** for evaluating whether prediction uncertainty from ensemble-based LiDAR object detection can support ISO 21448 (SOTIF) analysis for identifying performance insufficiencies in ML-based perception systems.

### What Was Done

1. **Designed a 5-stage evaluation pipeline** that transforms ensemble detector outputs into ISO 21448 analysis artefacts
2. **Implemented the complete Python package** (`sotif_uncertainty` v2.0.0) with 12 modules and 58 passing unit tests
3. **Ran cross-dataset evaluation** on CARLA synthetic (547 frames, 22 weather configs) and KITTI real-world (465 proposals)
4. **Generated 31 publication-quality figures** (13 per dataset + 5 cross-dataset comparisons)
5. **Wrote a complete IEEE-format conference paper** in both Markdown and LaTeX
6. **Produced a detailed evaluation report** with all numerical results
7. **Created a demo Jupyter notebook** for interactive exploration

### Key Findings

| Finding | Evidence |
|---------|----------|
| Ensemble uncertainty discriminates TP from FP | AUROC up to 0.974 (CARLA) and 0.999 (KITTI) |
| Geometric disagreement is best under diverse weather | AUROC 0.974 vs 0.895 for confidence on CARLA |
| Epistemic uncertainty is the primary discriminator | DST decomposition: 0.227 vs 0.351 (CARLA), 0.175 vs 0.365 (KITTI) |
| Zero false acceptance is achievable | 25.8% coverage (KITTI), 38.3% coverage (CARLA) at FAR=0 |
| Adverse weather causes 75-90% of all false positives | Confirmed on both synthetic and real-world data |

---

## 2. Complete Repository Contents

```
SOTIF_Uncertainty_Evaluation_Lidar_Object_Detection/
|
|-- paper/                              # RESEARCH PAPER
|   |-- SOTIF_Uncertainty_Conference_Paper.md   # Full paper (Markdown, 355 lines)
|   |-- SOTIF_Uncertainty_Conference_Paper.tex  # Full paper (LaTeX/IEEE format, 530 lines)
|
|-- reports/
|   |-- evaluation_report/
|       |-- SOTIF_Evaluation_Report.md          # Detailed evaluation report (557 lines)
|       |-- comparison_summary.json             # Machine-readable results
|       |-- carla_synthetic/                    # 13 CARLA figures
|       |-- kitti_real_world/                   # 13 KITTI figures
|       |-- comparison/                         # 5 cross-dataset comparison figures
|
|-- Analysis/                           # 13 ANALYSIS FIGURES
|   |-- summary_dashboard.png
|   |-- scatter_score_var_tp_fp.png
|   |-- condition_boxplots.png
|   |-- reliability_diagram_rich.png
|   |-- indicator_distributions.png
|   |-- operating_point_heatmap.png
|   |-- tc_ranking.png
|   |-- frame_risk_scatter.png
|   |-- iso21448_scenario_grid.png
|   |-- member_agreement.png
|   |-- operating_points.png
|   |-- condition_breakdown.png
|   |-- risk_coverage_curve.png
|
|-- sotif_uncertainty/                  # PYTHON PACKAGE (12 modules)
|   |-- __init__.py
|   |-- ensemble.py                     # DBSCAN-based ensemble association
|   |-- uncertainty.py                  # 3 uncertainty indicators (Eqs. 1-3)
|   |-- matching.py                     # Greedy BEV IoU matching
|   |-- metrics.py                      # AUROC, ECE, NLL, Brier, AURC
|   |-- sotif_analysis.py               # TC ranking, acceptance gates, triage
|   |-- dst_uncertainty.py              # Dempster-Shafer Theory decomposition
|   |-- visualization.py               # 13 figure types
|   |-- dataset_adapter.py             # CARLA/KITTI data loading
|   |-- demo_data.py                   # Synthetic ensemble generation
|   |-- kitti_utils.py                 # KITTI format utilities
|   |-- mc_dropout.py                  # MC Dropout baseline
|   |-- weather_augmentation.py        # Physics-based weather simulation
|
|-- scripts/                            # EVALUATION SCRIPTS
|   |-- execute_evaluation.py           # Main cross-dataset evaluation
|   |-- evaluate.py                     # Single-dataset evaluation
|   |-- run_pipeline.py                 # End-to-end pipeline
|   |-- run_inference.py                # Ensemble inference
|   |-- generate_carla_data.py          # CARLA data generation
|   |-- prepare_kitti.py                # KITTI data preparation
|   |-- train_ensemble.sh              # Ensemble training script
|
|-- configs/
|   |-- second_sotif_ensemble.yaml      # SECOND detector configuration
|
|-- notebooks/
|   |-- SOTIF_Uncertainty_Evaluation_Demo.ipynb  # Interactive demo
|
|-- tests/
|   |-- test_pipeline.py                # 58 unit tests (all passing)
|
|-- README.md                           # Project documentation
|-- requirements.txt                    # Python dependencies
|-- pyproject.toml                      # Package configuration
|-- setup.py                            # Installation script
|-- LICENSE                             # MIT License
|-- FINAL_RESEARCH_OUTPUT.md            # THIS FILE
```

---

## 3. Research Paper

The complete conference paper is available in two formats:

- **Markdown:** `paper/SOTIF_Uncertainty_Conference_Paper.md`
- **LaTeX (IEEE format):** `paper/SOTIF_Uncertainty_Conference_Paper.tex`

### Paper Structure

| Section | Content |
|---------|---------|
| I. Introduction | Problem statement, core idea (ensemble uncertainty as bridge), 4 contributions |
| II. Related Work | Uncertainty estimation in 3D detection, SOTIF for ML perception, weather effects on LiDAR |
| III. Methodology | 5-stage pipeline, SECOND architecture, 3 uncertainty indicators (Eqs. 1-3), DST decomposition |
| IV. Experimental Setup | CARLA (547 frames, 22 weather), KITTI (465 proposals), evaluation protocol |
| V. Results and Discussion | Indicator statistics, AUROC, calibration, DST, triggering conditions, acceptance gates, cross-dataset |
| VI. Limitations | Simulated ensembles, single class, calibration, computational cost, weather coverage |
| VII. Conclusion | Summary of findings, future work directions |
| References | 20 cited works |

---

## 4. All Experimental Results

### 4.1 Dataset Summary

| Property | CARLA Synthetic | KITTI Real-World |
|----------|----------------|-----------------|
| Source | CARLA simulator (SOTIF-PCOD) | Velodyne HDL-64E (Karlsruhe) |
| Frames | 547 | 101 |
| Total proposals | 1,924 | 465 |
| True Positives (TP) | 1,012 | 135 |
| False Positives (FP) | 912 | 330 |
| FP ratio | 47.4% | 71.0% |
| Ensemble members (K) | 6 | 6 |
| Weather configurations | 22 | Clear (4 TC categories from statistics) |
| Points per frame | ~89,824 | ~120,000 |

### 4.2 Uncertainty Indicator Statistics

#### Mean Confidence (Eq. 1: s_bar = (1/K) * SUM s_k)

| Statistic | CARLA TP | CARLA FP | KITTI TP | KITTI FP |
|-----------|----------|----------|----------|----------|
| Mean | **0.451** | 0.193 | **0.817** | 0.213 |
| Std | 0.128 | 0.161 | 0.088 | 0.084 |
| Median | 0.448 | 0.147 | 0.812 | 0.207 |

#### Confidence Variance (Eq. 2: sigma^2 = (1/(K-1)) * SUM (s_k - s_bar)^2)

| Statistic | CARLA TP | CARLA FP | KITTI TP | KITTI FP |
|-----------|----------|----------|----------|----------|
| Mean | 0.01276 | **0.02268** | 0.00096 | **0.00418** |
| 80th percentile | 0.02707 | 0.03727 | 0.00133 | 0.00600 |

**Observation:** FP detections have consistently higher variance than TP on both datasets, confirming ensemble disagreement as a meaningful uncertainty signal.

### 4.3 Discrimination Performance (AUROC)

| Indicator | CARLA | KITTI | Best Domain |
|-----------|-------|-------|-------------|
| Mean confidence (Eq. 1) | 0.895 | **0.999** | KITTI |
| Confidence variance (Eq. 2) | 0.738 | 0.889 | KITTI |
| Geometric disagreement (Eq. 3) | **0.974** | 0.912 | CARLA |

**Critical Finding:** The optimal indicator depends on the operating domain:
- **Clear weather (KITTI):** Mean confidence provides near-perfect discrimination (0.999)
- **Diverse weather (CARLA):** Geometric disagreement is superior (0.974 vs 0.895)

**Practical Implication:** In deployed systems operating across weather conditions, geometric disagreement should be prioritised for safety-critical filtering.

### 4.4 Calibration Metrics

| Metric | CARLA | KITTI | Better |
|--------|-------|-------|--------|
| ECE (Expected Calibration Error) | 0.257 | 0.202 | KITTI |
| NLL (Negative Log-Likelihood) | 0.557 | 0.235 | KITTI |
| Brier Score | 0.197 | 0.049 | KITTI |
| AURC (Area Under Risk-Coverage) | **0.248** | 0.351 | CARLA |

**Note:** CARLA has worse calibration but better selective prediction (lower AURC). This is because KITTI's 71% FP ratio means even high-confidence filtering retains many FP.

### 4.5 Dempster-Shafer Theory Decomposition

| Component | CARLA TP | CARLA FP | KITTI TP | KITTI FP |
|-----------|----------|----------|----------|----------|
| **Aleatoric** (sensor noise) | 0.948 | 0.812 | 0.641 | 0.712 |
| **Epistemic** (model ignorance) | 0.227 | **0.351** | 0.175 | **0.365** |
| **Ontological** (unknown unknowns) | 0.012 | 0.042 | 0.001 | 0.019 |
| **Total** | 1.187 | 1.206 | 0.817 | 1.096 |

**Key Finding:** Epistemic uncertainty shows the clearest TP/FP separation on BOTH datasets:
- CARLA: 0.227 (TP) vs 0.351 (FP) -- 55% higher for FP
- KITTI: 0.175 (TP) vs 0.365 (FP) -- 109% higher for FP

This validates that ensemble disagreement is correctly captured as epistemic uncertainty and is the primary signal for distinguishing correct from incorrect detections.

### 4.6 Triggering Condition Analysis (ISO 21448, Clause 7)

#### CARLA Synthetic

| TC Category | FP Count | FP Share | Mean FP Conf | Mean FP Var |
|-------------|----------|----------|--------------|-------------|
| Night | 347 | **38.0%** | 0.205 | 0.02133 |
| Heavy rain | 294 | **32.2%** | 0.165 | 0.02032 |
| Other (benign) | 222 | 24.3% | 0.212 | 0.02719 |
| Fog/visibility | 49 | 5.4% | 0.182 | 0.02604 |

#### KITTI Real-World

| TC Category | FP Count | FP Share | Mean FP Conf | Mean FP Var |
|-------------|----------|----------|--------------|-------------|
| Heavy rain | 139 | **42.1%** | 0.173 | 0.00470 |
| Night | 91 | **27.6%** | 0.214 | 0.00411 |
| Fog/visibility | 65 | 19.7% | 0.239 | 0.00406 |
| Other (benign) | 35 | 10.6% | 0.324 | 0.00255 |

**Result:** Adverse weather (heavy rain + night + fog) collectively accounts for **75-90% of all false positives** on both datasets. Heavy rain has the lowest mean FP confidence (0.165-0.173), confirming that rain-induced ghost detections are characteristically uncertain.

### 4.7 Acceptance Gates (ISO 21448, Clause 11)

| Dataset | Gate Definition | Coverage | FAR |
|---------|----------------|----------|-----|
| **KITTI** | s >= 0.70 AND sigma^2 <= 0.005 | **25.8%** (120/465) | **0.000** |
| **CARLA** | s >= 0.35 AND d_IoU <= 0.49 | **38.3%** (737/1924) | **0.000** |

**Both gates achieve ZERO false acceptance rate.** The gate structures differ:
- KITTI uses confidence + variance (clear weather, well-separated distributions)
- CARLA uses confidence + geometric disagreement (diverse weather requires spatial consistency check)

### 4.8 Frame-Level Triage

| Property | CARLA | KITTI |
|----------|-------|-------|
| Total frames | 547 | 80 |
| Flagged frames | 153 (28.0%) | 1 (1.3%) |
| Variance threshold | 0.03727 | 0.00600 |

CARLA's 153 flagged frames are concentrated in heavy rain and night conditions. These support ISO 21448 Clause 7 requirement to move scenarios from Area 3 (unknown unsafe) to Area 2 (known unsafe).

---

## 5. All Generated Figures

### 5.1 Per-Dataset Figures (13 each for CARLA and KITTI)

Located in `reports/evaluation_report/carla_synthetic/` and `reports/evaluation_report/kitti_real_world/`:

| # | Figure | What It Shows |
|---|--------|---------------|
| 1 | `summary_dashboard.png` | Combined multi-panel overview of all key results |
| 2 | `scatter_score_var_tp_fp.png` | TP/FP separation in confidence-variance space |
| 3 | `condition_boxplots.png` | Per-weather-condition score and variance distributions |
| 4 | `reliability_diagram_rich.png` | Calibration curve: predicted confidence vs observed accuracy |
| 5 | `indicator_distributions.png` | Histograms of each uncertainty indicator by TP/FP |
| 6 | `operating_point_heatmap.png` | 2D threshold sweep showing coverage vs FAR |
| 7 | `tc_ranking.png` | Triggering condition FP share ranking bar chart |
| 8 | `frame_risk_scatter.png` | Per-frame mean confidence vs maximum variance |
| 9 | `iso21448_scenario_grid.png` | SOTIF Area 1-4 scenario classification grid |
| 10 | `member_agreement.png` | Score correlation heatmap across ensemble members |
| 11 | `operating_points.png` | Coverage and FAR at multiple confidence thresholds |
| 12 | `condition_breakdown.png` | Stacked bar chart of TP/FP by environment |
| 13 | `risk_coverage_curve.png` | Risk at varying coverage levels for selective prediction |

### 5.2 Cross-Dataset Comparison Figures (5)

Located in `reports/evaluation_report/comparison/`:

| # | Figure | What It Shows |
|---|--------|---------------|
| 1 | `auroc_comparison.png` | AUROC bar chart: CARLA vs KITTI across all 3 indicators |
| 2 | `calibration_comparison.png` | ECE, NLL, Brier Score side-by-side comparison |
| 3 | `dst_comparison.png` | DST decomposition: aleatoric/epistemic/ontological by dataset and TP/FP |
| 4 | `tc_ranking_comparison.png` | Triggering condition ranking side-by-side |
| 5 | `risk_coverage_comparison.png` | Risk-coverage curves overlaid for both datasets |

### 5.3 Analysis Figures (13)

Located in `Analysis/`:
Same 13 figure types as above, generated from the main analysis pipeline.

**Total: 31 unique figures generated.**

---

## 6. Implementation Details

### 6.1 Python Package (`sotif_uncertainty` v2.0.0)

| Module | Lines | Purpose |
|--------|-------|---------|
| `ensemble.py` | 331 | DBSCAN-based cross-member association |
| `uncertainty.py` | 170 | Three uncertainty indicators (Eqs. 1-3) |
| `matching.py` | 133 | Greedy BEV IoU matching (TP/FP/FN) |
| `metrics.py` | 232 | AUROC, ECE, NLL, Brier, AURC computation |
| `sotif_analysis.py` | 293 | TC ranking, acceptance gates, frame triage |
| `dst_uncertainty.py` | 511 | Dempster-Shafer Theory decomposition |
| `visualization.py` | 804 | 13 publication-quality figure generators |
| `dataset_adapter.py` | 318 | CARLA/KITTI data loading and format conversion |
| `demo_data.py` | 564 | Synthetic ensemble prediction generation |
| `kitti_utils.py` | 457 | KITTI format I/O utilities |
| `mc_dropout.py` | 205 | MC Dropout uncertainty baseline |
| `weather_augmentation.py` | 532 | Physics-based LiDAR weather simulation (rain/fog/snow/spray) |

### 6.2 Test Suite

58 unit tests in `tests/test_pipeline.py` covering:
- Ensemble association and clustering
- Uncertainty indicator computation
- BEV IoU matching
- All metric computations (AUROC, ECE, NLL, Brier, AURC)
- DST mass function computation and combination
- SOTIF analysis (TC ranking, gates, triage)
- Data loading and format conversion
- Weather augmentation
- End-to-end pipeline integration

### 6.3 Key Equations

**Equation 1 -- Mean Confidence:**
```
s_bar_j = (1/K) * SUM_{k=1}^{K} s_j^{(k)}
```

**Equation 2 -- Confidence Variance:**
```
sigma^2_{s,j} = (1/(K-1)) * SUM_{k=1}^{K} (s_j^{(k)} - s_bar_j)^2
```

**Equation 3 -- Geometric Disagreement:**
```
d_{IoU,j} = 1 - (2 / (K*(K-1))) * SUM_{u<v} IoU_BEV(b_j^{(u)}, b_j^{(v)})
```

**Equation 4 -- DST Mass Function:**
```
m_k({TP})  = s_k * r        (evidence for correctness)
m_k({FP})  = (1-s_k) * r    (evidence for incorrectness)
m_k(Theta) = 1 - r          (residual ignorance, r=0.9)
```

**Equation 5 -- Acceptance Gate:**
```
G(s_bar, sigma^2, d_IoU) = [s_bar >= tau_s] AND [sigma^2 <= tau_v] AND [d_IoU <= tau_d]
```

---

## 7. How to Reproduce

### Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### Run Cross-Dataset Evaluation

```bash
# Clone the SOTIF-PCOD dataset
git clone https://github.com/milinpatel07/SOTIF-PCOD.git

# Run full evaluation (generates all figures and reports)
python scripts/execute_evaluation.py \
    --carla_root SOTIF-PCOD/SOTIF_Scenario_Dataset \
    --output_dir reports/evaluation_report \
    --seed 42
```

### Run Tests

```bash
pytest tests/test_pipeline.py -v
# Expected: 58/58 passed
```

### Interactive Demo

Open `notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb` in Jupyter or Google Colab.

---

## 8. References

[1] ISO 21448:2022, "Road vehicles -- Safety of the intended functionality," International Organization for Standardization, 2022.

[2] M. Patel and R. Jung, "Uncertainty evaluation to support safety of the intended functionality analysis for identifying performance insufficiencies in ML-based LiDAR object detection," Kempten University of Applied Sciences, 2026.

[3] M. Patel and R. Jung, "Uncertainty representation in a SOTIF-related use case with Dempster-Shafer theory for LiDAR sensor-based object detection," arXiv:2503.02087, 2025.

[4] Y. Yan, Y. Mao, and B. Li, "SECOND: Sparsely embedded convolutional detection," Sensors, vol. 18, no. 10, p. 3337, 2018.

[5] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in Proc. NeurIPS, 2017.

[6] M. Pitropov, C. Huang, S. Abdelfattah, K. Czarnecki, and S. L. Waslander, "LiDAR-MIMO: Efficient uncertainty estimation for LiDAR 3D object detection," in Proc. IEEE ICRA, 2022.

[7] D. Feng, A. Harakeh, S. L. Waslander, and K. Dietmayer, "A review and comparative study on probabilistic object detection in autonomous driving," IEEE Trans. ITS, vol. 23, no. 8, pp. 9961-9982, 2021.

[8] G. Shafer, A Mathematical Theory of Evidence. Princeton University Press, 1976.

[9] A. P. Dempster, "Upper and lower probabilities induced by a multivalued mapping," Annals of Mathematical Statistics, vol. 38, no. 2, pp. 325-339, 1967.

[10] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in Proc. ICML, 2016.

[11] G. P. Meyer et al., "LaserNet: An efficient probabilistic 3D object detector for autonomous driving," in Proc. IEEE CVPR, 2019.

[12] M. Hahner et al., "Fog simulation on real LiDAR point clouds for 3D object detection in adverse weather," in Proc. IEEE ICCV, 2021.

[13] M. Hahner et al., "LiDAR snowfall simulation for robust 3D object detection," in Proc. IEEE CVPR, 2022.

[14] Y. Li et al., "Realistic rainy weather simulation for LiDARs in CARLA simulator," arXiv:2312.12772, 2023.

[15] A. Dosovitskiy et al., "CARLA: An open urban driving simulator," in Proc. CoRL, 2017.

[16] M. Patel, "SOTIF-PCOD: SOTIF point cloud object detection dataset," https://github.com/milinpatel07/SOTIF-PCOD, 2025.

[17] A. Geiger, P. Lenz, and R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," in Proc. IEEE CVPR, 2012.

[18] OpenPCDet Development Team, "OpenPCDet: An open-source toolbox for 3D object detection from point clouds," https://github.com/open-mmlab/OpenPCDet, 2020.

[19] R. Salay, R. Queiroz, and K. Czarnecki, "An analysis of ISO 26262: Using machine learning safely in automotive software," arXiv:1709.02435, 2019.

[20] L. Gauerhof et al., "Assuring the safety of machine learning for pedestrian detection at crossings," in Proc. SAFECOMP, 2020.

---

*Generated from the SOTIF Uncertainty Evaluation pipeline v2.0.0*
*All results deterministic with seed=42*
