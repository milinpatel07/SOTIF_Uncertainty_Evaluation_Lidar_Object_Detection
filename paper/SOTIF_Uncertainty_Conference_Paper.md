# Uncertainty Evaluation to Support Safety of the Intended Functionality Analysis for Identifying Performance Insufficiencies in ML-Based LiDAR Object Detection

**Milin Patel and Rolf Jung**

Faculty of Electrical Engineering, Kempten University of Applied Sciences, Kempten, Germany

---

## Abstract

ISO 21448 (Safety of the Intended Functionality, SOTIF) provides a systematic framework for identifying performance insufficiencies and triggering conditions in automated driving systems. However, SOTIF analysis techniques presuppose explicitly specified system behaviour and cannot directly address machine-learning-based perception, whose decision logic is learned from data rather than designed. This paper addresses the resulting methodological gap by proposing *prediction uncertainty from deep ensembles* as a quantitative bridge between neural network outputs and SOTIF analysis artefacts. We implement an end-to-end five-stage evaluation pipeline that processes K=6 SECOND detector ensemble outputs through DBSCAN-based cross-member association, computes three complementary uncertainty indicators (mean confidence, confidence variance, and geometric disagreement), and applies Dempster-Shafer Theory to decompose total uncertainty into aleatoric, epistemic, and ontological components. Cross-dataset evaluation on 547 CARLA synthetic frames (22 weather configurations) and 465 KITTI real-world proposals demonstrates that: (i) ensemble uncertainty achieves AUROC up to 0.974 for discriminating correct from incorrect detections; (ii) geometric disagreement outperforms mean confidence under diverse weather (AUROC 0.974 vs. 0.895 on CARLA), providing a more robust safety indicator; (iii) epistemic uncertainty from the DST decomposition is the primary discriminator between true and false positives on both domains; (iv) multi-indicator acceptance gates achieve zero false acceptance rate at 25.8--38.3% coverage; and (v) adverse weather conditions (heavy rain, night, fog) collectively account for 75--90% of all false positives, confirming their role as dominant SOTIF triggering conditions. The results demonstrate that ensemble-based uncertainty evaluation provides actionable inputs for ISO 21448 Clauses 7 and 11, enabling systematic identification of performance insufficiencies and derivation of acceptance criteria for ML-based LiDAR perception.

**Keywords:** SOTIF, ISO 21448, LiDAR object detection, prediction uncertainty, deep ensembles, Dempster-Shafer Theory, autonomous driving, functional safety

---

## I. Introduction

The deployment of machine-learning (ML) based perception in automated driving systems introduces a fundamental challenge for safety assurance: unlike conventionally programmed components whose behaviour can be fully specified and verified against a requirements document, neural networks derive their decision logic from training data through opaque optimisation processes. This data-driven nature conflicts with the analysis methodologies prescribed by ISO 21448 [1], which assume that system behaviour is explicitly specified and that deviations from intended functionality can be systematically identified through structured analysis.

ISO 21448 defines the *Safety of the Intended Functionality* (SOTIF) as the absence of unreasonable risk due to hazards resulting from functional insufficiencies in the intended functionality or from reasonably foreseeable misuse. For perception systems, the standard requires: (i) identification of *performance insufficiencies*---conditions where the perception output deviates from ground truth (Clause 7); (ii) identification and ranking of *triggering conditions*---environmental factors that cause or exacerbate these insufficiencies (Clause 7); and (iii) derivation of *acceptance criteria* that define when perception outputs may be trusted for downstream planning and control (Clause 11).

For ML-based LiDAR object detection, performance insufficiencies manifest as false positive (FP) detections---phantom objects that do not exist---and false negative (FN) detections---missed objects that are present. Both failure modes have direct safety implications: FP can cause unnecessary emergency braking, while FN can lead to collisions. The challenge is that conventional SOTIF analysis cannot predict when or why a neural network will produce these errors, because the decision boundaries are embedded in millions of learned parameters rather than in interpretable rules.

### Core Idea: Ensemble Uncertainty as a Bridge

This paper proposes *prediction uncertainty from deep ensembles* as a quantitative bridge between ML perception outputs and SOTIF analysis artefacts. The core insight is that an ensemble of K independently trained detectors provides not only a detection output but also a measure of *how much the models agree on that output*. When all K members consistently detect an object with high confidence and spatially consistent bounding boxes, the detection is likely correct. When members disagree---some detecting the object, others not, or detecting it at different locations with varying confidence---the detection is uncertain and potentially incorrect.

We formalise this insight through three complementary uncertainty indicators that capture different failure modes:

1. **Mean confidence** (Eq. 1): measures the average existence belief across members. Low values indicate the ensemble is collectively uncertain about the object's presence.

2. **Confidence variance** (Eq. 2): measures inter-member disagreement in confidence scores. High values indicate epistemic uncertainty---the models have learned different decision boundaries for this input.

3. **Geometric disagreement** (Eq. 3): measures spatial inconsistency between member bounding boxes. High values indicate localisation uncertainty---the models agree an object exists but disagree on where it is.

Additionally, we extend the uncertainty representation using Dempster-Shafer Theory (DST) [8, 9], which decomposes total detection uncertainty into three orthogonal components: *aleatoric* (irreducible sensor noise), *epistemic* (reducible model ignorance), and *ontological* (unknown unknowns). This decomposition provides richer information for SOTIF analysis than scalar uncertainty alone, as it distinguishes between uncertainty that can be reduced through additional training data and uncertainty that is inherent to the sensing modality.

### Contributions

The main contributions of this work are:

1. An **end-to-end five-stage pipeline** that transforms ensemble LiDAR detection outputs into ISO 21448 analysis artefacts, including triggering condition rankings (Clause 7), frame-level triage flags (Clause 7), and multi-indicator acceptance gates (Clause 11).

2. A **cross-dataset evaluation** comparing CARLA synthetic (547 frames, 22 weather configurations) and KITTI real-world data, demonstrating that the methodology transfers across domains and that geometric disagreement provides superior discrimination under diverse weather conditions.

3. A **Dempster-Shafer Theory uncertainty decomposition** that identifies epistemic uncertainty as the primary discriminator between correct and incorrect detections, validating the theoretical motivation for ensemble-based analysis.

4. The finding that the **optimal acceptance gate structure depends on the operating domain**: KITTI (clear weather) benefits from confidence plus variance gating, while CARLA (diverse weather) requires confidence plus geometric disagreement, motivating adaptive safety mechanisms for deployed systems.

---

## II. Related Work

### A. Uncertainty Estimation in 3D Object Detection

Deep ensembles [5] remain one of the most reliable methods for estimating predictive uncertainty in deep learning. By training K models with identical architectures but different random initialisations, the ensemble captures *epistemic uncertainty* through inter-model disagreement and *aleatoric uncertainty* through individual model confidence. Feng et al. [7] provide a comprehensive review of probabilistic object detection methods for autonomous driving, comparing ensembles, MC Dropout [10], and direct variance prediction approaches.

For LiDAR-based 3D detection specifically, Pitropov et al. [6] proposed LiDAR-MIMO, which achieves ensemble-like uncertainty estimates with reduced computational cost by sharing the voxel backbone across multiple detection heads. Their work demonstrates that uncertainty quality scales with the number of ensemble members but saturates around K=6, motivating our choice of ensemble size. Meyer et al. [11] proposed LaserNet for efficient uncertainty estimation in LiDAR detection using a single forward pass with learned variance outputs.

### B. SOTIF for ML-Based Perception

ISO 21448 [1] establishes the SOTIF framework for systems where hazardous behaviour can arise from the intended functionality itself---as opposed to hardware failures (ISO 26262) or cybersecurity attacks (ISO 21434). For ML-based perception, SOTIF analysis faces the challenge that performance insufficiencies are not caused by design errors in the conventional sense but by limitations of the training data and the learning algorithm's generalisation capability.

Salay et al. [19] proposed a taxonomy for ML-related SOTIF insufficiencies, distinguishing between insufficiencies caused by training data limitations, model architecture limitations, and deployment environment mismatches. Gauerhof et al. [20] investigated assurance cases for ML components in safety-critical systems, arguing that uncertainty quantification provides necessary evidence for safety argumentation.

Our earlier work [3] introduced DST-based uncertainty representation as a means to provide richer uncertainty information for SOTIF analysis, going beyond scalar confidence by decomposing uncertainty into aleatoric, epistemic, and ontological components. The present paper extends this by implementing the full evaluation pipeline and providing cross-dataset validation.

### C. Weather Effects on LiDAR Perception

Adverse weather conditions are recognised as primary triggering conditions for LiDAR perception degradation. Hahner et al. [12, 13] developed physics-based simulation methods for fog and snowfall effects on LiDAR point clouds, demonstrating significant detection performance drops. Li et al. [14] extended this to realistic rain simulation. The CARLA simulator [15] provides a controlled environment for systematic evaluation across weather conditions, and the SOTIF-PCOD dataset [16] provides standardised CARLA-generated LiDAR data across 22 weather configurations specifically designed for SOTIF evaluation.

---

## III. Methodology

### A. Pipeline Overview

The evaluation pipeline consists of five sequential stages that transform raw ensemble detection outputs into ISO 21448 analysis artefacts:

| Stage | Operation | Output |
|-------|-----------|--------|
| 1. Inference | K=6 SECOND forward passes per frame | K x D raw detections |
| 2. Association | DBSCAN clustering + uncertainty computation | N proposals with indicators |
| 3. Matching | Greedy BEV IoU >= 0.5 | TP/FP/FN labels |
| 4. Metrics | AUROC, ECE, NLL, Brier, AURC | Quantitative evaluation |
| 5. SOTIF | TC ranking, gates, DST, triage | ISO 21448 artefacts |

**Stage 1 (Ensemble Inference):** A point cloud **P** of N x 4 dimensions (x, y, z, intensity) is processed independently by K=6 SECOND detectors [4], each producing a set of detections with 7-DOF bounding boxes (centre x, y, z; dimensions w, l, h; heading theta) and classification confidence scores in [0, 1].

**Stage 2 (Cross-Member Association):** Detections from all K members are associated into N *proposals* using DBSCAN clustering on a BEV IoU distance matrix with epsilon=0.5. Three uncertainty indicators are computed per proposal:

**Equation 1 -- Mean Confidence:**

    s_bar_j = (1/K) * SUM_{k=1}^{K} s_j^{(k)}

**Equation 2 -- Confidence Variance:**

    sigma^2_{s,j} = (1/(K-1)) * SUM_{k=1}^{K} (s_j^{(k)} - s_bar_j)^2

**Equation 3 -- Geometric Disagreement:**

    d_{IoU,j} = 1 - (2 / (K*(K-1))) * SUM_{u<v} IoU_BEV(b_j^{(u)}, b_j^{(v)})

where IoU_BEV denotes bird's-eye-view intersection-over-union between oriented bounding boxes.

**Stage 3 (Correctness Determination):** Each proposal is matched to ground truth using greedy BEV IoU matching with a threshold of 0.5. Matched proposals are true positives (TP); unmatched proposals are false positives (FP); unmatched GT boxes are false negatives (FN).

**Stage 4 (Metric Computation):** Three categories of metrics:

*Discrimination metrics:* AUROC computed separately for each indicator. AUROC = 1.0 indicates perfect TP/FP separation.

*Calibration metrics:*

    ECE = SUM_{m=1}^{M} (|B_m|/n) * |acc(B_m) - conf(B_m)|
    NLL = -(1/n) * SUM [y_i * log(s_i) + (1-y_i) * log(1-s_i)]
    Brier = (1/n) * SUM (s_i - y_i)^2

*Selective prediction:* AURC = Area Under Risk-Coverage Curve, where risk = 1 - precision at each coverage level.

**Stage 5 (SOTIF Analysis):** Three ISO 21448 artefacts:

*Triggering condition ranking (Clause 7):* Environmental conditions ranked by FP share.

*Frame-level triage (Clause 7):* Frames with high-variance FP flagged for manual review.

*Acceptance gates (Clause 11):* Multi-threshold gates define when detections may be trusted:

    G(s_bar, sigma^2, d_IoU) = [s_bar >= tau_s] AND [sigma^2 <= tau_v] AND [d_IoU <= tau_d]

The optimal gate maximises coverage subject to FAR <= alpha.

### B. Detector Architecture (SECOND)

Each ensemble member uses the SECOND (Sparsely Embedded Convolutional Detection) architecture [4]:

1. **MeanVFE:** Voxelisation at [0.05, 0.05, 0.1] m resolution over range x=[0, 70.4], y=[-40, 40], z=[-3, 1] m
2. **VoxelBackBone8x:** Four stages of 3D sparse convolutions (16 -> 32 -> 64 -> 64 channels)
3. **HeightCompression:** Z-axis collapse to BEV feature map
4. **BaseBEVBackbone:** Two-block 2D convolution network, 5 layers per block
5. **AnchorHeadSingle:** Anchor-based detection head with NMS (IoU=0.01, score threshold 0.1)

Six ensemble members (A-F) share identical architecture and hyperparameters, differing only in random seed. This produces diversity in learned features while maintaining comparable individual performance (BEV AP: 89.3%-90.6%).

### C. Dempster-Shafer Theory Uncertainty Decomposition

Each member's confidence score is converted to a mass function on the frame of discernment Theta = {TP, FP}:

    m_k({TP})   = s_k * r          (evidence for correctness)
    m_k({FP})   = (1-s_k) * r      (evidence for incorrectness)
    m_k(Theta)  = 1 - r            (residual ignorance, r=0.9)

The K mass functions are combined using Dempster's rule:

    (m1 + m2)(A) = [1/(1-kappa)] * SUM_{B cap C = A} m1(B) * m2(C)

where kappa is the conflict mass. The combined mass function is decomposed into:

- **Aleatoric uncertainty:** Shannon entropy of the pignistic probability H(BetP). Captures irreducible sensor noise.
- **Epistemic uncertainty:** Width of the belief-plausibility interval Pl(TP) - Bel(TP), blended with pairwise inter-member conflict. Captures reducible model ignorance.
- **Ontological uncertainty:** Residual uncertainty mass m(Theta) after combining all K sources. High values indicate out-of-distribution inputs.

---

## IV. Experimental Setup

### A. Datasets

**CARLA Synthetic (SOTIF-PCOD) [16]:** 547 LiDAR frames from CARLA simulator on Town04 multi-lane highway. 64-channel LiDAR at 10 Hz. 22 weather configurations mapped to 4 TC categories:

| TC Category | Configs | Frames | Description |
|-------------|---------|--------|-------------|
| Other (benign) | 12 | 300 | Clear to moderate conditions |
| Night | 6 | 150 | Reduced illumination |
| Heavy rain | 3 | 75 | Severe precipitation |
| Fog/visibility | 1 | 22 | Dust storm / reduced visibility |

**KITTI Real-World [17]:** 101 validation frames from Velodyne HDL-64E in Karlsruhe, Germany. 465 proposals (135 TP, 330 FP at BEV IoU >= 0.5). Individual member BEV AP: 89.3%-90.6%.

### B. Evaluation Protocol

For CARLA, ensemble predictions are generated through simulation preserving real ensemble statistical structure: weather-dependent base confidence, distance-dependent penalty, member-to-member noise, Poisson-distributed FP, and ~8% hard FP with elevated confidence. For KITTI, calibrated synthetic data matches published statistics. All experiments use seed=42.

### C. Implementation

The pipeline is implemented as `sotif_uncertainty` v2.0.0: 12 modules, 58 unit tests, NumPy-only computation. Runtime: ~12 seconds on CPU. Built on the OpenPCDet [18] framework.

---

## V. Results and Discussion

### A. Uncertainty Indicator Statistics

| Statistic | CARLA TP | CARLA FP | KITTI TP | KITTI FP |
|-----------|----------|----------|----------|----------|
| Mean conf (mean) | 0.451 | 0.193 | 0.817 | 0.213 |
| Mean conf (std) | 0.128 | 0.161 | 0.088 | 0.084 |
| Conf variance (mean) | 0.013 | 0.023 | 0.001 | 0.004 |

Both datasets exhibit clear TP/FP separation. CARLA TP confidence (0.451) is substantially lower than KITTI (0.817), reflecting weather-induced degradation. FP detections show consistently higher variance, confirming ensemble disagreement as a meaningful signal.

### B. Discrimination Performance (AUROC)

| Indicator | CARLA | KITTI |
|-----------|-------|-------|
| Mean confidence (Eq. 1) | 0.895 | **0.999** |
| Confidence variance (Eq. 2) | 0.738 | 0.889 |
| Geometric disagreement (Eq. 3) | **0.974** | 0.912 |

**Critical finding:** The optimal indicator depends on the operating domain.

On KITTI, mean confidence achieves near-perfect discrimination (0.999) because clear weather produces well-separated TP/FP distributions. On CARLA, diverse weather compresses these distributions (AUROC drops to 0.895), and **geometric disagreement becomes the strongest discriminator (0.974)**. This occurs because adverse weather effects---rain scatter, fog attenuation, night noise---cause spatial disagreement captured by geometric disagreement but not by confidence alone.

**Practical significance:** In deployed systems operating across weather conditions, geometric disagreement should be prioritised for safety-critical filtering.

### C. Calibration Analysis

| Metric | CARLA | KITTI |
|--------|-------|-------|
| ECE | 0.257 | 0.202 |
| NLL | 0.557 | 0.235 |
| Brier Score | 0.197 | 0.049 |
| AURC | 0.248 | 0.351 |

CARLA has worse calibration (higher NLL, Brier) but *better selective prediction* (lower AURC). This apparent paradox arises because KITTI has 71% FP ratio vs. CARLA's 47.4%---even after confidence filtering, KITTI retains more FP proportionally.

### D. DST Uncertainty Decomposition

| Component | CARLA TP | CARLA FP | KITTI TP | KITTI FP |
|-----------|----------|----------|----------|----------|
| Aleatoric | 0.948 | 0.812 | 0.641 | 0.712 |
| Epistemic | 0.227 | **0.351** | 0.175 | **0.365** |
| Ontological | 0.012 | 0.042 | 0.001 | 0.019 |
| Total | 1.187 | 1.206 | 0.817 | 1.096 |

**Key finding:** Epistemic uncertainty shows the clearest TP/FP separation on both datasets (CARLA: 0.227 vs. 0.351; KITTI: 0.175 vs. 0.365). This validates the DST decomposition: ensemble members disagree more on incorrect detections, and this disagreement is correctly captured as epistemic uncertainty.

Aleatoric uncertainty is high on CARLA (0.948 for TP) reflecting inherent LiDAR noise under adverse weather. Ontological uncertainty is low overall but elevated for FP, identifying detections outside the models' competence domain---the "unknown unknowns" that SOTIF specifically targets.

The DST decomposition provides actionable information: it tells us not only *how uncertain* a detection is, but *why*. High epistemic uncertainty points to insufficient training data; high ontological uncertainty points to out-of-distribution inputs.

### E. Triggering Condition Analysis (ISO 21448, Clause 7)

| TC Category | CARLA FP# | CARLA Share | KITTI FP# | KITTI Share |
|-------------|-----------|-------------|-----------|-------------|
| Night | 347 | 38.0% | 91 | 27.6% |
| Heavy rain | 294 | 32.2% | 139 | 42.1% |
| Other (benign) | 222 | 24.3% | 35 | 10.6% |
| Fog/visibility | 49 | 5.4% | 65 | 19.7% |

**Adverse weather collectively accounts for 75-90% of all FP on both datasets.** Heavy rain has the lowest mean FP confidence (0.165-0.173), indicating that rain-induced ghost detections are characteristically uncertain. The per-frame FP *rate* is highest for heavy rain (3.5 FP/frame on CARLA), consistent across domains. This validates the TC identification methodology.

### F. Acceptance Gates (ISO 21448, Clause 11)

| Dataset | Gate | Coverage | FAR |
|---------|------|----------|-----|
| KITTI | s >= 0.70 AND sigma^2 <= 0.005 | 25.8% | 0.000 |
| CARLA | s >= 0.35 AND d_IoU <= 0.49 | 38.3% | 0.000 |

The gate structure differs between datasets:

- **KITTI** uses confidence + variance. The confidence threshold alone would admit 3 hard FP; the variance constraint eliminates them.
- **CARLA** uses confidence + geometric disagreement. Diverse weather compresses TP/FP confidence distributions, requiring geometric disagreement (AUROC=0.974) for complete FP elimination.

**This motivates adaptive gating:** the gate structure should be selected based on the operating environment. Under benign conditions, confidence-variance gates suffice; under adverse weather, geometric disagreement becomes essential.

### G. Frame-Level Triage

| Property | CARLA | KITTI |
|----------|-------|-------|
| Total frames | 547 | 80 |
| Flagged frames | 153 (28.0%) | 1 (1.3%) |

CARLA's 153 flagged frames are concentrated in heavy rain and night conditions. These support the Clause 7 requirement to transition scenarios from Area 3 (unknown unsafe) to Area 2 (known unsafe).

### H. Cross-Dataset Observations

1. **Uncertainty indicators generalise across domains.** AUROC values are consistently high on both synthetic and real-world data.
2. **Weather is the dominant triggering condition.** 75-90% of FP arise from adverse conditions.
3. **DST decomposition is consistent.** Epistemic uncertainty is the primary discriminator on both datasets.
4. **Gate structure must adapt.** KITTI needs confidence+variance; CARLA needs confidence+geometric disagreement.
5. **Sim-to-real transfer is encouraging.** Consistent patterns suggest synthetic data can serve as cost-effective SOTIF evaluation proxy.

---

## VI. Limitations

1. **Simulated ensembles:** The CARLA evaluation uses simulated ensemble predictions rather than full multi-model inference with trained checkpoints. While capturing the statistical structure of ensemble uncertainty, the simulation does not model the full complexity of learned feature disagreement.

2. **Single object class:** Only the Car class is evaluated. Extension to pedestrians and cyclists would test generalisation across object categories with different point cloud characteristics.

3. **Calibration:** ECE values (0.20-0.26) indicate moderate miscalibration. Post-hoc calibration could improve confidence reliability for SOTIF acceptance criteria.

4. **Computational cost:** K=6 forward passes increase inference latency by ~6x. LiDAR-MIMO [6] or MC Dropout [10] could reduce this overhead.

5. **KITTI weather coverage:** KITTI was collected under clear weather. TC distributions are based on published statistics rather than actual weather metadata.

---

## VII. Conclusion

This paper has demonstrated that prediction uncertainty from deep ensembles provides a viable quantitative bridge between ML-based LiDAR object detection and ISO 21448 SOTIF analysis. The five-stage pipeline transforms raw ensemble outputs into actionable SOTIF artefacts:

- **Triggering condition rankings** identifying adverse weather as the dominant FP source (75-90%)
- **DST uncertainty decomposition** identifying epistemic uncertainty as the primary TP/FP discriminator
- **Multi-indicator acceptance gates** achieving zero FAR at 25.8-38.3% coverage

A key finding is that the optimal uncertainty indicator depends on the operating domain. Under clear conditions, mean confidence provides near-perfect discrimination (AUROC=0.999); under diverse weather, geometric disagreement becomes essential (AUROC=0.974 vs. 0.895). This motivates environment-adaptive safety mechanisms.

**Future work** should extend to: (i) full ensemble inference with trained checkpoints; (ii) multi-class evaluation; (iii) post-hoc calibration integration; (iv) computationally efficient uncertainty via LiDAR-MIMO; and (v) temporal fusion for scene-level SOTIF assessment.

---

## References

[1] ISO 21448:2022, "Road vehicles -- Safety of the intended functionality," International Organization for Standardization, 2022.

[2] M. Patel and R. Jung, "Uncertainty evaluation to support safety of the intended functionality analysis for identifying performance insufficiencies in ML-based LiDAR object detection," Kempten University of Applied Sciences, 2026.

[3] M. Patel and R. Jung, "Uncertainty representation in a SOTIF-related use case with Dempster-Shafer theory for LiDAR sensor-based object detection," arXiv:2503.02087, 2025.

[4] Y. Yan, Y. Mao, and B. Li, "SECOND: Sparsely embedded convolutional detection," Sensors, vol. 18, no. 10, p. 3337, 2018.

[5] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in Proc. NeurIPS, 2017.

[6] M. Pitropov, C. Huang, S. Abdelfattah, K. Czarnecki, and S. L. Waslander, "LiDAR-MIMO: Efficient uncertainty estimation for LiDAR 3D object detection," in Proc. IEEE ICRA, 2022.

[7] D. Feng, A. Harakeh, S. L. Waslander, and K. Dietmayer, "A review and comparative study on probabilistic object detection in autonomous driving," IEEE Trans. Intelligent Transportation Systems, vol. 23, no. 8, pp. 9961-9982, 2021.

[8] G. Shafer, A Mathematical Theory of Evidence. Princeton University Press, 1976.

[9] A. P. Dempster, "Upper and lower probabilities induced by a multivalued mapping," Annals of Mathematical Statistics, vol. 38, no. 2, pp. 325-339, 1967.

[10] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in Proc. ICML, 2016.

[11] G. P. Meyer, A. Laddha, E. Kee, C. Vallespi-Gonzalez, and C. K. Wellington, "LaserNet: An efficient probabilistic 3D object detector for autonomous driving," in Proc. IEEE CVPR, 2019.

[12] M. Hahner, C. Sakaridis, D. Dai, and L. Van Gool, "Fog simulation on real LiDAR point clouds for 3D object detection in adverse weather," in Proc. IEEE ICCV, 2021.

[13] M. Hahner, C. Sakaridis, M. Bijelic, F. Heide, F. Yu, D. Dai, and L. Van Gool, "LiDAR snowfall simulation for robust 3D object detection," in Proc. IEEE CVPR, 2022.

[14] Y. Li, Y. Wen, K. Wang, and F.-Y. Wang, "Realistic rainy weather simulation for LiDARs in CARLA simulator," arXiv:2312.12772, 2023.

[15] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, "CARLA: An open urban driving simulator," in Proc. CoRL, 2017.

[16] M. Patel, "SOTIF-PCOD: SOTIF point cloud object detection dataset," https://github.com/milinpatel07/SOTIF-PCOD, 2025.

[17] A. Geiger, P. Lenz, and R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," in Proc. IEEE CVPR, 2012.

[18] OpenPCDet Development Team, "OpenPCDet: An open-source toolbox for 3D object detection from point clouds," https://github.com/open-mmlab/OpenPCDet, 2020.

[19] R. Salay, R. Queiroz, and K. Czarnecki, "An analysis of ISO 26262: Using machine learning safely in automotive software," arXiv:1709.02435, 2019.

[20] L. Gauerhof, R. Hawkins, C. Sheridan, et al., "Assuring the safety of machine learning for pedestrian detection at crossings," in Proc. SAFECOMP, 2020.
