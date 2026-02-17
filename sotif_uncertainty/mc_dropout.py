"""
Monte Carlo Dropout Uncertainty Estimation.

Alternative to deep ensembles for uncertainty quantification.
Instead of training K independent models, MC Dropout performs
K stochastic forward passes through a single model with dropout
enabled at inference time.

Advantages:
    - Single model (no ensemble training needed)
    - Lower memory footprint
    - Easier to deploy

Disadvantages:
    - Typically lower uncertainty quality than deep ensembles
    - Requires dropout layers in the architecture
    - Stochastic passes are sequential (no parallelism benefit)

The uncertainty indicators computed are identical to ensemble-based
ones (mean confidence, confidence variance, geometric disagreement),
enabling direct comparison using the same evaluation pipeline.

Reference:
    Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning." ICML.

    Feng et al. (2018). "Towards Safe Autonomous Driving: Capture
    Uncertainty in the Deep Neural Network For Lidar 3D Vehicle
    Detection." ICRA.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def mc_dropout_inference(
    model,
    data_loader,
    n_passes: int = 10,
    score_thresh: float = 0.1,
    device: str = "cuda",
) -> List[List[Dict]]:
    """
    Run MC Dropout inference: K stochastic forward passes with dropout enabled.

    This function wraps a PyTorch model and performs multiple forward passes
    with dropout active, collecting predictions for uncertainty estimation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained detector model with dropout layers.
    data_loader : DataLoader
        Test data loader.
    n_passes : int
        Number of stochastic forward passes (analogous to K in ensembles).
    score_thresh : float
        Minimum score to keep a detection.
    device : str
        Device for inference ('cuda' or 'cpu').

    Returns
    -------
    list of list of dict
        Outer list: n_passes "virtual members".
        Inner list: per-frame predictions in the same format as
        ensemble member outputs (compatible with cluster_detections).
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for MC Dropout inference. "
            "Install: pip install torch"
        )

    def enable_dropout(model):
        """Enable dropout layers during eval mode."""
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    all_passes = []
    model.eval()

    for k in range(n_passes):
        # Re-enable dropout for each pass
        enable_dropout(model)

        pass_preds = []
        with torch.no_grad():
            for batch_dict in data_loader:
                # Move to device
                for key in batch_dict:
                    if isinstance(batch_dict[key], torch.Tensor):
                        batch_dict[key] = batch_dict[key].to(device)

                pred_dicts, _ = model(batch_dict)

                for i, pred in enumerate(pred_dicts):
                    mask = pred["pred_scores"] >= score_thresh
                    pass_preds.append({
                        "boxes_lidar": pred["pred_boxes"][mask].cpu().numpy(),
                        "score": pred["pred_scores"][mask].cpu().numpy(),
                        "pred_labels": pred["pred_labels"][mask].cpu().numpy(),
                        "frame_id": batch_dict["frame_id"][i],
                    })

        all_passes.append(pass_preds)

    return all_passes


def simulate_mc_dropout(
    base_scores: np.ndarray,
    base_boxes: np.ndarray,
    n_passes: int = 10,
    dropout_rate: float = 0.3,
    score_noise_std: float = 0.05,
    position_noise_std: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate MC Dropout by adding stochastic noise to base predictions.

    This allows testing the uncertainty pipeline without a trained model.
    Simulates the effect of dropout by:
    1. Randomly dropping detections (probability = dropout_rate)
    2. Adding noise to confidence scores
    3. Adding noise to box positions

    Parameters
    ----------
    base_scores : np.ndarray, shape (N,)
        Base confidence scores from a single model.
    base_boxes : np.ndarray, shape (N, 7)
        Base bounding boxes [x, y, z, dx, dy, dz, heading].
    n_passes : int
        Number of stochastic passes to simulate.
    dropout_rate : float
        Probability of dropping a detection in each pass.
    score_noise_std : float
        Standard deviation of score perturbation.
    position_noise_std : float
        Standard deviation of position perturbation (meters).
    seed : int
        Random seed.

    Returns
    -------
    mc_scores : np.ndarray, shape (N, n_passes)
        Score matrix (0 if detection dropped in that pass).
    mc_boxes : np.ndarray, shape (N, n_passes, 7)
        Box matrix (NaN if detection dropped).
    """
    rng = np.random.RandomState(seed)
    N = len(base_scores)

    mc_scores = np.zeros((N, n_passes))
    mc_boxes = np.full((N, n_passes, 7), np.nan)

    for k in range(n_passes):
        # Randomly drop detections
        keep_mask = rng.random(N) > dropout_rate

        for i in range(N):
            if keep_mask[i]:
                # Add score noise
                noisy_score = base_scores[i] + rng.normal(0, score_noise_std)
                mc_scores[i, k] = np.clip(noisy_score, 0.01, 0.99)

                # Add position noise
                mc_boxes[i, k] = base_boxes[i].copy()
                mc_boxes[i, k, :3] += rng.normal(0, position_noise_std, 3)
                mc_boxes[i, k, 3:6] += rng.normal(0, 0.05, 3)
                mc_boxes[i, k, 6] += rng.normal(0, 0.02)

    return mc_scores, mc_boxes


def compare_ensemble_vs_mcdropout(
    ensemble_scores: np.ndarray,
    mc_scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compare uncertainty quality between deep ensemble and MC Dropout.

    Computes AUROC for both methods using mean confidence and
    confidence variance as indicators.

    Parameters
    ----------
    ensemble_scores : np.ndarray, shape (N, K)
        Scores from K ensemble members.
    mc_scores : np.ndarray, shape (N, T)
        Scores from T MC Dropout passes.
    labels : np.ndarray, shape (N,)
        Ground truth labels (1=TP, 0=FP).

    Returns
    -------
    dict with 'ensemble' and 'mc_dropout' sub-dicts, each containing
    'auroc_mean_conf' and 'auroc_conf_var'.
    """
    from sotif_uncertainty.metrics import compute_auroc

    # Ensemble indicators
    ens_mean = np.mean(ensemble_scores, axis=1)
    ens_var = np.var(ensemble_scores, axis=1, ddof=1)

    # MC Dropout indicators
    mc_mean = np.mean(mc_scores, axis=1)
    # Handle all-zero rows (detection never seen)
    valid = np.sum(mc_scores > 0, axis=1) > 1
    mc_var = np.zeros(len(mc_scores))
    mc_var[valid] = np.var(mc_scores[valid], axis=1, ddof=1)

    return {
        "ensemble": {
            "auroc_mean_conf": compute_auroc(ens_mean, labels, higher_is_correct=True),
            "auroc_conf_var": compute_auroc(ens_var, labels, higher_is_correct=False),
        },
        "mc_dropout": {
            "auroc_mean_conf": compute_auroc(mc_mean, labels, higher_is_correct=True),
            "auroc_conf_var": compute_auroc(mc_var, labels, higher_is_correct=False),
        },
    }
