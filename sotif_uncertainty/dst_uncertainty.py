"""
Dempster-Shafer Theory (DST) Uncertainty Representation for SOTIF.

Implements evidence-based uncertainty decomposition for LiDAR object
detection using Dempster-Shafer Theory / belief functions. Decomposes
total uncertainty into three orthogonal components:

    - Aleatoric uncertainty: inherent sensor noise, irreducible
    - Epistemic uncertainty: model ignorance, reducible with more data
    - Ontological uncertainty: unknown unknowns, out-of-distribution

This provides a richer uncertainty representation than pure ensemble
variance, enabling more nuanced SOTIF triggering condition analysis
per ISO 21448 and ISO/PAS 8800.

The theory maps ensemble detection outputs to mass functions on the
frame of discernment {Correct, Incorrect, Unknown}, then combines
evidence from K ensemble members using Dempster's rule of combination.

Reference:
    Patel, M. and Jung, R. (2025). "Uncertainty Representation in a
    SOTIF-Related Use Case with Dempster-Shafer Theory for LiDAR
    Sensor-Based Object Detection." arxiv:2503.02087.

    Shafer, G. (1976). "A Mathematical Theory of Evidence."

    Dempster, A. (1967). "Upper and lower probabilities induced
    by a multivalued mapping."
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


# =========================================================================
# Mass function representation
# =========================================================================

class MassFunction:
    """
    Dempster-Shafer mass function (basic probability assignment).

    Represents evidence over the frame of discernment
    Theta = {correct, incorrect}.

    Mass assignments:
        m({correct})   = belief that detection is correct (TP)
        m({incorrect}) = belief that detection is incorrect (FP)
        m(Theta)       = uncertainty mass (ignorance)
        m({})          = conflict (after combination)

    Invariant: m(correct) + m(incorrect) + m(Theta) = 1.0
    """

    def __init__(
        self,
        m_correct: float = 0.0,
        m_incorrect: float = 0.0,
        m_uncertain: float = 1.0,
    ):
        total = m_correct + m_incorrect + m_uncertain
        if total < 1e-10:
            self.m_correct = 0.0
            self.m_incorrect = 0.0
            self.m_uncertain = 1.0
        else:
            self.m_correct = m_correct / total
            self.m_incorrect = m_incorrect / total
            self.m_uncertain = m_uncertain / total

    @property
    def belief_correct(self) -> float:
        """Belief in correctness (lower bound of probability)."""
        return self.m_correct

    @property
    def belief_incorrect(self) -> float:
        """Belief in incorrectness (lower bound)."""
        return self.m_incorrect

    @property
    def plausibility_correct(self) -> float:
        """Plausibility of correctness (upper bound of probability)."""
        return self.m_correct + self.m_uncertain

    @property
    def plausibility_incorrect(self) -> float:
        """Plausibility of incorrectness (upper bound)."""
        return self.m_incorrect + self.m_uncertain

    @property
    def uncertainty_interval(self) -> Tuple[float, float]:
        """
        Uncertainty interval for correctness: [Bel, Pl].

        Width = Pl - Bel = m(Theta) = ignorance.
        """
        return (self.belief_correct, self.plausibility_correct)

    @property
    def pignistic_probability(self) -> float:
        """
        Pignistic probability of correctness (BetP).

        BetP distributes ignorance equally between hypotheses.
        BetP(correct) = Bel(correct) + m(Theta) / 2
        """
        return self.m_correct + self.m_uncertain / 2.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "m_correct": self.m_correct,
            "m_incorrect": self.m_incorrect,
            "m_uncertain": self.m_uncertain,
            "belief_correct": self.belief_correct,
            "belief_incorrect": self.belief_incorrect,
            "plausibility_correct": self.plausibility_correct,
            "plausibility_incorrect": self.plausibility_incorrect,
            "pignistic_probability": self.pignistic_probability,
        }


# =========================================================================
# Dempster's Rule of Combination
# =========================================================================

def dempster_combine(m1: MassFunction, m2: MassFunction) -> MassFunction:
    """
    Combine two mass functions using Dempster's rule.

    Dempster's rule computes the orthogonal sum of two bodies of
    evidence, normalizing by the conflict factor.

    Parameters
    ----------
    m1, m2 : MassFunction
        Two independent sources of evidence.

    Returns
    -------
    MassFunction
        Combined mass function.
    """
    # Compute all intersections
    # {C} cap {C} = {C}, {C} cap Theta = {C}, Theta cap {C} = {C}
    # {I} cap {I} = {I}, {I} cap Theta = {I}, Theta cap {I} = {I}
    # Theta cap Theta = Theta
    # {C} cap {I} = emptyset (conflict)
    # {I} cap {C} = emptyset (conflict)

    m_cc = m1.m_correct * m2.m_correct
    m_ct = m1.m_correct * m2.m_uncertain
    m_tc = m1.m_uncertain * m2.m_correct

    m_ii = m1.m_incorrect * m2.m_incorrect
    m_it = m1.m_incorrect * m2.m_uncertain
    m_ti = m1.m_uncertain * m2.m_incorrect

    m_tt = m1.m_uncertain * m2.m_uncertain

    # Conflict
    conflict = (
        m1.m_correct * m2.m_incorrect + m1.m_incorrect * m2.m_correct
    )

    # Normalization factor
    norm = 1.0 - conflict
    if norm < 1e-10:
        warnings.warn("Total conflict in Dempster combination. Returning vacuous mass.")
        return MassFunction(0.0, 0.0, 1.0)

    combined_correct = (m_cc + m_ct + m_tc) / norm
    combined_incorrect = (m_ii + m_it + m_ti) / norm
    combined_uncertain = m_tt / norm

    return MassFunction(combined_correct, combined_incorrect, combined_uncertain)


def dempster_combine_multiple(masses: List[MassFunction]) -> MassFunction:
    """
    Combine multiple mass functions using iterated Dempster's rule.

    Parameters
    ----------
    masses : list of MassFunction
        K independent sources of evidence.

    Returns
    -------
    MassFunction
        Combined mass function from all K sources.
    """
    if len(masses) == 0:
        return MassFunction(0.0, 0.0, 1.0)
    if len(masses) == 1:
        return masses[0]

    result = masses[0]
    for i in range(1, len(masses)):
        result = dempster_combine(result, masses[i])

    return result


# =========================================================================
# Ensemble to mass function conversion
# =========================================================================

def score_to_mass(
    score: float,
    detected: bool = True,
    method: str = "confidence",
) -> MassFunction:
    """
    Convert a single ensemble member's detection output to a mass function.

    Three conversion methods are available:
    - 'confidence': Direct confidence-to-mass mapping
    - 'evidential': Evidence-based mapping with uncertainty preservation
    - 'cautious': Conservative mapping that assigns more to uncertainty

    Parameters
    ----------
    score : float
        Detection confidence score from one ensemble member.
    detected : bool
        Whether this member detected the object.
    method : str
        Conversion method: 'confidence', 'evidential', or 'cautious'.

    Returns
    -------
    MassFunction
        Mass function for this member's evidence.
    """
    if not detected or score <= 0:
        # Non-detection: evidence against correctness
        return MassFunction(0.0, 0.3, 0.7)

    score = np.clip(score, 0.0, 1.0)

    if method == "confidence":
        # Direct mapping: high score -> belief in correctness
        m_correct = score * 0.8
        m_incorrect = (1.0 - score) * 0.5
        m_uncertain = 1.0 - m_correct - m_incorrect

    elif method == "evidential":
        # Evidence strength scales with score, preserving more uncertainty
        evidence_strength = score ** 1.5
        m_correct = evidence_strength * 0.7
        m_incorrect = (1.0 - score) ** 2 * 0.4
        m_uncertain = 1.0 - m_correct - m_incorrect

    elif method == "cautious":
        # Conservative: assigns significant mass to uncertainty
        m_correct = score * 0.5
        m_incorrect = (1.0 - score) * 0.3
        m_uncertain = 1.0 - m_correct - m_incorrect

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure non-negative
    m_uncertain = max(0.0, m_uncertain)

    return MassFunction(m_correct, m_incorrect, m_uncertain)


def ensemble_to_dst(
    scores: np.ndarray,
    method: str = "confidence",
) -> MassFunction:
    """
    Convert ensemble member scores for a single proposal to a combined
    DST mass function.

    Parameters
    ----------
    scores : np.ndarray, shape (K,)
        Confidence scores from K ensemble members.
        Score = 0 means the member did not detect the object.
    method : str
        Mass conversion method (see score_to_mass).

    Returns
    -------
    MassFunction
        Combined mass function from all K members.
    """
    masses = []
    for score in scores:
        detected = score > 0
        mass = score_to_mass(float(score), detected, method)
        masses.append(mass)

    return dempster_combine_multiple(masses)


# =========================================================================
# Batch processing for the evaluation pipeline
# =========================================================================

def compute_dst_indicators(
    scores: np.ndarray,
    method: str = "confidence",
) -> Dict[str, np.ndarray]:
    """
    Compute DST-based uncertainty indicators for all proposals.

    For each of N proposals with K ensemble member scores, computes:
    - Belief in correctness (lower bound)
    - Plausibility of correctness (upper bound)
    - Uncertainty mass (ignorance = Pl - Bel)
    - Pignistic probability (point estimate)
    - Conflict level (disagreement between members)

    Parameters
    ----------
    scores : np.ndarray, shape (N, K)
        Ensemble member confidence scores.
    method : str
        Mass conversion method.

    Returns
    -------
    dict with keys:
        'belief' : (N,) Belief in correctness
        'plausibility' : (N,) Plausibility of correctness
        'uncertainty_mass' : (N,) Ignorance (Pl - Bel)
        'pignistic_prob' : (N,) Pignistic probability
        'dissonance' : (N,) Measure of inter-member conflict
    """
    N = scores.shape[0]

    belief = np.zeros(N)
    plausibility = np.zeros(N)
    uncertainty_mass = np.zeros(N)
    pignistic = np.zeros(N)
    dissonance = np.zeros(N)

    for i in range(N):
        combined = ensemble_to_dst(scores[i], method)
        belief[i] = combined.belief_correct
        plausibility[i] = combined.plausibility_correct
        uncertainty_mass[i] = combined.m_uncertain
        pignistic[i] = combined.pignistic_probability

        # Compute dissonance (conflict between members)
        member_masses = []
        for score in scores[i]:
            detected = score > 0
            member_masses.append(score_to_mass(float(score), detected, method))

        if len(member_masses) >= 2:
            total_conflict = 0.0
            n_pairs = 0
            for a in range(len(member_masses)):
                for b in range(a + 1, len(member_masses)):
                    conflict = (
                        member_masses[a].m_correct * member_masses[b].m_incorrect
                        + member_masses[a].m_incorrect * member_masses[b].m_correct
                    )
                    total_conflict += conflict
                    n_pairs += 1
            dissonance[i] = total_conflict / n_pairs if n_pairs > 0 else 0.0

    return {
        "belief": belief,
        "plausibility": plausibility,
        "uncertainty_mass": uncertainty_mass,
        "pignistic_prob": pignistic,
        "dissonance": dissonance,
    }


# =========================================================================
# Uncertainty decomposition (aleatoric / epistemic / ontological)
# =========================================================================

def decompose_uncertainty_dst(
    scores: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    method: str = "confidence",
) -> Dict[str, np.ndarray]:
    """
    Decompose detection uncertainty into aleatoric, epistemic, and
    ontological components using DST.

    - Aleatoric: Irreducible sensor/environment noise.
      Measured by average within-member uncertainty.

    - Epistemic: Model ignorance, reducible with more training data.
      Measured by between-member disagreement (dissonance).

    - Ontological: Unknown unknowns, out-of-distribution inputs.
      Measured by combined uncertainty mass after fusion.
      High ontological uncertainty indicates the input lies outside
      the known operating domain.

    Parameters
    ----------
    scores : np.ndarray, shape (N, K)
        Ensemble member confidence scores.
    boxes : np.ndarray, shape (N, K, 7), optional
        Bounding boxes for spatial uncertainty.
    method : str
        Mass conversion method.

    Returns
    -------
    dict with keys:
        'aleatoric' : (N,) Aleatoric uncertainty
        'epistemic' : (N,) Epistemic uncertainty
        'ontological' : (N,) Ontological uncertainty
        'total' : (N,) Total uncertainty
        'dst_indicators' : dict from compute_dst_indicators()
    """
    N, K = scores.shape

    dst = compute_dst_indicators(scores, method)

    # Aleatoric: average individual member's uncertainty
    # (how uncertain is each member about its own detection?)
    aleatoric = np.zeros(N)
    for i in range(N):
        member_uncertainties = []
        for k in range(K):
            s = scores[i, k]
            if s > 0:
                # Entropy-like measure of individual member confidence
                s_clipped = np.clip(s, 1e-7, 1.0 - 1e-7)
                h = -(s_clipped * np.log2(s_clipped) + (1 - s_clipped) * np.log2(1 - s_clipped))
                member_uncertainties.append(h)
        aleatoric[i] = np.mean(member_uncertainties) if member_uncertainties else 1.0

    # Epistemic: dissonance between members (disagreement)
    epistemic = dst["dissonance"]

    # Ontological: residual uncertainty mass after combining all evidence
    # High values indicate the combined evidence is still highly uncertain,
    # suggesting the input is outside the model's competence
    ontological = dst["uncertainty_mass"]

    # Spatial epistemic component (if boxes available)
    if boxes is not None:
        from sotif_uncertainty.uncertainty import compute_geometric_disagreement
        geo_disagree = compute_geometric_disagreement(boxes)
        # Blend spatial disagreement into epistemic
        epistemic = 0.6 * epistemic + 0.4 * geo_disagree

    total = aleatoric + epistemic + ontological

    return {
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "ontological": ontological,
        "total": total,
        "dst_indicators": dst,
    }


# =========================================================================
# SOTIF integration: DST-based acceptance gate
# =========================================================================

def dst_acceptance_gate(
    dst_indicators: Dict[str, np.ndarray],
    tau_belief: float = 0.5,
    tau_uncertainty: float = 0.3,
    tau_dissonance: float = 0.2,
) -> np.ndarray:
    """
    DST-based acceptance gate for SOTIF evaluation.

    A detection is accepted if:
    1. Belief in correctness >= tau_belief
    2. Uncertainty mass <= tau_uncertainty
    3. Dissonance <= tau_dissonance

    This provides a more principled acceptance criterion than
    simple confidence thresholding, as it explicitly accounts
    for ignorance and inter-member conflict.

    Parameters
    ----------
    dst_indicators : dict
        Output from compute_dst_indicators().
    tau_belief : float
        Minimum belief threshold.
    tau_uncertainty : float
        Maximum uncertainty mass threshold.
    tau_dissonance : float
        Maximum dissonance threshold.

    Returns
    -------
    np.ndarray, dtype bool
        True if detection is accepted.
    """
    accepted = (
        (dst_indicators["belief"] >= tau_belief)
        & (dst_indicators["uncertainty_mass"] <= tau_uncertainty)
        & (dst_indicators["dissonance"] <= tau_dissonance)
    )
    return accepted


def compute_dst_operating_points(
    dst_indicators: Dict[str, np.ndarray],
    labels: np.ndarray,
    tau_belief_range: Optional[np.ndarray] = None,
    tau_uncertainty_range: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Compute operating point table using DST indicators.

    Parameters
    ----------
    dst_indicators : dict
        Output from compute_dst_indicators().
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).
    tau_belief_range : np.ndarray, optional
        Belief thresholds to sweep.
    tau_uncertainty_range : np.ndarray, optional
        Uncertainty mass thresholds to sweep.

    Returns
    -------
    list of dict
        Operating points with coverage, FAR, etc.
    """
    if tau_belief_range is None:
        tau_belief_range = np.arange(0.3, 0.91, 0.05)
    if tau_uncertainty_range is None:
        tau_uncertainty_range = np.array([0.1, 0.2, 0.3, 0.5, np.inf])

    n = len(labels)
    results = []

    for tb in tau_belief_range:
        for tu in tau_uncertainty_range:
            accepted = dst_acceptance_gate(
                dst_indicators,
                tau_belief=tb,
                tau_uncertainty=tu,
                tau_dissonance=np.inf,
            )

            n_accepted = int(np.sum(accepted))
            if n_accepted == 0:
                coverage = 0.0
                far = 0.0
                fp = 0
            else:
                coverage = n_accepted / n
                fp = int(np.sum(accepted & (labels == 0)))
                far = fp / n_accepted

            parts = [f"bel>={tb:.2f}"]
            if not np.isinf(tu):
                parts.append(f"unc<={tu:.2f}")
            desc = " & ".join(parts)

            results.append({
                "gate": desc,
                "tau_belief": float(tb),
                "tau_uncertainty": float(tu),
                "coverage": coverage,
                "retained": n_accepted,
                "fp": fp,
                "far": far,
            })

    return results
