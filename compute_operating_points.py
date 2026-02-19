#!/usr/bin/env python3
"""Compute all operating points to find the best gates for the paper."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sotif_uncertainty.carla_case_study import generate_carla_case_study
from sotif_uncertainty.uncertainty import compute_all_indicators
from sotif_uncertainty.sotif_analysis import acceptance_gate, compute_coverage_far

data = generate_carla_case_study(seed=42)
ind = compute_all_indicators(data["scores"], data["boxes"])
mc = ind["mean_confidence"]
cv = ind["confidence_variance"]
gd = ind["geometric_disagreement"]
labels = data["labels"]

print("=" * 90)
print("OPERATING POINT SWEEP")
print("=" * 90)

# Broad sweep
results = []
for ts in np.arange(0.20, 0.75, 0.05):
    for td in [0.20, 0.25, 0.30, 0.35, 0.40, 0.49, 0.60, np.inf]:
        for tv in [0.002, 0.005, 0.010, 0.020, 0.030, np.inf]:
            acc = acceptance_gate(mc, cv, gd, tau_s=ts, tau_v=tv, tau_d=td)
            cov, far, ret, fp_ct = compute_coverage_far(acc, labels)
            parts = [f"s>={ts:.2f}"]
            if not np.isinf(td): parts.append(f"d<={td:.2f}")
            if not np.isinf(tv): parts.append(f"var<={tv:.3f}")
            gate_str = " & ".join(parts)
            results.append({"gate": gate_str, "cov": cov, "far": far, "ret": ret, "fp": fp_ct})

# Also single-indicator gates
for td in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    acc = acceptance_gate(mc, cv, gd, tau_s=0.0, tau_d=td)
    cov, far, ret, fp_ct = compute_coverage_far(acc, labels)
    results.append({"gate": f"d<={td:.2f}", "cov": cov, "far": far, "ret": ret, "fp": fp_ct})

# Sort by coverage desc
results.sort(key=lambda x: -x["cov"])

# Show FAR=0 gates
print("\n--- All gates with FAR = 0.000 (sorted by coverage descending) ---")
print(f"  {'Gate':<45} {'Coverage':>10} {'Retained':>10} {'FP':>5}")
zero_far = [r for r in results if r["far"] == 0.0 and r["ret"] > 0]
for r in zero_far:
    print(f"  {r['gate']:<45} {r['cov']:>9.1%} {r['ret']:>10} {r['fp']:>5}")

# Show best gates at various FAR levels
print("\n--- Best coverage at each FAR level ---")
for alpha in [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]:
    valid = [r for r in results if r["far"] <= alpha and r["ret"] > 0]
    if valid:
        best = max(valid, key=lambda x: x["cov"])
        print(f"  FAR <= {alpha:.2f}: {best['gate']:<40} cov={best['cov']:.1%}  far={best['far']:.3f}  ret={best['ret']}  fp={best['fp']}")

# Show the specific gates the paper currently uses
print("\n--- Paper's current gate configurations (recomputed) ---")
paper_gates = [
    (0.50, np.inf, np.inf, "s>=0.50"),
    (0.50, np.inf, 0.49,  "s>=0.50 & d<=0.49"),
    (0.35, np.inf, 0.49,  "s>=0.35 & d<=0.49"),
    (0.35, np.inf, np.inf, "s>=0.35"),
    (0.00, np.inf, 0.30,  "d<=0.30"),
    (0.35, np.inf, 0.20,  "s>=0.35 & d<=0.20"),
    (0.30, np.inf, 0.20,  "s>=0.30 & d<=0.20"),
    (0.25, np.inf, 0.20,  "s>=0.25 & d<=0.20"),
    (0.20, np.inf, 0.20,  "s>=0.20 & d<=0.20"),
    (0.00, np.inf, 0.20,  "d<=0.20"),
    (0.00, np.inf, 0.15,  "d<=0.15"),
]
print(f"  {'Gate':<45} {'Coverage':>10} {'FAR':>10} {'Retained':>10} {'FP':>5}")
for ts, tv, td, label in paper_gates:
    acc = acceptance_gate(mc, cv, gd, tau_s=ts, tau_d=td, tau_v=tv)
    cov, far, ret, fp_ct = compute_coverage_far(acc, labels)
    print(f"  {label:<45} {cov:>9.1%} {far:>10.3f} {ret:>10} {fp_ct:>5}")

# Interesting selective gates
print("\n--- Promising paper-worthy gates ---")
interesting = [
    (0.35, np.inf, 0.20, "s>=0.35 & d<=0.20"),
    (0.30, np.inf, 0.20, "s>=0.30 & d<=0.20"),
    (0.25, np.inf, 0.20, "s>=0.25 & d<=0.20"),
    (0.35, 0.020, 0.30, "s>=0.35 & var<=0.020 & d<=0.30"),
    (0.30, 0.020, 0.25, "s>=0.30 & var<=0.020 & d<=0.25"),
    (0.30, 0.020, 0.20, "s>=0.30 & var<=0.020 & d<=0.20"),
    (0.25, 0.020, 0.20, "s>=0.25 & var<=0.020 & d<=0.20"),
    (0.20, 0.020, 0.20, "s>=0.20 & var<=0.020 & d<=0.20"),
    (0.20, 0.030, 0.20, "s>=0.20 & var<=0.030 & d<=0.20"),
    (0.50, 0.010, np.inf, "s>=0.50 & var<=0.010"),
    (0.40, 0.010, 0.30, "s>=0.40 & var<=0.010 & d<=0.30"),
    (0.35, 0.010, 0.25, "s>=0.35 & var<=0.010 & d<=0.25"),
]
print(f"  {'Gate':<50} {'Coverage':>10} {'FAR':>10} {'Retained':>10} {'FP':>5}")
for ts, tv, td, label in interesting:
    acc = acceptance_gate(mc, cv, gd, tau_s=ts, tau_d=td, tau_v=tv)
    cov, far, ret, fp_ct = compute_coverage_far(acc, labels)
    print(f"  {label:<50} {cov:>9.1%} {far:>10.3f} {ret:>10} {fp_ct:>5}")
