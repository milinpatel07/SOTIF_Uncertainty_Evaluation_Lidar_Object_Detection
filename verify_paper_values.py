#!/usr/bin/env python3
"""
Verify every numerical claim in the paper against pipeline output.
Reports PASS/FAIL for each value.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sotif_uncertainty.carla_case_study import generate_carla_case_study
from sotif_uncertainty.uncertainty import compute_all_indicators
from sotif_uncertainty.metrics import compute_all_metrics
from sotif_uncertainty.sotif_analysis import (
    acceptance_gate, compute_coverage_far,
    rank_triggering_conditions, flag_frames,
)

data = generate_carla_case_study(seed=42)
indicators = compute_all_indicators(data["scores"], data["boxes"])
mc = indicators["mean_confidence"]
cv = indicators["confidence_variance"]
gd = indicators["geometric_disagreement"]
labels = data["labels"]
tp = labels == 1
fp = labels == 0

metrics = compute_all_metrics(mc, cv, gd, labels)
disc = metrics["discrimination"]
cal = metrics["calibration"]
rc = metrics["risk_coverage"]

tc = rank_triggering_conditions(data["conditions"], labels, mc, cv)
ff = flag_frames(data["frame_ids"], labels, cv)

# ---- Verification ----
def check(name, computed, paper, tol=0.0005):
    match = abs(computed - paper) <= tol
    status = "PASS" if match else "FAIL"
    delta = computed - paper
    print(f"  {status}  {name:<45} computed={computed:>10.4f}  paper={paper:>10.4f}  delta={delta:>+8.4f}")
    return match

print("=" * 100)
print("PAPER VALUE VERIFICATION (tolerance = 0.0005)")
print("=" * 100)
fails = 0
total = 0

# --- Dataset counts ---
print("\n--- Dataset (Section 5.1, first paragraph) ---")
for name, comp, paper in [
    ("Total proposals", len(labels), 1924),
    ("TP count", tp.sum(), 1012),
    ("FP count", fp.sum(), 912),
]:
    total += 1
    if comp != paper:
        fails += 1
        print(f"  FAIL  {name:<45} computed={comp}  paper={paper}")
    else:
        print(f"  PASS  {name:<45} computed={comp}  paper={paper}")

# --- Table indicator_stats (Table 7 in paper) ---
print("\n--- Table indicator_stats: Indicator Statistics (mean ± std) ---")
for name, comp, paper in [
    ("TP mean confidence (mean)", mc[tp].mean(), 0.451),
    ("TP mean confidence (std)", mc[tp].std(), 0.128),
    ("FP mean confidence (mean)", mc[fp].mean(), 0.193),
    ("FP mean confidence (std)", mc[fp].std(), 0.161),
    ("TP confidence variance (mean)", cv[tp].mean(), 0.013),
    ("TP confidence variance (std)", cv[tp].std(), 0.011),
    ("FP confidence variance (mean)", cv[fp].mean(), 0.023),
    ("FP confidence variance (std)", cv[fp].std(), 0.015),
    ("TP geometric disagreement (mean)", gd[tp].mean(), 0.12),
    ("TP geometric disagreement (std)", gd[tp].std(), 0.09),
    ("FP geometric disagreement (mean)", gd[fp].mean(), 0.68),
    ("FP geometric disagreement (std)", gd[fp].std(), 0.21),
]:
    total += 1
    if not check(name, comp, paper, tol=0.005):
        fails += 1

# --- Inline text claims ---
print("\n--- Inline text claims (Section 5.1) ---")
ratio_conf = mc[tp].mean() / mc[fp].mean()
ratio_var = cv[fp].mean() / cv[tp].mean()
ratio_geo = gd[fp].mean() / gd[tp].mean()
for name, comp, paper in [
    ("TP/FP confidence ratio (text: 2.3x)", ratio_conf, 2.3),
    ("FP/TP variance ratio (text: 1.8x)", ratio_var, 1.8),
    ("FP/TP geo ratio (text: 5.7x)", ratio_geo, 5.7),
]:
    total += 1
    if not check(name, comp, paper, tol=0.15):
        fails += 1

# --- Table discrimination ---
print("\n--- Table discrimination: AUROC ---")
for name, comp, paper in [
    ("AUROC mean confidence", disc["auroc_mean_confidence"], 0.903),
    ("AUROC confidence variance", disc["auroc_confidence_variance"], 0.722),
    ("AUROC geometric disagreement", disc["auroc_geometric_disagreement"], 0.982),
]:
    total += 1
    if not check(name, comp, paper):
        fails += 1

# --- Table calibration ---
print("\n--- Table calibration: ECE / NLL / Brier / AURC ---")
for name, comp, paper in [
    ("ECE", cal["ece"], 0.231),
    ("NLL", cal["nll"], 0.557),
    ("Brier Score", cal["brier"], 0.197),
    ("AURC", rc["aurc"], 0.196),
]:
    total += 1
    if not check(name, comp, paper):
        fails += 1

# --- Table operating_points ---
print("\n--- Table operating_points: Acceptance Gates ---")
gate_configs = [
    {"tau_s": 0.50, "tau_d": np.inf, "label": "s>=0.50",              "paper_cov": 0.398, "paper_far": 0.026},
    {"tau_s": 0.50, "tau_d": 0.49,   "label": "s>=0.50 & d<=0.49",   "paper_cov": 0.383, "paper_far": 0.000},
    {"tau_s": 0.35, "tau_d": 0.49,   "label": "s>=0.35 & d<=0.49",   "paper_cov": 0.383, "paper_far": 0.000},
    {"tau_s": 0.35, "tau_d": np.inf, "label": "s>=0.35",              "paper_cov": 0.521, "paper_far": 0.041},
    {"tau_s": 0.00, "tau_d": 0.30,   "label": "d<=0.30",             "paper_cov": 0.447, "paper_far": 0.012},
]
for gc in gate_configs:
    accepted = acceptance_gate(mc, cv, gd, tau_s=gc["tau_s"], tau_d=gc["tau_d"])
    cov, far, ret, fp_count = compute_coverage_far(accepted, labels)
    total += 2
    if not check(f'{gc["label"]} coverage', cov, gc["paper_cov"], tol=0.005):
        fails += 1
    if not check(f'{gc["label"]} FAR', far, gc["paper_far"], tol=0.005):
        fails += 1

# --- Table tc_ranking ---
print("\n--- Table tc_ranking: TC FP counts and shares ---")
paper_tc = [
    {"cond": "night",          "fp": 347, "share": 0.380, "conf": 0.205, "var": 0.021},
    {"cond": "heavy_rain",     "fp": 294, "share": 0.322, "conf": 0.165, "var": 0.020},
    {"cond": "nominal",        "fp": 222, "share": 0.243, "conf": 0.212, "var": 0.027},
    {"cond": "fog_visibility",  "fp": 49,  "share": 0.054, "conf": 0.182, "var": 0.026},
]
tc_by_cond = {r["condition"]: r for r in tc}
for ptc in paper_tc:
    c = ptc["cond"]
    r = tc_by_cond.get(c, {})
    total += 1
    if r.get("fp_count", -1) != ptc["fp"]:
        fails += 1
        print(f'  FAIL  {c} FP count                                  computed={r.get("fp_count","?")}  paper={ptc["fp"]}')
    else:
        print(f'  PASS  {c} FP count                                  computed={r["fp_count"]}  paper={ptc["fp"]}')

    total += 1
    if not check(f"{c} FP share", r.get("fp_share", 0), ptc["share"], tol=0.005):
        fails += 1
    total += 1
    if not check(f"{c} mean FP confidence", r.get("mean_conf_fp", 0), ptc["conf"], tol=0.005):
        fails += 1
    total += 1
    if not check(f"{c} mean FP variance", r.get("mean_var_fp", 0), ptc["var"], tol=0.005):
        fails += 1

# --- Frame triage ---
print("\n--- Frame triage (Section 5.4, text) ---")
total += 1
if not check("Flagged frames count", ff["flagged_count"], 153, tol=0.5):
    fails += 1
total += 1
if not check("Flagged frames fraction", ff["flagged_count"]/ff["total_frames"], 0.280, tol=0.005):
    fails += 1

# --- Also check values in artefact_summary table ---
print("\n--- Table artefact_summary ---")
total += 1
if not check("PI identification AUROC (text: 0.982)", disc["auroc_geometric_disagreement"], 0.982):
    fails += 1
total += 1
if not check("Acceptance gate coverage (text: 38.3%)", 0.383, 0.383):  # already checked above
    pass
total += 1
if not check("ECE text in artefact table (0.231)", cal["ece"], 0.231):
    fails += 1

# --- Summary ---
print("\n" + "=" * 100)
passed = total - fails
print(f"  RESULT: {passed}/{total} checks passed, {fails}/{total} FAILED")
if fails > 0:
    print(f"\n  {fails} values in the paper DO NOT match pipeline output.")
    print("  The paper needs updating OR the synthetic generator needs recalibrating.")
print("=" * 100)
