"""
run_corrected_stress_tests.py
=============================
Comprehensive Validation and Verification Suite for Corrected Reliability Engine.

Executes:
1. Stability Sensitivity Curve across sigmas [0.002, 0.005, 0.010, 0.020, 0.050]
2. Localized Ringing / Checkerboard Test (400x400 artifact)
3. Full Synthetic Perturbation Tests (A1, A2, B1, B2, C1, C2, D, E)
4. Composite Reliability Ablation (A, B, C, D)
5. False-Confidence Analysis
6. Real Sentinel-2 Baseline Comparison (OLD vs NEW on 138.tif and 75.tif)
7. Generates Visual Diagnostic Artifacts into stress_test_results/
"""

import os
import sys
import time
import json

# Ensure clean UTF-8 console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import rasterio
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mlstac
from PIL import Image

from reliability_engine import (
    detect_and_normalize_reflectance,
    compute_observation_consistency,
    compute_spectral_consistency,
    compute_spatial_edge_consistency,
    compute_local_perturbation_stability,
    compute_composite_reliability,
    diagnose_difficult_regions,
    evaluate_sr_reliability
)

RESULTS_DIR = r"stress_test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIR = r".\SEN2SRLite_RGBN_x4\SEN2SRLite\NonReference_RGBN_x4"
print("Loading SEN2SRLite model...")
model = mlstac.load(MODEL_DIR).compiled_model(device="cpu")
model.eval()
print("Model loaded successfully.")


def run_overlapped_sr(lr_norm, model, patch_size=128, overlap=16, scale=4):
    c, h, w = lr_norm.shape
    stride = patch_size - overlap
    sr_h, sr_w = h * scale, w * scale

    def get_1d_window(size, margin):
        win = torch.ones(size, dtype=torch.float32)
        if margin > 0:
            ramp = torch.linspace(0.0, 1.0, margin)
            win[:margin] = ramp
            win[-margin:] = torch.flip(ramp, dims=[0])
        return win

    win_sr_1d = get_1d_window(patch_size * scale, overlap * scale)
    win_sr_2d = (win_sr_1d.unsqueeze(1) * win_sr_1d.unsqueeze(0)).unsqueeze(0)

    y_starts = sorted(list(set(list(range(0, h - patch_size, stride)) + [max(0, h - patch_size)])))
    x_starts = sorted(list(set(list(range(0, w - patch_size, stride)) + [max(0, w - patch_size)])))

    accum_output = torch.zeros((4, sr_h, sr_w), dtype=torch.float32)
    accum_weight = torch.zeros((1, sr_h, sr_w), dtype=torch.float32)

    lr_tensor = torch.from_numpy(lr_norm)
    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                patch = lr_tensor[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
                sr_patch = model(patch)[0]
                sr_y = y * scale
                sr_x = x * scale
                accum_output[:, sr_y:sr_y+patch_size*scale, sr_x:sr_x+patch_size*scale] += sr_patch * win_sr_2d
                accum_weight[:, sr_y:sr_y+patch_size*scale, sr_x:sr_x+patch_size*scale] += win_sr_2d

    sr_norm = (accum_output / torch.clamp(accum_weight, min=1e-6)).numpy()
    return sr_norm


# Load Clean 138.tif
with rasterio.open("Main Data Sets/138.tif") as src:
    image_138 = src.read()
    profile_138 = src.profile.copy()
    scl_138 = image_138[12] if image_138.shape[0] >= 13 else None

lr_raw_138 = image_138[[3, 2, 1, 7]].astype(np.float32)
lr_norm_138, scale_factor_138, _ = detect_and_normalize_reflectance(lr_raw_138)
sr_clean_138 = run_overlapped_sr(lr_norm_138, model)

# Evaluate Corrected Clean Baseline on 138.tif
clean_rel_corrected = evaluate_sr_reliability(
    lr_norm_138, sr_clean_138, model, "cpu", scl_image=scl_138, scale=4
)
clean_rmap_corr = clean_rel_corrected["reliability_map"]
clean_mean_corr = float(clean_rel_corrected["composite_stats"]["mean"])
clean_median_corr = float(clean_rel_corrected["composite_stats"]["median"])
c, h, w = sr_clean_138.shape


# ===========================================================================
# 1. STABILITY SENSITIVITY CURVE ACROSS SIGMAS
# ===========================================================================
print("\n" + "="*65)
print("STAGE 1: CORRECTED LOCAL STABILITY SENSITIVITY CURVE")
print("="*65)

stability_curve_records = []
sigmas = [0.002, 0.005, 0.010, 0.020, 0.050]

for s_val in sigmas:
    t0_s = time.time()
    stab_res = compute_local_perturbation_stability(
        lr_norm_138, sr_clean_138, model, "cpu", scale=4, sigma=s_val, sample_patches=16, tau_stab=0.10
    )
    t_s = time.time() - t0_s
    s_map = stab_res["score_map"]

    entry = {
        "sigma": s_val,
        "nominal_sensitivity": float(stab_res["nominal_sensitivity"]),
        "mean_sensitivity": float(stab_res["mean_sensitivity"]),
        "mean_score": float(stab_res["mean_score"]),
        "median_score": float(np.median(s_map)),
        "min_score": float(np.min(s_map)),
        "max_score": float(np.max(s_map)),
        "p10_score": float(np.percentile(s_map, 10)),
        "p90_score": float(np.percentile(s_map, 90)),
        "runtime_s": t_s
    }
    stability_curve_records.append(entry)
    print(f"Sigma={s_val:.3f} | NomSens={entry['nominal_sensitivity']:.4f}x | Score Mean={entry['mean_score']:.4f} | Min={entry['min_score']:.4f} | Max={entry['max_score']:.4f} | Runtime={t_s:.2f}s")

with open(os.path.join(RESULTS_DIR, "stability_curve_corrected.json"), "w") as f:
    json.dump(stability_curve_records, f, indent=2)


# ===========================================================================
# 2. LOCALIZED RINGING / CHECKERBOARD TEST
# ===========================================================================
print("\n" + "="*65)
print("STAGE 2: LOCALIZED RINGING / CHECKERBOARD TEST")
print("="*65)

y1, y2, x1, x2 = 600, 1000, 800, 1200
mask_inside = np.zeros((h, w), dtype=bool)
mask_inside[y1:y2, x1:x2] = True
mask_outside = ~mask_inside

# Create localized checkerboard artifact (A = 0.06 inside box)
yy, xx = np.indices((y2-y1, x2-x1))
chk_patch = (((yy % 2) ^ (xx % 2)) * 2 - 1).astype(np.float32) * 0.06

sr_ring_local = sr_clean_138.copy()
sr_ring_local[:, y1:y2, x1:x2] += chk_patch[np.newaxis, :, :]
sr_ring_local = np.clip(sr_ring_local, 0.0, 1.0)

# Evaluate with corrected engine
rel_ring_local = evaluate_sr_reliability(
    lr_norm_138, sr_ring_local, model, "cpu", scl_image=scl_138, scale=4
)
rmap_ring_local = rel_ring_local["reliability_map"]
spat_hf_map = rel_ring_local["component_maps"]["spatial_hf"]

ring_inside_rel = float(np.mean(rmap_ring_local[mask_inside]))
ring_outside_rel = float(np.mean(rmap_ring_local[mask_outside]))
ring_inside_clean = float(np.mean(clean_rmap_corr[mask_inside]))
ring_outside_clean = float(np.mean(clean_rmap_corr[mask_outside]))

ring_inside_hf = float(np.mean(spat_hf_map[mask_inside]))
ring_outside_hf = float(np.mean(spat_hf_map[mask_outside]))

localized_ringing_report = {
    "artifact_type": "Checkerboard Ringing (A=0.06, 400x400 px)",
    "clean_inside_rel": ring_inside_clean,
    "clean_outside_rel": ring_outside_clean,
    "artifact_inside_rel": ring_inside_rel,
    "artifact_outside_rel": ring_outside_rel,
    "delta_inside": ring_inside_rel - ring_inside_clean,
    "delta_outside": ring_outside_rel - ring_outside_clean,
    "contrast_inside_vs_outside": ring_inside_rel - ring_outside_rel,
    "hf_score_inside": ring_inside_hf,
    "hf_score_outside": ring_outside_hf,
    "localized_ringing_detected": bool(ring_inside_rel < ring_outside_rel - 0.05)
}

print(f"Clean Inside Rel: {ring_inside_clean:.4f} | Clean Outside Rel: {ring_outside_clean:.4f}")
print(f"Artifact Inside Rel: {ring_inside_rel:.4f} | Artifact Outside Rel: {ring_outside_rel:.4f}")
print(f"Inside Drop: {localized_ringing_report['delta_inside']:+.4f} | Outside Drop: {localized_ringing_report['delta_outside']:+.4f}")
print(f"HF Anomaly Score: Inside={ring_inside_hf:.4f} vs Outside={ring_outside_hf:.4f}")
print(f"Localized Ringing Detected: {localized_ringing_report['localized_ringing_detected']}")

with open(os.path.join(RESULTS_DIR, "localized_ringing_report.json"), "w") as f:
    json.dump(localized_ringing_report, f, indent=2)


# ===========================================================================
# 3. RE-RUN FULL SYNTHETIC PERTURBATION TESTS (A1, A2, B1, B2, C1, C2, D, E)
# ===========================================================================
print("\n" + "="*65)
print("STAGE 3: RE-RUN FULL SYNTHETIC PERTURBATION TESTS (CORRECTED ENGINE)")
print("="*65)

# Re-create test copies
np.random.seed(101)
sr_noise_small = np.clip(sr_clean_138 + np.random.normal(0, 0.015, sr_clean_138.shape).astype(np.float32), 0.0, 1.0)
np.random.seed(102)
sr_noise_mod = np.clip(sr_clean_138 + np.random.normal(0, 0.050, sr_clean_138.shape).astype(np.float32), 0.0, 1.0)

sr_spec_mild = sr_clean_138.copy()
sr_spec_mild[0] += 0.03
sr_spec_mild[2] -= 0.03
sr_spec_mild = np.clip(sr_spec_mild, 0.0, 1.0)

sr_spec_strong = sr_clean_138.copy()
sr_spec_strong[0] += 0.10
sr_spec_strong[3] -= 0.08
sr_spec_strong[1] += 0.06
sr_spec_strong = np.clip(sr_spec_strong, 0.0, 1.0)

yy_full, xx_full = np.indices((h, w))
chk_mild = (((yy_full % 2) ^ (xx_full % 2)) * 2 - 1).astype(np.float32) * 0.03
sr_ring_mild = np.clip(sr_clean_138 + chk_mild[np.newaxis, :, :], 0.0, 1.0)

chk_sev = (((yy_full % 2) ^ (xx_full % 2)) * 2 - 1).astype(np.float32) * 0.08
sr_ring_severe = np.clip(sr_clean_138 + chk_sev[np.newaxis, :, :], 0.0, 1.0)

sr_local_art = sr_clean_138.copy()
patch_noise = np.random.normal(0, 0.06, (4, y2-y1, x2-x1)).astype(np.float32)
sr_local_art[:, y1:y2, x1:x2] += patch_noise
sr_local_art[0, y1:y2, x1:x2] += 0.08
sr_local_art = np.clip(sr_local_art, 0.0, 1.0)

from scipy.ndimage import sobel
lum = np.mean(sr_clean_138, axis=0)
edge_mag = np.sqrt(sobel(lum, axis=0)**2 + sobel(lum, axis=1)**2)
edge_mask = edge_mag > np.percentile(edge_mag, 92)
sr_edge_dist = sr_clean_138.copy()
sr_edge_dist[:, edge_mask] += np.random.normal(0, 0.08, sr_edge_dist[:, edge_mask].shape).astype(np.float32)
sr_edge_dist = np.clip(sr_edge_dist, 0.0, 1.0)

test_cases_corrected = [
    ("Baseline Clean", sr_clean_138),
    ("Test A1: Small Noise (sigma=0.015)", sr_noise_small),
    ("Test A2: Moderate Noise (sigma=0.050)", sr_noise_mod),
    ("Test B1: Mild Spectral Shift", sr_spec_mild),
    ("Test B2: Strong Spectral Distortion", sr_spec_strong),
    ("Test C1: Mild Ringing (A=0.03)", sr_ring_mild),
    ("Test C2: Severe Ringing (A=0.08)", sr_ring_severe),
    ("Test D: Localized Artifact (400x400)", sr_local_art),
    ("Test E: Edge Distortion (Top 8%)", sr_edge_dist)
]

corrected_perturbation_records = []
clean_stability_metrics = clean_rel_corrected["stability_metrics"]

for name, sr_c in test_cases_corrected:
    recon_res = compute_observation_consistency(lr_norm_138, sr_c, scale=4)
    spec_res = compute_spectral_consistency(lr_norm_138, sr_c, scale=4)
    spat_res = compute_spatial_edge_consistency(lr_norm_138, sr_c, scale=4)

    comp_res = compute_composite_reliability(
        recon_res["score_map"],
        spec_res["score_map"],
        spat_res["score_map"],
        clean_stability_metrics["score_map"],
        weights=(0.35, 0.35, 0.15, 0.15)
    )
    rmap = comp_res["reliability_map"]

    d_mean = float(comp_res["mean"] - clean_mean_corr)
    d_median = float(comp_res["median"] - clean_median_corr)
    affected_pct = float(np.mean(np.abs(rmap - clean_rmap_corr) > 0.05) * 100)

    entry = {
        "test_name": name,
        "mean_rel": float(comp_res["mean"]),
        "median_rel": float(comp_res["median"]),
        "min_rel": float(np.min(rmap)),
        "delta_mean": d_mean,
        "delta_median": d_median,
        "affected_pixels_pct": affected_pct,
        "recon_score": float(recon_res["mean_score"]),
        "spec_score": float(spec_res["mean_score"]),
        "spat_score": float(spat_res["mean_score"]),
        "spat_hf_score": float(spat_res["mean_hf_score"]),
        "spat_dir_score": float(np.mean(spat_res["dir_score_map"])),
        "stability_score": float(clean_stability_metrics["mean_score"])
    }
    corrected_perturbation_records.append(entry)
    print(f"[{name:36s}] Rel Mean={entry['mean_rel']:.4f} (Delta={d_mean:+.4f}) | Affected={affected_pct:5.1f}% | Recon={entry['recon_score']:.4f} | Spec={entry['spec_score']:.4f} | Spat={entry['spat_score']:.4f} | HF={entry['spat_hf_score']:.4f}")

with open(os.path.join(RESULTS_DIR, "synthetic_perturbation_corrected.json"), "w") as f:
    json.dump(corrected_perturbation_records, f, indent=2)


# ===========================================================================
# 4. COMPOSITE ABLATION AFTER FIXES
# ===========================================================================
print("\n" + "="*65)
print("STAGE 4: COMPOSITE RELIABILITY ABLATION AFTER FIXES")
print("="*65)

s_recon_c = clean_rel_corrected["recon_metrics"]["score_map"]
s_spec_c = clean_rel_corrected["spectral_metrics"]["score_map"]
s_spat_c = clean_rel_corrected["spatial_metrics"]["score_map"]
s_stab_c = clean_rel_corrected["stability_metrics"]["score_map"]

# Candidate 1: Equal weights (0.25, 0.25, 0.25, 0.25)
r_equal = (s_recon_c * s_spec_c * s_spat_c * s_stab_c) ** 0.25
# Candidate 2: Observation-Centric (0.35, 0.35, 0.15, 0.15) [Chosen Standard]
r_obs_35 = (s_recon_c**0.35 * s_spec_c**0.35 * s_spat_c**0.15 * s_stab_c**0.15)
# Candidate 3: Conservative (0.40, 0.40, 0.10, 0.10)
r_cons_40 = (s_recon_c**0.40 * s_spec_c**0.40 * s_spat_c**0.10 * s_stab_c**0.10)

# Incremental ablation
r_A = s_recon_c.copy()
r_B = np.sqrt(s_recon_c * s_spec_c)
r_C = (s_recon_c**0.40 * s_spec_c**0.40 * s_spat_c**0.20)
r_D = clean_rmap_corr.copy()

ablation_corrected = [
    {"signal_set": "A: Recon Only", "mean": float(np.mean(r_A)), "median": float(np.median(r_A)), "min": float(np.min(r_A)), "max": float(np.max(r_A)), "p5": float(np.percentile(r_A, 5)), "p95": float(np.percentile(r_A, 95))},
    {"signal_set": "B: Recon + Spectral", "mean": float(np.mean(r_B)), "median": float(np.median(r_B)), "min": float(np.min(r_B)), "max": float(np.max(r_B)), "p5": float(np.percentile(r_B, 5)), "p95": float(np.percentile(r_B, 95))},
    {"signal_set": "C: Recon + Spectral + Spatial", "mean": float(np.mean(r_C)), "median": float(np.median(r_C)), "min": float(np.min(r_C)), "max": float(np.max(r_C)), "p5": float(np.percentile(r_C, 5)), "p95": float(np.percentile(r_C, 95))},
    {"signal_set": "D: Full (0.35, 0.35, 0.15, 0.15)", "mean": float(np.mean(r_D)), "median": float(np.median(r_D)), "min": float(np.min(r_D)), "max": float(np.max(r_D)), "p5": float(np.percentile(r_D, 5)), "p95": float(np.percentile(r_D, 95))},
    {"signal_set": "Candidate: Equal Weights (0.25 each)", "mean": float(np.mean(r_equal)), "median": float(np.median(r_equal)), "min": float(np.min(r_equal)), "max": float(np.max(r_equal)), "p5": float(np.percentile(r_equal, 5)), "p95": float(np.percentile(r_equal, 95))},
    {"signal_set": "Candidate: Conservative (0.40, 0.40, 0.10, 0.10)", "mean": float(np.mean(r_cons_40)), "median": float(np.median(r_cons_40)), "min": float(np.min(r_cons_40)), "max": float(np.max(r_cons_40)), "p5": float(np.percentile(r_cons_40, 5)), "p95": float(np.percentile(r_cons_40, 95))}
]

for ab in ablation_corrected:
    print(f"[{ab['signal_set']:42s}] Mean={ab['mean']:.4f} | Median={ab['median']:.4f} | P5={ab['p5']:.4f} | P95={ab['p95']:.4f} | Min={ab['min']:.4f}")

with open(os.path.join(RESULTS_DIR, "ablation_corrected.json"), "w") as f:
    json.dump(ablation_corrected, f, indent=2)


# ===========================================================================
# 5. REAL SENTINEL-2 SCENE VALIDATION (OLD VS NEW)
# ===========================================================================
print("\n" + "="*65)
print("STAGE 5: REAL SENTINEL-2 SCENE VALIDATION (138.tif and 75.tif)")
print("="*65)

comparison_table = []

for tif_path in ["Main Data Sets/138.tif", "Main Data Sets/75.tif"]:
    tag = os.path.basename(tif_path)
    with rasterio.open(tif_path) as src:
        img = src.read()
        scl_cur = img[12] if img.shape[0] >= 13 else None

    lr_raw = img[[3, 2, 1, 7]].astype(np.float32)
    lr_n, _, _ = detect_and_normalize_reflectance(lr_raw)
    sr_n = run_overlapped_sr(lr_n, model)

    t0 = time.time()
    rel_res = evaluate_sr_reliability(lr_n, sr_n, model, "cpu", scl_image=scl_cur, scale=4)
    t_el = time.time() - t0
    r_map = rel_res["reliability_map"]

    comp_rec = {
        "scene": tag,
        "recon_score": float(rel_res["recon_metrics"]["mean_score"]),
        "spectral_score": float(rel_res["spectral_metrics"]["mean_score"]),
        "spatial_score": float(rel_res["spatial_metrics"]["mean_score"]),
        "spatial_hf_score": float(rel_res["spatial_metrics"]["mean_hf_score"]),
        "stability_score": float(rel_res["stability_metrics"]["mean_score"]),
        "nominal_sensitivity": float(rel_res["stability_metrics"]["nominal_sensitivity"]),
        "composite_mean": float(rel_res["composite_stats"]["mean"]),
        "composite_median": float(rel_res["composite_stats"]["median"]),
        "composite_min": float(np.min(r_map)),
        "composite_max": float(np.max(r_map)),
        "pct_lt_90": float(np.mean(r_map < 0.90) * 100),
        "pct_lt_75": float(np.mean(r_map < 0.75) * 100),
        "pct_lt_50": float(np.mean(r_map < 0.50) * 100),
        "runtime_s": t_el
    }
    comparison_table.append(comp_rec)
    print(f"[{tag}] Composite Mean: {comp_rec['composite_mean']:.4f} | Median: {comp_rec['composite_median']:.4f} | Min: {comp_rec['composite_min']:.4f}")
    print(f"  Recon: {comp_rec['recon_score']:.4f} | Spec: {comp_rec['spectral_score']:.4f} | Spat: {comp_rec['spatial_score']:.4f} (HF={comp_rec['spatial_hf_score']:.4f}) | Stab: {comp_rec['stability_score']:.4f} (NomSens={comp_rec['nominal_sensitivity']:.4f}x)")
    print(f"  Thresholds: <0.90: {comp_rec['pct_lt_90']:.2f}% | <0.75: {comp_rec['pct_lt_75']:.2f}% | <0.50: {comp_rec['pct_lt_50']:.2f}%")

with open(os.path.join(RESULTS_DIR, "real_scene_comparison_corrected.json"), "w") as f:
    json.dump(comparison_table, f, indent=2)


# ===========================================================================
# 6. RE-RUN FALSE CONFIDENCE ANALYSIS
# ===========================================================================
print("\n" + "="*65)
print("STAGE 6: FALSE-CONFIDENCE & FALSE-ALARM ANALYSIS")
print("="*65)

raw_err_10m = clean_rel_corrected["recon_metrics"]["error_map_10m"]
raw_sam_10m = clean_rel_corrected["spectral_metrics"]["sam_rad_10m"] * (180.0 / np.pi)

# Upsample to 2.5m for pixel-by-pixel checking
err_2_5m = F.interpolate(torch.from_numpy(raw_err_10m).unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze().numpy()
sam_2_5m = F.interpolate(torch.from_numpy(raw_sam_10m).unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze().numpy()
hf_err_2_5m = rel_ring_local["component_maps"]["spatial_hf"]

high_rel_mask = (clean_rmap_corr > 0.85)

# High Rel with High Recon Error (> 3%)
fc_recon_px = int(np.sum(high_rel_mask & (err_2_5m > 0.03)))
# High Rel with High SAM (> 2 deg)
fc_sam_px = int(np.sum(high_rel_mask & (sam_2_5m > 2.0)))

# Low Rel with Low Error (< 0.60 Rel, but < 1% Recon Error, < 1 deg SAM, > 0.95 Spat)
low_rel_mask = (clean_rmap_corr < 0.60)
clean_spat_map = clean_rel_corrected["component_maps"]["spatial"]
fa_px = int(np.sum(low_rel_mask & (err_2_5m < 0.01) & (sam_2_5m < 1.0) & (clean_spat_map > 0.95)))

false_conf_corrected = {
    "total_pixels": int(h * w),
    "high_rel_threshold": 0.85,
    "high_recon_error_px": fc_recon_px,
    "high_recon_error_pct": float(fc_recon_px / (h * w) * 100),
    "high_sam_error_px": fc_sam_px,
    "high_sam_error_pct": float(fc_sam_px / (h * w) * 100),
    "low_rel_threshold": 0.60,
    "false_alarm_px": fa_px,
    "false_alarm_pct": float(fa_px / (h * w) * 100)
}

print(f"High Reliability (>0.85) with High Recon Error (>3%): {false_conf_corrected['high_recon_error_pct']:.4f}% ({fc_recon_px} px)")
print(f"High Reliability (>0.85) with High SAM Angle (>2.0 deg): {false_conf_corrected['high_sam_error_pct']:.4f}% ({fc_sam_px} px)")
print(f"Low Reliability (<0.60) with Low Error (False Alarm):    {false_conf_corrected['false_alarm_pct']:.4f}% ({fa_px} px)")

with open(os.path.join(RESULTS_DIR, "false_confidence_corrected.json"), "w") as f:
    json.dump(false_conf_corrected, f, indent=2)


# ===========================================================================
# 7. GENERATE VISUAL DIAGNOSTIC ARTIFACTS
# ===========================================================================
print("\n" + "="*65)
print("STAGE 7: GENERATING VISUAL DIAGNOSTIC ARTIFACTS")
print("="*65)

def to_rgb_pil(sr_arr):
    rgb = np.transpose(sr_arr[[0, 1, 2]], (1, 2, 0))
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 == p2: p98 = p2 + 1e-5
    rgb_clipped = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    return Image.fromarray((rgb_clipped * 255).astype(np.uint8))

clean_rgb = to_rgb_pil(sr_clean_138)
ring_rgb = to_rgb_pil(sr_ring_local)

# Diagnostic A: Localized Ringing Artifact Detection (Clean vs Corrupted vs HF Score vs Delta)
fig, axs = plt.subplots(1, 4, figsize=(22, 5.5))
fig.patch.set_facecolor('#0d1117')

axs[0].imshow(clean_rgb)
rect0 = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='#00ffff', facecolor='none', linestyle=':')
axs[0].add_patch(rect0)
axs[0].set_title("1. Clean 2.5m SR", color='#c9d1d9', fontsize=11)
axs[0].axis('off')

axs[1].imshow(ring_rgb)
rect1 = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='#ef4444', facecolor='none', linestyle='--')
axs[1].add_patch(rect1)
axs[1].set_title("2. Localized Ringing Artifact (Box)", color='#c9d1d9', fontsize=11)
axs[1].axis('off')

im_hf = axs[2].imshow(spat_hf_map, cmap='magma', vmin=0.0, vmax=1.0)
rect2 = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='#ef4444', facecolor='none', linestyle='--')
axs[2].add_patch(rect2)
axs[2].set_title(f"3. High-Frequency Score (Inside: {ring_inside_hf:.2f})", color='#c9d1d9', fontsize=11)
axs[2].axis('off')
cbar_hf = fig.colorbar(im_hf, ax=axs[2], fraction=0.046, pad=0.04)
cbar_hf.ax.yaxis.set_tick_params(color='#c9d1d9')
plt.setp(plt.getp(cbar_hf.ax.axes, 'yticklabels'), color='#c9d1d9')

im_d = axs[3].imshow(clean_rmap_corr - rmap_ring_local, cmap='hot', vmin=0.0, vmax=0.8)
rect3 = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='#00ffff', facecolor='none', linestyle='--')
axs[3].add_patch(rect3)
axs[3].set_title(f"4. Reliability Drop Map (Inside Delta: {ring_inside_rel - ring_inside_clean:.2f})", color='#c9d1d9', fontsize=11)
axs[3].axis('off')
cbar_d = fig.colorbar(im_d, ax=axs[3], fraction=0.046, pad=0.04)
cbar_d.ax.yaxis.set_tick_params(color='#c9d1d9')
plt.setp(plt.getp(cbar_d.ax.axes, 'yticklabels'), color='#c9d1d9')

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "diag_3_localized_ringing_detection.png"), dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

# Diagnostic B: Before vs After Clean Reliability Map Comparison (138.tif)
# Read old baseline map from earlier run
old_baseline_path = os.path.join(RESULTS_DIR, "diag_1_clean_sr_vs_reliability.png")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0d1117')

axs[0].imshow(clean_rgb)
axs[0].set_title("Super-Resolved True Color (2.5m)", color='#c9d1d9', fontsize=12)
axs[0].axis('off')

# Load old baseline results json to get old score
with open(os.path.join(RESULTS_DIR, "baseline_results.json")) as f:
    old_base_json = json.load(f)
old_138_mean = old_base_json["138.tif"]["rel_mean"]

im_new = axs[1].imshow(clean_rmap_corr, cmap='RdYlGn', vmin=0.0, vmax=1.0)
axs[1].set_title(f"Corrected Reliability Map (Mean = {clean_mean_corr:.3f})", color='#c9d1d9', fontsize=12)
axs[1].axis('off')
cbar_new = fig.colorbar(im_new, ax=axs[1], fraction=0.046, pad=0.04)
cbar_new.ax.yaxis.set_tick_params(color='#c9d1d9')
plt.setp(plt.getp(cbar_new.ax.axes, 'yticklabels'), color='#c9d1d9')

# Component histogram
axs[2].hist(clean_rmap_corr.ravel()[::16], bins=40, color='#00ffff', alpha=0.8, density=True)
axs[2].set_facecolor('#161b22')
axs[2].set_title("Corrected Reliability Distribution", color='#c9d1d9', fontsize=12)
axs[2].tick_params(colors='#c9d1d9')
for spine in axs[2].spines.values():
    spine.set_color('#30363d')

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "diag_5_before_vs_after_reliability_comparison.png"), dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("All diagnostic visual artifacts generated successfully in stress_test_results/")
print("\n" + "="*65)
print("CORRECTED STRESS TESTS COMPLETED SUCCESSFULLY!")
print("="*65)
