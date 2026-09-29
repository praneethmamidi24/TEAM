import json
import numpy as np

with open('multimodel_vistaara_results.json', 'r') as f:
    data = json.load(f)

print("=== 1. AUDITING TASK FITNESS FOR SCENE 75 ===")
for cand in ["Candidate A", "Candidate B"]:
    entry = data["scene_evaluations"]["scene_75"][cand]
    recon = entry["reconstruction_mean"]
    spec = entry["spectral_mean"]
    spat = entry["spatial_mean"]
    stab = entry["stability_mean"]
    r_mean = entry["mean_reliability"]

    # Formulas:
    # Biophysical: 0.50 * recon + 0.50 * spec
    calc_biophys = 0.50 * recon + 0.50 * spec
    # Structural: 0.50 * spat + 0.30 * stab + 0.20 * recon
    calc_struct = 0.50 * spat + 0.30 * stab + 0.20 * recon
    # General: geometric mean (0.35, 0.35, 0.15, 0.15)
    calc_general = (recon**0.35) * (spec**0.35) * (spat**0.15) * (stab**0.15)

    stored_fit = data["task_fitness"]["scene_75"][cand]

    print(f"\n--- {cand} (Scene 75) ---")
    print(f"Raw Components: Recon={recon:.6f}, Spec={spec:.6f}, Spat={spat:.6f}, Stab={stab:.6f}")
    print(f"Biophysical: 0.50*{recon:.4f} + 0.50*{spec:.4f} = {calc_biophys:.6f} | Stored: {stored_fit['Task 1: Biophysical / Radiometric Fidelity']:.6f}")
    print(f"Structural:  0.50*{spat:.4f} + 0.30*{stab:.4f} + 0.20*{recon:.4f} = {calc_struct:.6f} | Stored: {stored_fit['Task 2: Structural / Infrastructure Delineation']:.6f}")
    print(f"General:     ({recon:.4f}^0.35)*({spec:.4f}^0.35)*({spat:.4f}^0.15)*({stab:.4f}^0.15) = {calc_general:.6f} | Stored: {stored_fit['Task 3: Balanced General-Purpose SR']:.6f}")

print("\n=== 1b. AUDITING TASK FITNESS FOR SCENE 138 ===")
for cand in ["Candidate A", "Candidate B"]:
    entry = data["scene_evaluations"]["scene_138"][cand]
    recon = entry["reconstruction_mean"]
    spec = entry["spectral_mean"]
    spat = entry["spatial_mean"]
    stab = entry["stability_mean"]

    calc_biophys = 0.50 * recon + 0.50 * spec
    calc_struct = 0.50 * spat + 0.30 * stab + 0.20 * recon
    calc_general = (recon**0.35) * (spec**0.35) * (spat**0.15) * (stab**0.15)

    stored_fit = data["task_fitness"]["scene_138"][cand]
    print(f"\n--- {cand} (Scene 138) ---")
    print(f"Biophysical: {calc_biophys:.6f} | Stored: {stored_fit['Task 1: Biophysical / Radiometric Fidelity']:.6f}")
    print(f"Structural:  {calc_struct:.6f} | Stored: {stored_fit['Task 2: Structural / Infrastructure Delineation']:.6f}")
    print(f"General:     {calc_general:.6f} | Stored: {stored_fit['Task 3: Balanced General-Purpose SR']:.6f}")

print("\n=== 2. AUDITING DOWNSTREAM URBAN RESULTS (SCENE 75) ===")
ds_75_a = data["downstream_evaluations"]["scene_75"]["Candidate A"]
ds_75_b = data["downstream_evaluations"]["scene_75"]["Candidate B"]

print("Candidate A Downstream Details:")
print(json.dumps(ds_75_a, indent=2))

fp_raw_a = ds_75_a["unfiltered"]["false_positives"]
fp_gated_a = ds_75_a["vistaara_gated"]["false_positives"]
fp_red_a = (fp_raw_a - fp_gated_a) / fp_raw_a * 100.0
print(f"Cand A FP Reduction: ({fp_raw_a} - {fp_gated_a}) / {fp_raw_a} = {fp_red_a:.4f}%")

print("\nCandidate B Downstream Details:")
print(json.dumps(ds_75_b, indent=2))

print("\n=== 3. AUDITING CONTROLLED FAILURES (SCENE 138) ===")
for cand in ["Candidate A", "Candidate B"]:
    print(f"\n--- {cand} Failures ---")
    for f in data["controlled_failures"]["scene_138"][cand]:
        print(f"  {f['failure_type']}: Delta R = {f['delta_r_patch']:+.4f}, Primary = {f['primary_detector']} ({f['primary_component_drop']:+.4f}), False Alarm Delta = {f['false_alarm_delta']:.8f}")

print("\n=== 4. AUDITING TIMINGS & PROFILES ===")
for s in ["scene_138", "scene_75"]:
    for cand in ["Candidate A", "Candidate B"]:
        timings = data["scene_evaluations"][s][cand]["timings"]
        print(f"{s} | {cand}: Total Reliability Time = {timings['total_reliability_s']:.4f}s (Recon={timings['reconstruction_s']:.4f}s, Spec={timings['spectral_s']:.4f}s, Spat={timings['spatial_s']:.4f}s, Stab={timings['stability_s']:.4f}s)")
