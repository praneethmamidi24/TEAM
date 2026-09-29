# VISTAARA Prototype Technical Validation Report
## Model-Agnostic Post-Inference Evaluation & Task Decision Layer
**Problem Statement SIH26142:** *Deep Learning Based Super-Resolution Mapping from Medium Resolution Satellite Imageries*

---

### Executive Summary
This document provides a comprehensive technical audit and validation record for the redesigned **VISTAARA Prototype** (`app.py`). The prototype has been completely refactored around one central operational purpose:

> **"Given multiple super-resolution outputs generated from the same Sentinel-2 input, VISTAARA evaluates which output is more suitable for a specific geospatial task."**

VISTAARA is **not** a super-resolution model, nor does it compete with deep learning architectures. It is a **model-agnostic, post-inference evaluation and quality-control decision layer** designed to solve the critical operational bottleneck in Earth Observation: **the absence of ground-truth sub-meter reference imagery at satellite pass time**.

---

### 1. System Role & Architectural Boundaries
* **What VISTAARA Is:** A physics-grounded, multi-objective validation and decision engine that inspects super-resolved multi-spectral rasters without requiring ground truth. It evaluates physical sensor conservation, spectral angle/ratio consistency, high-frequency spatial artifacting, and perturbation stability.
* **What VISTAARA Is Not:** It is not an SR model, not an upscale algorithm, not a GAN, and not a bloated collection of disconnected domain modules (flood segmentation, damage assessment, chatbot).
* **Target Pipeline Position:**
  $$\text{Sentinel-2 10m Input} \longrightarrow \text{Candidate SR Models} \longrightarrow \mathbf{VISTAARA\ Evaluation} \longrightarrow \text{Task Fitness} \longrightarrow \mathbf{Reliability\ Decision\ Gate}$$

---

### 2. Controlled Ablation Experimental Design
To rigorously evaluate whether VISTAARA provides discriminative decision-making, two candidates were evaluated under a strict controlled ablation setup:

| Feature | Candidate A (Physically Constrained) | Candidate B (Unconstrained) | Ablation Status |
| :--- | :--- | :--- | :--- |
| **Model Family** | SEN2SRLite SPAN | Pure CNNSR | Identical Backbone |
| **Parameters** | 580,740 (`model.safetensor`) | 580,740 (`model.safetensor`) | **Strictly Identical Weights** |
| **Backbone Architecture** | Split-Attention SPAB CNN (6 blocks) | Split-Attention SPAB CNN (6 blocks) | Strictly Identical |
| **Input Spectral Bands** | B04 (Red), B03 (Green), B02 (Blue), B08 (NIR) | B04, B03, B02, B08 | Strictly Identical |
| **Tiling & Blending** | 128px patch, 16px overlap, 2D trapezoidal | 128px patch, 16px overlap, 2D trapezoidal | Strictly Identical |
| **Scale Factor** | $4\times$ ($10\text{ m} \to 2.5\text{ m}$) | $4\times$ ($10\text{ m} \to 2.5\text{ m}$) | Strictly Identical |
| **Experimental Variable** | **Fourier-Domain HardConstraint** | **None** (Clamped $\ge 0.0$) | **Single Isolated Variable** |

* **Candidate A (Physically Constrained):** Replaces synthesized low-frequency Fourier components with exact low-resolution sensor observations ($y_{low} = x$), guaranteeing conservation of radiant energy.
* **Candidate B (Unconstrained):** Executes pure end-to-end data-driven synthesis without frequency overrides, maximizing local edge gradients.

---

### 3. 5-Stage Progressive Disclosure Interface
The redesigned Streamlit prototype (`app.py`) presents an intuitive 5-stage workflow structured for both rapid 30-second judge comprehension and deep technical inspection:

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│   1. Mission    │ ──> │ 2. Input/Candidates │ ──> │ 3. Visual Comparison │ ──> │ 4. VISTAARA Engine  │ ──> │ 5. Task Decision  │
│ Operating Role  │     │  Scene Metadata     │     │  10m vs Cand A vs B  │     │  4-Pillar Scorecard │     │  Task Fitness &   │
│ Aphorism Banner │     │  Ablation Design    │     │  Raw Edge Indicators │     │  Reliability Maps   │     │  QC Decision Gate │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘     └─────────────────────┘     └───────────────────┘
```

1. **Stage 1 (Mission):** Establishes the core premise (*"A sharper-looking output is not automatically the most suitable output"*), explains why ground truth is absent, and outlines the workflow.
2. **Stage 2 (Input & Candidates):** Ingests the 13-band Sentinel-2 raster, verifies BOA surface reflectance normalization ($DN / 10000$), details the candidate architectures, and highlights the controlled ablation setup.
3. **Stage 3 (Candidate Comparison):** 3-column RGB visualization (10m Input vs Candidate A 2.5m vs Candidate B 2.5m), interactive split-screen slider, and raw edge indicators ($HF_{ratio}$, Sobel edge coherence).
4. **Stage 4 (VISTAARA Evaluation):** 4-pillar consistency scorecard table ($M_{recon}, M_{spec}, M_{spat}, M_{stab}$, Composite $R$), dense spatial reliability maps $R(x, y)$, coverage tier distribution, and expandable stress-test diagnostics.
5. **Stage 5 (Task Decision):** Interactive task selector (Biophysical, Structural, General), task fitness calculations, authoritative recommendation banner, downstream demonstrations (Reliability-Aware NDVI and 91.91% Urban False Positive Reduction), and GeoTIFF/CSV exports.

---

### 4. Full Numerical Audit on Benchmark Scenes
All values below were verified through live end-to-end pipeline execution in `test_redesigned_prototype.py`:

| Evaluation Dimension | Metric | Scene 138 (Agri & Mixed) Cand A | Scene 138 Cand B | Scene 75 (Coastal & Urban) Cand A | Scene 75 Cand B |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **VISTAARA Pillar 1** | Reconstruction ($M_{recon}$, $w=0.35$) | **0.9243** | 0.7390 | **0.9521** | 0.8055 |
| **VISTAARA Pillar 2** | Spectral ($M_{spec}$, $w=0.35$) | **0.8927** | 0.8228 | **0.8856** | 0.8103 |
| **VISTAARA Pillar 3** | Spatial ($M_{spat}$, $w=0.15$) | 0.9980 | 0.9980 | 0.9983 | **0.9983** |
| **VISTAARA Pillar 4** | Stability ($M_{stab}$, $w=0.15$) | **0.9698** | 0.9669 | 0.9740 | **0.9745** |
| **VISTAARA Composite** | Composite Reliability ($R$) | **0.9283** | 0.8324 | **0.9363** | 0.8554 |
| **Operational Coverage** | High Reliability ($\ge 0.93$) | **67.00%** | 2.98% | **86.09%** | 3.17% |
| **Operational Coverage** | Caution Region ($0.75 - 0.93$) | 30.64% | 85.50% | 13.13% | 95.12% |
| **Operational Coverage** | Low Consistency ($< 0.75$) | 2.36% | 11.52% | 0.78% | 1.71% |
| **Raw Edge Indicator** | $HF_{ratio}$ (Laplacian Energy) | 1.8627 | **1.9149** | 1.7799 | **1.7892** |
| **Raw Edge Indicator** | Sobel Edge Coherence | 0.7253 | **0.7341** | 0.6414 | **0.6495** |
| **Task Fitness A** | Biophysical ($0.50 M_{rec} + 0.50 M_{spec}$) | **0.9085** | 0.7809 | **0.9189** | 0.8079 |
| **Task Fitness B** | Structural ($0.50 M_{spa} + 0.30 M_{sta} + 0.20 M_{rec}$) | **0.9748** | 0.9369 | **0.9818** | 0.9526 |
| **Task Fitness C** | General ($R$) | **0.9283** | 0.8324 | **0.9363** | 0.8554 |

---

### 5. Mathematical Definitions of the 4 Consistency Pillars
1. **Reconstruction Consistency ($M_{recon}$):**
   $$M_{recon} = \exp\left(-\frac{\|\mathcal{D}_{PSF}(I_{SR}) - I_{LR}\|_1}{\sigma_{recon}}\right)$$
   Measures conservation of sensor radiant flux when the $2.5\text{ m}$ output is downsampled back to $10\text{ m}$ through the Sentinel-2 optical Point Spread Function.
2. **Spectral Consistency ($M_{spec}$):**
   $$M_{spec} = \exp\left(-\frac{\theta_{SAM}(I_{SR}, I_{LR\uparrow})}{\sigma_{SAM}}\right) \times \exp\left(-\frac{|NDVI_{SR} - NDVI_{LR\uparrow}|}{\sigma_{NDVI}}\right)$$
   Evaluates conservation of inter-band ratios and multi-spectral angle trajectory across B04, B03, B02, and B08.
3. **Spatial Consistency ($M_{spat}$):**
   $$M_{spat} = \left(1 - \text{CheckerboardEnergy}(I_{SR})\right) \times \text{EdgeCoherence}(I_{SR}, I_{LR\uparrow})$$
   Isolates high-frequency Fourier checkerboard artifacts and measures gradient directional alignment.
4. **Local Stability ($M_{stab}$):**
   $$M_{stab} = \exp\left(-\frac{\|I_{SR}(x + \delta) - I_{SR}(x)\|_1}{\sigma_{stab}}\right)$$
   Evaluates robustness under sub-pixel Gaussian radiometric input perturbations ($\delta \sim \mathcal{N}(0, 0.01^2)$).
5. **Composite Reliability ($R$):**
   $$R = M_{recon}^{0.35} \times M_{spec}^{0.35} \times M_{spat}^{0.15} \times M_{stab}^{0.15}$$
   A strict geometric mean ensuring that severe failure in any individual physical dimension penalizes composite reliability.

---

### 6. Root-Cause Explanation of Candidate Differences
The evaluation reveals the exact structural root-cause of performance divergence:
* **The Low-Frequency Drop:** Candidate B drops by **$-0.1466$** in Reconstruction Consistency ($0.9521 \to 0.8055$ on Scene 75). Without the Fourier HardConstraint, the unconstrained neural network drifts away from low-frequency sensor observations.
* **The Coverage Collapse:** Because $R$ requires simultaneous compliance across all four pillars, Candidate B's certified high-reliability coverage ($\ge 0.93$) collapses from **$86.09\%$ down to $3.17\%$**. An unconstrained model is therefore rejected wholesale by the VISTAARA quality-control gate.

---

### 7. Resolution of the Visual Sharpness Discrepancy
* **The Observation:** In Stage 3, Candidate B shows slightly higher raw edge energy ($HF_{ratio} = 1.7892$ vs $1.7799$) and higher Sobel edge coherence ($0.6495$ vs $0.6414$). Visually, building contours appear crisper.
* **The Scientific Resolution:** Despite higher raw edge indicators, **Candidate A achieves higher structural task fitness ($0.9818 > 0.9526$)**. When evaluating structural tasks under physical sensor conservation, the 20% weight assigned to reconstruction consistency ($0.20 \cdot M_{recon}$) penalizes Candidate B's low-frequency degradation. Sharpness achieved via physical drift does not constitute structural fidelity.

---

### 8. Operational Decision-Support Coverage Tiers
VISTAARA segments rasters into three actionable decision-support tiers ($\tau_{high} = 0.93$, $\tau_{mod} = 0.75$):
* **High Consistency Tier ($R \ge 0.93$):** Certified for autonomous, mission-critical GIS pipelines and automated feature extraction.
* **Caution Tier ($0.75 \le R < 0.93$):** Flagged for human-in-the-loop review or conservative tolerance settings.
* **Low Consistency Tier ($R < 0.75$):** Rejected / uncertified due to high risk of hallucination or radiometric drift.

On Scene 75, Candidate A qualifies **$86.09\%$** of the scene into the High Consistency Tier, whereas Candidate B qualifies only **$3.17\%$**, proving VISTAARA's power as an automated gatekeeper.

---

### 9. Downstream Task 1: Biophysical & Radiometric Analysis
* **Task Definition:** Precision agriculture, crop vigor assessment, and water body monitoring.
* **Fitness Formula:** $F_{bio} = 0.50 \cdot M_{recon} + 0.50 \cdot M_{spec}$
* **Results on Scene 75:**
  * **Candidate A:** $0.50(0.9521) + 0.50(0.8856) = \mathbf{0.9189}$
  * **Candidate B:** $0.50(0.8055) + 0.50(0.8103) = \mathbf{0.8079}$ ($\Delta = -0.1110$)
* **Prototype Demonstration:** The UI renders 2.5m NDVI rasters for Candidate A and Candidate B alongside the VISTAARA-certified NDVI mask ($R \ge 0.93$). Unconstrained radiometric drift in Candidate B leads to systematic skew across vegetated parcels, whereas Candidate A preserves reliable BOA surface reflectance.

---

### 10. Downstream Task 2: Structural & Urban Decision Gating
* **Task Definition:** Automated urban impervious surface extraction evaluated against the official Sentinel-2 Scene Classification Layer (SCL Class 5: Non-vegetated / Urban) reference mask.
* **Fitness Formula:** $F_{struct} = 0.50 \cdot M_{spat} + 0.30 \cdot M_{stab} + 0.20 \cdot M_{recon}$
* **Benchmark Results on Scene 75 (2,087,616 Reference Pixels):**

| Downstream Metric | Unfiltered Candidate A | Reliability-Gated ($R \ge 0.93$) | Operational Impact |
| :--- | :---: | :---: | :--- |
| **True Positive Pixels** | 985,732 | 904,581 | Valid urban detections preserved |
| **False Positive Pixels** | 62,844 | 5,081 | **57,763 spurious flags eliminated** |
| **False Positive Reduction** | Baseline | **91.915%** | **Over 91.9% noise suppression** |
| **Delineation Precision** | 94.01% | **99.44%** | **+5.43% precision gain** |
| **Operational Recall** | 47.22% | 43.33% | Controlled -3.89% trade-off |
| **F1-Score / IoU** | 0.6286 / 0.4584 | 0.6036 / 0.4323 | Robust operational quality |

* **Candidate B Gate Rejection:** Candidate B qualifies only 3.17% of pixels above threshold, correctly preventing unconstrained models from polluting automated cadastral or urban GIS databases.

---

### 11. Downstream Task 3: Balanced General Cartography
* **Task Definition:** Regional mapping, visual GIS basemaps, and general spatial visualization.
* **Fitness Formula:** $F_{gen} = R$ (Composite Reliability)
* **Results on Scene 75:** Candidate A scores **0.9363** vs Candidate B's **0.8554**. Candidate A delivers stable multi-purpose performance across 86.09% of the scene area.

---

### 12. Stress-Test Diagnostics & OpenSR LAM Comparison
* **Controlled Failure Sensitivity:** Independent stress testing verified that VISTAARA's 4 components respond specifically to targeted failure modes:
  * High-frequency ringing $\to$ Spatial Consistency ($M_{spat} \downarrow 0.1264$).
  * Spectral skew $\to$ Spectral Consistency ($M_{spec} \downarrow 0.2227$).
  * Radiometric shift $\to$ Reconstruction Consistency ($M_{recon} \downarrow 0.2373$).
* **Fair Comparison with ESA OpenSR Local Attention Maps (LAM):**
  * *OpenSR LAM:* Computes backward gradient attribution around a single point patch. Valuable for model development, but computationally intensive and unable to produce dense inference-time spatial decision rasters.
  * *VISTAARA:* Operates forward-only across the full raster in seconds, generating actionable spatial reliability maps and task fitness scores post-inference without requiring model gradients or retraining.

---

### 13. Adherence to 7 Core Scientific Boundaries
1. **Ill-Posed Inverse Problem:** Multiple high-resolution scenes can downsample to the identical Sentinel-2 observation. Super-resolution is synthetic estimation.
2. **Representation vs Native:** All outputs are labeled as "2.5m super-resolved representations", never "native 2.5m imagery."
3. **Consistency vs Ground Truth:** High VISTAARA reliability denotes high physical and spectral consistency with input data, not ground-truth certainty.
4. **Non-Frequentist Nature:** Reliability scores are bounded deterministic composites, not statistical probabilities.
5. **Dynamic Normalization:** Header metadata correctly governs conversion from integer DN to physical surface reflectance.
6. **Decision-Gating Trade-Off:** Quality-control filtering eliminates 91.915% of false positives with a quantified, realistic recall trade-off.
7. **Model-Agnostic Interface:** VISTAARA treats candidate models as black boxes, making it forward-compatible with CNNs, Transformers, and Diffusion models.

---

### 14. Automated Pipeline & Export Verification
The automated verification script `test_redesigned_prototype.py` executed cleanly with exit code 0:
* `app.py` compiles without syntax errors.
* Processing speed: ~30s per 512×512 scene on CPU (both candidates + reliability + downstream tasks).
* Session caching ensures instant (< 50ms) stage transitions in Streamlit.
* GeoTIFF exports preserve original CRS (`EPSG:32643`), 4 bands (B04, B03, B02, B08), $4\times$ dimensions ($2048 \times 2048$), and affine resolution ($2.5\text{ m}$).
* Comprehensive metrics exported to `prototype_metrics_check.csv`.

---

### 15. Evaluator Walkthrough (30–60 Second Judge Pitch)
When presenting to SIH judges, execute this concise narrative:

1. **Mission (0–10s):** *"Judges, VISTAARA solves the core dilemma of satellite super-resolution: at satellite pass time, high-resolution ground truth does not exist. A sharper image is not automatically the most suitable image. VISTAARA is an objective, model-agnostic evaluation and decision layer that determines which candidate model is fit for a specific geospatial task."*
2. **Input & Candidates (10–20s):** *"We ingest a Sentinel-2 10m scene and evaluate two candidates sharing the exact same 580,000-parameter CNN backbone. Candidate A has a physical sensor conservation constraint; Candidate B is unconstrained."*
3. **Candidate Comparison (20–30s):** *"Visually, Candidate B has higher raw edge energy and looks crisper. But is it reliable? Let's check VISTAARA."*
4. **VISTAARA Evaluation (30–45s):** *"VISTAARA reveals that Candidate B suffered a severe drop in sensor conservation ($0.95 \to 0.80$), causing certified high-reliability coverage to collapse from 86.1% to 3.2%. Candidate B achieved sharpness by distorting physical reflectance."*
5. **Task Decision (45–60s):** *"Under Task A (Biophysical), Candidate A is chosen because it preserves NDVI. Under Task B (Urban Structural), applying VISTAARA reliability gating suppresses 91.91% of false positive urban detections, increasing precision from 94.0% to 99.4%."*
