# Verified Demo Outputs: VISTAARA GIS Products & Decision Summaries

**Directory:** `VISTAARA_Project_2_OCR_GSR/modified project/verified_demo_outputs`  
**Purpose:** Pre-computed, verified geospatial rasters and decision summaries generated directly from the deep learning super-resolution pipeline. Allows evaluators and GIS analysts to inspect verified 2.5m outputs immediately without re-running CPU inference.

---

## 1. Summary of Verified Output Products

| Filename | Product Description | Format & Data Type | Dimensions | Spatial GSD | Size |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`Scene_138_Decision_Summary.csv`** | Comprehensive decision metrics & task fitness for Scene 138 | UTF-8 CSV Table | 24 rows | — | 813 B |
| **`Scene_75_Decision_Summary.csv`** | Comprehensive decision metrics & task fitness for Scene 75 | UTF-8 CSV Table | 24 rows | — | 815 B |
| **`Scene_138_CandidateA_2.5m.tif`** | Candidate A Physically Constrained 4-band SR GeoTIFF | 4-Band Float32 GeoTIFF (DEFLATE) | $2048 \times 2048$ | 2.5 m | 57.2 MB |
| **`Scene_138_CandidateB_2.5m.tif`** | Candidate B Unconstrained Data-Driven 4-band SR GeoTIFF | 4-Band Float32 GeoTIFF (DEFLATE) | $2048 \times 2048$ | 2.5 m | 57.3 MB |
| **`Scene_138_VISTAARA_Reliability.tif`** | Dense Continuous Reliability Map $R(x, y)$ | 1-Band Float32 GeoTIFF (DEFLATE) | $2048 \times 2048$ | 2.5 m | 13.2 MB |

---

## 2. Geospatial Integrity & CRS Preservation

Every exported GeoTIFF strictly preserves the original satellite coordinate reference system and rescales the affine transformation matrix:

* **Coordinate Reference System (CRS):** `EPSG:32643` (WGS 84 / UTM Zone 43N)
* **Native Pixel Resolution:** $10.0\text{ m} \times 10.0\text{ m}$ ($512 \times 512$ pixels)
* **Super-Resolved Resolution:** $2.5\text{ m} \times 2.5\text{ m}$ ($2048 \times 2048$ pixels)
* **Magnification:** Exactly $4\times$ spatial magnification ($16\times$ pixel density increase, $4,194,304$ total pixels)
* **Affine Transform Scaling:**
  $$\text{Pixel Spacing} = \frac{\Delta x_{native}}{4} = \frac{10.0\text{ m}}{4} = 2.5\text{ m}$$
  $$\text{Origin Coordinates } (X_0, Y_0) \text{ preserved exactly at } (725560.00, 1469280.00)$$

---

## 3. Band Allocation in 4-Band GeoTIFFs

In strict compliance with Sentinel-2 VNIR physical band specifications:

* **Band 1:** B04 — Red ($665\text{ nm}$)
* **Band 2:** B03 — Green ($560\text{ nm}$)
* **Band 3:** B02 — Blue ($490\text{ nm}$)
* **Band 4:** B08 — Broad NIR ($842\text{ nm}$)

> **True Color RGB Composite:** To display true color in QGIS or ArcGIS, map Band 1 $\to$ Red, Band 2 $\to$ Green, Band 3 $\to$ Blue.  
> **False Color CIR Composite:** Map Band 4 (NIR) $\to$ Red, Band 1 (Red) $\to$ Green, Band 2 (Green) $\to$ Blue.

---

## 4. Key Decision Metrics Comparison (From CSVs)

| Evaluation Parameter | Scene 138: Candidate A | Scene 138: Candidate B | Scene 75: Candidate A | Scene 75: Candidate B |
| :--- | :---: | :---: | :---: | :---: |
| **Reconstruction Consistency ($M_{recon}$)** | **0.924254** | 0.738961 | **0.952137** | 0.805523 |
| **Spectral Consistency ($M_{spec}$)** | **0.892661** | 0.822802 | **0.885639** | 0.810348 |
| **Spatial Consistency ($M_{spat}$)** | 0.997973 | 0.997990 | 0.998314 | 0.998316 |
| **Local Stability ($M_{stab}$)** | **0.969781** | 0.966938 | 0.973965 | 0.974508 |
| **Composite Reliability ($R$)** | **0.928345** | 0.832434 | **0.936258** | 0.855437 |
| **High Reliability Coverage ($R \ge 0.93$)** | **67.00%** | 2.98% | **86.09%** | 3.17% |
| **Task A Fitness (Biophysical / NDVI)** | **0.908457** | 0.780881 | **0.918888** | 0.807936 |
| **Task B Fitness (Structural / Edge)** | **0.974836** | 0.936863 | **0.981775** | 0.952613 |
| **Task C Fitness (Balanced General)** | **0.928345** | 0.832434 | **0.936258** | 0.855437 |
| **Decision Recommendation** | **Candidate A** | Disfavored | **Candidate A** | Disfavored |
| **Urban FP Reduction ($\tau_{rel} = 0.93$)** | N/A (Rural scene) | N/A | **91.91%** | (Drift prevents gating) |

---

## 5. How to Inspect in GIS Software

1. Open **QGIS** or **ArcGIS Pro**.
2. Drag and drop `Scene_138_CandidateA_2.5m.tif` and `Scene_138_VISTAARA_Reliability.tif`.
3. Set Layer Properties:
   - For `CandidateA`: Set Symbology to Multiband Color (Red=Band 1, Green=Band 2, Blue=Band 3), Stretch to MinMax (2%–98% percentile).
   - For `Reliability`: Set Symbology to Singleband Pseudocolor, select colormap `Viridis` or `RdYlGn`, set Min=0.50, Max=1.00.
4. Overlay the layers with 40% transparency on the Reliability layer to observe where low-reliability pixels correspond to complex mixed pixels and water boundaries.
