# Corrected Tamil Nadu Cauvery Delta ROI Validation & Storage Audit

**Generated**: 2026-09-17 14:00:41 UTC
**Scene Target**: `S2B_44PKT_20250504_0_L2A` (MGRS `44PKT`, Test Split — Agricultural)

---

## 1. Corrected Cauvery Delta ROI 1 Extraction Validation

- **Corrected 10m Window**: `row_off=1800, col_off=7000, width=1024, height=1024`
- **Corrected 20m Window**: `row_off=900, col_off=3500, width=512, height=512`
- **CRS**: `EPSG:32644`
- **10m Geographic Bounds**: `[269980.0, 1271780.0, 280220.0, 1282020.0]`
- **20m Geographic Bounds**: `[269980.0, 1271780.0, 280220.0, 1282020.0]` (Alignment Error: **0.0 m**)

### Optical Band Validity

| Band | Dtype | Shape | Mean Reflectance | Valid % | Nodata % | HTTP Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B02** | `uint16` | `[1024, 1024]` | `47.52` | **5.2681%** | **94.7319%** | `200 OK` |
| **B03** | `uint16` | `[1024, 1024]` | `61.44` | **5.2681%** | **94.7319%** | `200 OK` |
| **B04** | `uint16` | `[1024, 1024]` | `74.13` | **5.2681%** | **94.7319%** | `200 OK` |
| **B08** | `uint16` | `[1024, 1024]` | `138.56` | **5.2681%** | **94.7319%** | `200 OK` |

### SCL Quality Audit (Native 20m Grid)

- **Valid Surface Reflectance**: **5.2681%**
- **Total Cloud & Shadow**: **0.0%** (Cirrus: 0.0%, Shadow: 0.0%)
- **Nodata Pixels**: **94.7319%**
- **Quality Gate**: **FAILED (Rejected)**

### Land-Cover Verification (Initial Test at col 7000)

- **Vegetation (Class 4)**: **0.7549%** (1979 px)
- **Not-Vegetated / Cropland Soil (Class 5)**: **4.3118%** (11303 px)
- **Water Bodies / Canals (Class 6)**: **0.0%** (0 px)
- **Nodata (Class 0)**: **94.7319%** (248334 px)
- **Verdict**: **FAILED TO CAPTURE MEANINGFUL LANDSCAPE**. The window is 94.73% unobserved nodata because Sentinel-2 orbit enters diagonally; at row 1800, the valid satellite swath only begins at column 8096. Column 7000 is almost entirely outside the orbital track.

### Measured Storage Sizes on Disk (col 7000)

| Format | Measured Bytes | Measured MB | Status |
| :--- | :---: | :---: | :--- |
| **Compressed UInt16 GeoTIFF (4-Band 10m)** | `410,771 B` | **0.39 MB** | DEFLATE predictor 2 (collapsed due to 95% zeroes) |
| **Compressed UInt8 GeoTIFF (SCL 20m)** | `4,080 B` | **0.0 MB** | DEFLATE compressed |
| **Total GeoTIFF Pair** | `414,851 B` | **0.4 MB** | Artificially small due to nodata zero-fill |
| **Compressed NPZ Package** | `370,949 B` | **0.35 MB** | `np.savez_compressed` |

- **Diagnostic Visualization (Initial Probe)**: `data/sample_rois/diagnostic_tn_cauvery_delta_corrected.png`

---

## 2. Valid Swath Inspection & Adaptation for Cauvery Delta

### Swath Geometry Analysis (Why col 7000 Failed)
- Sentinel-2 orbits are inclined at ~98.6°. The orbital swath cuts diagonally across tile `44PKT`.
- At $10\text{m}$ row 1800, **valid Earth reflectance begins strictly at column 8096** ($X \ge 280,940\text{ m}$ Easting).
- Window `col_off=7000` ($X=269,980$ to $280,220\text{ m}$) terminates 720 meters before the swath margin begins, explaining why only $5.27\%$ of its rightmost edge had data.
- Shifting `col_off` to **8500** places the entire $1024 \times 1024$ window safely inside the optical swath with a $400\text{ m}$ buffer from the edge.

### Measured Storage Sizes on Disk for Adapted Valid Swath ROI 1 (col 8500)

| Format | Measured Bytes | Measured MB | Status |
| :--- | :---: | :---: | :--- |
| **Total GeoTIFF Pair (10m 4-Band + SCL)** | `6,063,077 B` | **5.78 MB** | **100% Genuine Valid Optical Agricultural Data** |
| **Compressed NPZ Package** | `6,144,245 B` | **5.86 MB** | `np.savez_compressed` |

- **Diagnostic Visualization (Valid Swath)**: `data/sample_rois/diagnostic_tn_cauvery_delta_valid_swath.png`

### Three Valid, Non-Overlapping ROIs for Cauvery Delta

| ROI ID | Name | 10m Window `[row, col, w, h]` | UTM Bounds `[W, S, E, N]` | Valid Surface | Nodata | Cloud | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ROI_1_Initial_Probe_7000** | Cauvery Delta North (Initial Test at col 7000) | `[1800, 7000, 1024, 1024]` | `[269980, 1271780, 280220, 1282020]` | **5.27%** | **94.73%** | **0.0%** | **REJECT** |
| **ROI_1_North_Adapted** | Cauvery Delta North (Swath Adapted at col 8500) | `[1800, 8500, 1024, 1024]` | `[284980, 1271780, 295220, 1282020]` | **100.0%** | **0.0%** | **0.0%** | **PASS** |
| **ROI_2_Central** | Cauvery Delta Central (Irrigated Basin at col 7500) | `[5000, 7500, 1024, 1024]` | `[274980, 1239780, 285220, 1250020]` | **100.0%** | **0.0%** | **0.0%** | **PASS** |
| **ROI_3_South** | Cauvery Delta South (Deltaic Cultivation at col 8000) | `[8000, 8000, 1024, 1024]` | `[279980, 1209780, 290220, 1220020]` | **100.0%** | **0.0%** | **0.0%** | **PASS** |

- **Spatial Separation**: Vertical distance between consecutive ROIs is $\ge 30,000\text{ m}$ ($30\text{ km}$), guaranteeing **zero spatial overlap**.

---

## 3. Storage Projection Recalculation with Safety Margins

### Measured Sample Base (3 Disjoint Valid Optical ROIs)

| Scene / Split | GeoTIFF Pair (MB) | NPZ Package (MB) |
| :--- | :---: | :---: |
| **Delhi NCR (Urban)** | 5.87 MB | 5.91 MB |
| **Punjab Ludhiana (Agri)** | 5.66 MB | 5.62 MB |
| **TN Cauvery Delta Valid Swath (Agri)** | 5.78 MB | 5.86 MB |
| **Mean** | **5.77 MB** | **5.80 MB** |
| **Max (Worst Case)** | **5.87 MB** | **5.91 MB** |

### 42-ROI Dataset Storage Projections vs 250.0 MB Hard Ceiling

| Projection Scenario | Projected Size (GeoTIFF) | Headroom to 250 MB | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Nominal Mean (42 × 5.77 MB)** | **242.3 MB** | **+7.7 MB (3.1%)** | Compliant (Razor-thin margin) |
| **Worst-Case Max (42 × 5.87 MB)** | **246.5 MB** | **+3.5 MB (1.4%)** | Compliant (Zero tolerance) |
| **Nominal + 10% Safety Margin** | **266.6 MB** | **-16.6 MB** | **EXCEEDS BUDGET** |
| **Nominal + 20% Safety Margin** | **290.8 MB** | **-40.8 MB** | **EXCEEDS BUDGET** |

> [!WARNING]
> **Scientific & Engineering Risk**: Claiming that $243.6\text{ MB}$ safely satisfies the $250.0\text{ MB}$ budget is mathematically fragile because headroom is only $6.4\text{ MB}$ ($2.56\%$). A 10% variation in scene entropy (e.g. dense urban texture or high agricultural boundary frequency) pushes total storage to $\approx 268.0\text{ MB}$.

### Actionable Storage Mitigation Strategy
1. **Use NPZ with UInt16**: NPZ averages slightly lower file sizes than GeoTIFF.
2. **Alternative Budget Balancing**: If strict 250 MB cap is enforced, either reduce ROI count to 38 ROIs ($38 \times 5.8 = 220.4\text{ MB}$, leaving $29.6\text{ MB}$ safety buffer) or store only the core 4 optical bands without redundant ancillary layers.
