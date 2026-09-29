# Complete 42-ROI Physical Extraction & Quality Audit Report

**Generated**: 2026-09-17 17:09:48 UTC
**Authoritative Scope**: 14 Scenes x 3 ROIs = 42 ROIs (18 Train, 12 Validation, 12 Test)
**Audit Outcome**: **29 PASSED**, **13 REJECTED** (Pass Rate: 69.0%)

---

## 1. Executive Summary & Verification Methodology

Every planned ROI across all 14 authoritative Sentinel-2 L2A scenes in India was physically queried and evaluated against the native optical bands (`B02`, `B03`, `B04`, `B08` at 10m GSD) and the native Scene Classification Layer (`SCL` at 20m GSD).

### Quality Thresholds Enforced
1. **Valid Surface Reflectance**: $\ge 98.0\%$
2. **Cloud & Cloud Shadow**: $\le 2.0\%$
3. **Zero Nodata Tolerance**: Strictly $0.0\%$ (swath margin and nodata wedge detection)
4. **Spatial & CRS Alignment**: Exact spatial bounding box match ($0.0\text{ m}$ difference between 10m optical and 20m SCL)
5. **Coordinate Policy**: In strict adherence to experimental instructions, **no coordinates were automatically altered**. All rejected ROIs are reported with their exact failure mechanism.

---

## 2. Complete 42-ROI Quality Audit Inventory

| ROI UID | Scene Name | Split | Category | MGRS | Window (row, col) | Valid Surface % | Nodata % | Cloud % | Status | File Size | Audit Verdict / Failure Mechanism |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `URB_01_DELHI_ROI_1` | Delhi NCR | Train | Urban | `MGRS-43RFN` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.87 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `URB_01_DELHI_ROI_2` | Delhi NCR | Train | Urban | `MGRS-43RFN` | `(4978, 4978)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.88 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `URB_01_DELHI_ROI_3` | Delhi NCR | Train | Urban | `MGRS-43RFN` | `(8156, 8156)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.94 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `URB_02_BLR_ROI_1` | Bengaluru | Train | Urban | `MGRS-43PGQ` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.95 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `URB_02_BLR_ROI_2` | Bengaluru | Train | Urban | `MGRS-43PGQ` | `(4978, 4978)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.99 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `URB_02_BLR_ROI_3` | Bengaluru | Train | Urban | `MGRS-43PGQ` | `(8156, 8156)` | 100.00% | 0.00% | 0.00% | **PASSED** | 6.11 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `URB_03_MUM_ROI_1` | Mumbai MMR | Validation | Urban | `MGRS-43QCB` | `(1800, 1800)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `URB_03_MUM_ROI_2` | Mumbai MMR | Validation | Urban | `MGRS-43QCB` | `(4978, 4978)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `URB_03_MUM_ROI_3` | Mumbai MMR | Validation | Urban | `MGRS-43QCB` | `(8156, 8156)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `URB_04_HYD_ROI_1` | Hyderabad | Test | Urban | `MGRS-43QHV` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.74 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `URB_04_HYD_ROI_2` | Hyderabad | Test | Urban | `MGRS-43QHV` | `(4978, 4978)` | 100.00% | 0.00% | 0.00% | **PASSED** | 6.06 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `URB_04_HYD_ROI_3` | Hyderabad | Test | Urban | `MGRS-43QHV` | `(8156, 8156)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.95 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_01_PUNJAB_ROI_1` | Punjab Ludhiana Plains | Train | Agricultural | `MGRS-43RFP` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.66 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_01_PUNJAB_ROI_2` | Punjab Ludhiana Plains | Train | Agricultural | `MGRS-43RFP` | `(4978, 4978)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `AGR_01_PUNJAB_ROI_3` | Punjab Ludhiana Plains | Train | Agricultural | `MGRS-43RFP` | `(8156, 8156)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `AGR_02_MALWA_ROI_1` | Madhya Pradesh Malwa | Train | Agricultural | `MGRS-43QEF` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.83 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_02_MALWA_ROI_2` | Madhya Pradesh Malwa | Train | Agricultural | `MGRS-43QEF` | `(4978, 4978)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.75 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_02_MALWA_ROI_3` | Madhya Pradesh Malwa | Train | Agricultural | `MGRS-43QEF` | `(8156, 8156)` | 100.00% | 0.00% | 0.00% | **PASSED** | 6.11 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_03_GODAVARI_ROI_1` | Andhra Godavari Delta | Validation | Agricultural | `MGRS-44QNE` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.65 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_03_GODAVARI_ROI_2` | Andhra Godavari Delta | Validation | Agricultural | `MGRS-44QNE` | `(4978, 4978)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.67 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_03_GODAVARI_ROI_3` | Andhra Godavari Delta | Validation | Agricultural | `MGRS-44QNE` | `(8156, 8156)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.74 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_04_CAUVERY_ROI_1` | Tamil Nadu Cauvery Delta | Test | Agricultural | `MGRS-44PKT` | `(1800, 8500)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.79 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_04_CAUVERY_ROI_2` | Tamil Nadu Cauvery Delta | Test | Agricultural | `MGRS-44PKT` | `(5000, 7500)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.77 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `AGR_04_CAUVERY_ROI_3` | Tamil Nadu Cauvery Delta | Test | Agricultural | `MGRS-44PKT` | `(8000, 8000)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.79 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_01_WGHATS_ROI_1` | Western Ghats Agumbe | Train | Forest | `MGRS-43PER` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.76 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_01_WGHATS_ROI_2` | Western Ghats Agumbe | Train | Forest | `MGRS-43PER` | `(4978, 4978)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.79 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_01_WGHATS_ROI_3` | Western Ghats Agumbe | Train | Forest | `MGRS-43PER` | `(8156, 8156)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.75 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_02_KANHA_ROI_1` | Central India Kanha | Validation | Forest | `MGRS-44QMK` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.76 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_02_KANHA_ROI_2` | Central India Kanha | Validation | Forest | `MGRS-44QMK` | `(4978, 4978)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.82 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_02_KANHA_ROI_3` | Central India Kanha | Validation | Forest | `MGRS-44QMK` | `(8156, 8156)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.77 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_03_ASSAM_ROI_1` | Assam Kaziranga Buffer | Test | Forest | `MGRS-46REQ` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.90 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_03_ASSAM_ROI_2` | Assam Kaziranga Buffer | Test | Forest | `MGRS-46REQ` | `(4978, 4978)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.91 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `FOR_03_ASSAM_ROI_3` | Assam Kaziranga Buffer | Test | Forest | `MGRS-46REQ` | `(8156, 8156)` | 100.00% | 0.00% | 0.00% | **PASSED** | 5.94 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `WAT_01_CHILIKA_ROI_1` | Chilika Lake Lagoon | Train | Water & Mixed | `MGRS-45QTB` | `(1800, 1800)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `WAT_01_CHILIKA_ROI_2` | Chilika Lake Lagoon | Train | Water & Mixed | `MGRS-45QTB` | `(4978, 4978)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `WAT_01_CHILIKA_ROI_3` | Chilika Lake Lagoon | Train | Water & Mixed | `MGRS-45QTB` | `(8156, 8156)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `WAT_02_SUNDAR_ROI_1` | Sundarbans Estuary | Validation | Water & Mixed | `MGRS-45QXD` | `(1800, 1800)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `WAT_02_SUNDAR_ROI_2` | Sundarbans Estuary | Validation | Water & Mixed | `MGRS-45QXD` | `(4978, 4978)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `WAT_02_SUNDAR_ROI_3` | Sundarbans Estuary | Validation | Water & Mixed | `MGRS-45QXD` | `(8156, 8156)` | 4.33% | 95.67% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 95.6734% nodata; Low valid surface: 4.3266% (threshold >= 98.0%) |
| `WAT_03_VEMBANAD_ROI_1` | Vembanad Backwaters | Test | Water & Mixed | `MGRS-43PFL` | `(1800, 1800)` | 100.00% | 0.00% | 0.00% | **PASSED** | 4.71 MB | Passed Quality Gate (100% Valid Clear Surface) |
| `WAT_03_VEMBANAD_ROI_2` | Vembanad Backwaters | Test | Water & Mixed | `MGRS-43PFL` | `(4978, 4978)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |
| `WAT_03_VEMBANAD_ROI_3` | Vembanad Backwaters | Test | Water & Mixed | `MGRS-43PFL` | `(8156, 8156)` | 0.00% | 100.00% | 0.00% | **REJECTED** | — | Swath margin nodata wedge: 100.0% nodata; Low valid surface: 0.0% (threshold >= 98.0%) |

---

## 3. Dataset Balance Analysis across Physically Valid ROIs

### Split & Category Distribution Comparison

| Dimension | Category / Split | Planned ROIs | Physically Valid ROIs | Retention Rate | Status & Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Split** | Train | 18 | **13** | 72.2% | Robust training base across 5 scenes |
| **Split** | Validation | 12 | **6** | 50.0% | 6 ROIs across Godavari & Kanha; Mumbai & Sundarbans in swath margin |
| **Split** | Test | 12 | **10** | 83.3% | Strong benchmark base across Hyderabad, Cauvery, Kaziranga, Vembanad |
| **Category** | Urban | 12 | **9** | 75.0% | Excellent: Delhi (3), Bengaluru (3), Hyderabad (3) fully passed |
| **Category** | Agricultural | 12 | **10** | 83.3% | Excellent: Malwa (3), Godavari (3), Cauvery (3), Punjab (1) fully passed |
| **Category** | Forest | 9 | **9** | 100.0% | **100% Passed**: Agumbe (3), Kanha (3), Kaziranga (3) fully intact |
| **Category** | Water & Mixed | 9 | **1** | 11.1% | Action needed: Chilika & Sundarbans in swath margin; Vembanad ROI 1 passed |
| **Total** | **All Categories** | **42** | **29** | **69.0%** | **29 Physically Valid ROIs on Disk** |

### Analysis of the 13 Rejected ROIs

Every single one of the 13 rejected ROIs failed due to **orbital swath margin nodata wedges** ($95.7\%$ to $100.0\%$ nodata) caused by the satellite's diagonal descending orbit crossing rectangular MGRS tiles:

1. **Mumbai MMR (`URB_03_MUM`, 43QCB)** — ROIs 1, 2, 3 ($100\%$ nodata):
   - *Diagnosis*: Scene has $77.05\%$ nodata. The valid orbital strip is located exclusively in the Eastern portion of the tile (longitudes $> 73.1^\circ\text{E}$).
   - *Recommendation*: Can be safely adjusted within the same scene by selecting columns in the Eastern valid strip, or preserved as pending.
2. **Punjab Ludhiana (`AGR_01_PUNJAB`, 43RFP)** — ROIs 2, 3 ($100\%$ nodata):
   - *Diagnosis*: Scene has $65.67\%$ nodata. ROI 1 at `(1800, 1800)` is in the valid strip and passed. ROIs 2 and 3 fall beyond the diagonal orbital edge.
   - *Recommendation*: Can be safely adjusted within the valid Eastern strip of the same scene.
3. **Chilika Lake (`WAT_01_CHILIKA`, 45QTB)** — ROIs 1, 2, 3 ($100\%$ nodata):
   - *Diagnosis*: Scene has $87.01\%$ nodata. The default coordinates fall into the unrecorded Western half of tile 45QTB.
   - *Recommendation*: Valid lagoon waters are concentrated at columns $8000-10500$. Can be safely adjusted within the same scene.
4. **Sundarbans Estuary (`WAT_02_SUNDAR`, 45QXD)** — ROIs 1, 2, 3 ($95.7-100\%$ nodata):
   - *Diagnosis*: Scene has $81.14\%$ nodata. Swath edge clips columns $< 8500$.
   - *Recommendation*: Can be safely adjusted into the valid Eastern mangrove channels of the same scene.
5. **Vembanad Backwaters (`WAT_03_VEMBANAD`, 43PFL)** — ROIs 2, 3 ($100\%$ nodata):
   - *Diagnosis*: Scene has $67.74\%$ nodata. ROI 1 at `(1800, 1800)` passed ($100\%$ valid). ROIs 2 and 3 fall into the Southern nodata sector.
   - *Recommendation*: Can be safely adjusted in the Northern backwater channels.

---

## 4. Empirical Storage Audit

### Measured Storage across Successfully Extracted ROIs

- **Physically Extracted & Validated ROIs on Disk**: **29 ROIs**
- **Total Measured Storage (GeoTIFF Pairs)**: **168.36 MB**
- **Empirical Mean Storage per Valid ROI**: **5.81 MB**
- **Empirical Maximum Storage per ROI**: **6.11 MB** (Delhi NCR / Bengaluru complex urban built-up)
- **Empirical Minimum Storage per ROI**: **4.71 MB** (Vembanad homogeneous water)

### Storage Projections vs. 250.0 MB Hard Ceiling

| Projection Model | Formula / Basis | Projected Storage | Headroom to 250 MB | Headroom % | Compliance Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Current 29 Valid ROIs (Actual)** | Empirical Disk Footprint | **168.36 MB** | **+81.6 MB** | **32.7%** | **Compliant (Healthy)** |
| **Full 42 ROIs (Nominal Mean)** | $42 \times 5.81\text{ MB}$ | **244.0 MB** | **+6.0 MB** | **2.4%** | **Compliant (Within Limit)** |
| **Full 42 ROIs (Worst-Case Peak)** | $42 \times 6.11\text{ MB}$ | **256.6 MB** | **-6.6 MB** | **-2.6%** | Near Ceiling (Exceeds by 6.6 MB if all ROIs are urban peak) |
| **Full 42 ROIs (+10% Safety Buffer)** | Nominal $\times 1.10$ | **268.4 MB** | **-18.4 MB** | *Deficit* | Exceeds budget if 10% contingency realized |

---

## 5. Audit Conclusions & Next Protocol Gates

1. **Complete Audit Accomplished**: All 42 planned ROI coordinates were queried against actual Sentinel-2 optical bands and SCL layers. 29 ROIs passed all strict radiometric and geometric quality gates and are physically saved on disk as lossless GeoTIFF pairs.
2. **Zero Coordinate Modifications**: In compliance with instructions, no coordinates were changed automatically. The 13 rejected ROIs have been explicitly mapped to orbital swath boundary effects.
3. **Pristine Environment**: Baseline SEN2SRLite model files, weights, and architectures remain 100% untouched. No model training has been initiated.
