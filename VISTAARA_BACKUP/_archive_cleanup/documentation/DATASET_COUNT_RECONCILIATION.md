# Dataset Count Reconciliation & Authoritative Dataset Audit

**Generated**: 2026-09-17 UTC  
**Project**: VISTAARA Project 2 — OCR-GSR Experimental Module  
**Audit Target**: Complete Reconciliation of Dataset Scenes, Splits, MGRS Tiles, ROI Counts, and Storage Projections  

---

## Executive Summary

A comprehensive forensic audit of the project metadata was conducted across all actual metadata files:
1. `ocr_gsr_discovered_scenes.json`
2. `deep_scene_validation_report.json`
3. `roi_extraction_validation.json`
4. `cauvery_three_roi_validation.json`
5. `authoritative_dataset_inventory.json`

### Key Audit Findings

1. **Authoritative Dataset Identity**:  
   The actual metadata in `ocr_gsr_discovered_scenes.json` and `deep_scene_validation_report.json` contains **exactly 14 scenes**, organized into:
   - **Train**: **6 scenes** (18 planned ROIs)
   - **Validation**: **4 scenes** (12 planned ROIs)
   - **Test**: **4 scenes** (12 planned ROIs)
   - **Total**: **14 scenes** $\times$ **3 ROIs/scene** = **42 planned ROIs**

2. **Alignment with the Original 14-Scene Plan**:  
   The 14 scenes in the authoritative metadata files are a **100% exact match** to the original 14-scene plan. Not a single scene was missing, renamed, or reassigned in the underlying metadata.

3. **Origin of Discrepancies (Ahmedabad, Varanasi, Kolkata, Nizamabad)**:  
   In the previous turn's report (`validate_cauvery_three_rois.py` lines 521–560 and `cauvery_three_roi_validation.md`), an ungrounded, fabricated 14-scene breakdown was introduced. It hallucinated scenes (*Ahmedabad*, *UP Varanasi*, *Kolkata*), misattributed MGRS tiles (*Nizamabad* for *Godavari Delta*), and swapped splits (*Bengaluru* moved to Test, *Kaziranga* moved to Val, *Kanha* and *Sundarbans* moved to Train) to force an artificial 8 Train / 3 Val / 3 Test structure.

4. **Invalidity of the 42-to-38 ROI Claim**:  
   The previous claim that reducing the dataset from 42 to 38 ROIs "preserves 5 ROIs per category across 4 Train classes" was mathematically derived from the hallucinated 8-scene Train split ($4 \times 2 = 8$). In reality, the true Train split has **6 scenes** (Urban: 2, Agricultural: 2, Forest: 1, Water: 1). Dropping 1 ROI per category would create an unbalanced 5 / 5 / 2 / 2 distribution. The 38-ROI proposal as previously formulated is **scientifically invalid and rejected**.

---

## 1. Actual Authoritative Dataset Inventory

The authoritative dataset consists of the following 14 scenes, all verified for reachability, valid geometry, and zero MGRS duplication:

| # | Region ID | Scene Name | Split | Category | MGRS Tile | STAC Item ID | CRS | Planned ROIs | Physical Extraction Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :-: | :--- |
| **1** | `URB_01_DELHI` | Delhi NCR | **Train** | Urban | `43RFN` | `S2B_43RFN_20250324_0_L2A` | EPSG:32643 | 3 | ROI 1: **EXTRACTED (Passed)**; ROIs 2–3: Pending |
| **2** | `URB_02_BLR` | Bengaluru | **Train** | Urban | `43PGQ` | `S2B_43PGQ_20250206_0_L2A` | EPSG:32643 | 3 | ROIs 1–3: Pending Download |
| **3** | `AGR_01_PUNJAB` | Punjab Ludhiana Plains | **Train** | Agricultural | `43RFP` | `S2B_43RFP_20241207_0_L2A` | EPSG:32643 | 3 | ROI 1: **EXTRACTED (Passed)**; ROIs 2–3: Pending |
| **4** | `AGR_02_MALWA` | Madhya Pradesh Malwa | **Train** | Agricultural | `43QEF` | `S2B_43QEF_20250202_0_L2A` | EPSG:32643 | 3 | ROIs 1–3: Pending Download |
| **5** | `FOR_01_WGHATS` | Western Ghats Agumbe | **Train** | Forest | `43PER` | `S2C_43PER_20250224_0_L2A` | EPSG:32643 | 3 | ROIs 1–3: Pending Download |
| **6** | `WAT_01_CHILIKA` | Chilika Lake Lagoon | **Train** | Water & Mixed | `45QTB` | `S2B_45QTB_20250121_0_L2A` | EPSG:32645 | 3 | ROIs 1–3: Pending Download |
| **7** | `URB_03_MUM` | Mumbai MMR | **Validation** | Urban | `43QCB` | `S2B_43QCB_20250205_0_L2A` | EPSG:32643 | 3 | ROIs 1–3: Pending Download |
| **8** | `AGR_03_GODAVARI`| Andhra Godavari Delta | **Validation** | Agricultural | `44QNE` | `S2B_44QNE_20250213_0_L2A` | EPSG:32644 | 3 | ROIs 1–3: Pending Download |
| **9** | `FOR_02_KANHA` | Central India Kanha | **Validation** | Forest | `44QMK` | `S2B_44QMK_20250407_0_L2A` | EPSG:32644 | 3 | ROIs 1–3: Pending Download |
| **10**| `WAT_02_SUNDAR` | Sundarbans Estuary | **Validation** | Water & Mixed | `45QXD` | `S2B_45QXD_20250204_0_L2A` | EPSG:32645 | 3 | ROIs 1–3: Pending Download |
| **11**| `URB_04_HYD` | Hyderabad | **Test** | Urban | `43QHV` | `S2B_43QHV_20250328_0_L2A` | EPSG:32643 | 3 | ROIs 1–3: Pending Download |
| **12**| `AGR_04_CAUVERY`| Tamil Nadu Cauvery Delta | **Test** | Agricultural | `44PKT` | `S2B_44PKT_20250504_0_L2A` | EPSG:32644 | 3 | ROIs 1, 2, 3: **ALL EXTRACTED (Passed)** |
| **13**| `FOR_03_ASSAM` | Assam Kaziranga Buffer | **Test** | Forest | `46REQ` | `S2A_46REQ_20241228_0_L2A` | EPSG:32646 | 3 | ROIs 1–3: Pending Download |
| **14**| `WAT_03_VEMBANAD`| Vembanad Backwaters | **Test** | Water & Mixed | `43PFL` | `S2A_43PFL_20250105_0_L2A` | EPSG:32643 | 3 | ROIs 1–3: Pending Download |

---

## 2. Comparison with the Original 14-Scene Plan

A side-by-side comparison between the user's original planned scenes and the authoritative metadata demonstrates complete alignment:

| Split | Original Planned Scene | Actual Metadata Scene ID | Actual MGRS Tile | Status in Metadata |
| :--- | :--- | :--- | :---: | :--- |
| **TRAIN** | Delhi NCR | `S2B_43RFN_20250324_0_L2A` | `43RFN` | **MATCH (Present in Train)** |
| **TRAIN** | Bengaluru | `S2B_43PGQ_20250206_0_L2A` | `43PGQ` | **MATCH (Present in Train)** |
| **TRAIN** | Punjab Ludhiana | `S2B_43RFP_20241207_0_L2A` | `43RFP` | **MATCH (Present in Train)** |
| **TRAIN** | MP Malwa Plateau | `S2B_43QEF_20250202_0_L2A` | `43QEF` | **MATCH (Present in Train)** |
| **TRAIN** | Western Ghats Agumbe | `S2C_43PER_20250224_0_L2A` | `43PER` | **MATCH (Present in Train)** |
| **TRAIN** | Chilika Lagoon | `S2B_45QTB_20250121_0_L2A` | `45QTB` | **MATCH (Present in Train)** |
| **VAL** | Mumbai MMR | `S2B_43QCB_20250205_0_L2A` | `43QCB` | **MATCH (Present in Validation)** |
| **VAL** | Godavari Delta | `S2B_44QNE_20250213_0_L2A` | `44QNE` | **MATCH (Present in Validation)** |
| **VAL** | Kanha Deciduous | `S2B_44QMK_20250407_0_L2A` | `44QMK` | **MATCH (Present in Validation)** |
| **VAL** | Sundarbans Estuary | `S2B_45QXD_20250204_0_L2A` | `45QXD` | **MATCH (Present in Validation)** |
| **TEST** | Hyderabad | `S2B_43QHV_20250328_0_L2A` | `43QHV` | **MATCH (Present in Test)** |
| **TEST** | Tamil Nadu Cauvery Delta | `S2B_44PKT_20250504_0_L2A` | `44PKT` | **MATCH (Present in Test)** |
| **TEST** | Assam Kaziranga | `S2A_46REQ_20241228_0_L2A` | `46REQ` | **MATCH (Present in Test)** |
| **TEST** | Vembanad Backwaters | `S2A_43PFL_20250105_0_L2A` | `43PFL` | **MATCH (Present in Test)** |

---

## 3. Forensic Analysis of Discrepancies

### Why Did the Previous Report Mention Ahmedabad, Varanasi, Kolkata, and Nizamabad?

In the previous turn, script `validate_cauvery_three_rois.py` included a hardcoded narrative section (lines 521–560) attempting to rationalize a reduction from 42 to 38 ROIs. The assistant fabricated an imaginary dataset table:

```
Fabricated Table (from previous report):
Train (8 scenes): Urban (Delhi, Ahmedabad), Agri (Punjab, Varanasi), Forest (Shimoga, Kanha), Water (Chilika, Sundarbans)
Val (3 scenes): Urban (Kolkata), Agri (Nizamabad), Forest (Kaziranga)
Test (3 scenes): Urban (Bengaluru), Agri (Cauvery), Water (Vembanad)
```

### Detailed Discrepancy Breakdown

1. **Ahmedabad (`43QEB`)**: Completely fabricated. Does not exist in any project metadata or STAC query.
2. **UP Varanasi (`44QLF`)**: Completely fabricated. Does not exist in any project metadata or STAC query.
3. **Kolkata (`45QXB`)**: Completely fabricated. Does not exist in any project metadata or STAC query.
4. **Nizamabad (`44QNE`)**: MGRS tile `44QNE` actually belongs to **Andhra Godavari Delta** (`AGR_03_GODAVARI`). It was erroneously renamed "Telangana Nizamabad" in the narrative.
5. **Western Ghats Shimoga (`43PCM`)**: The actual scene is **Western Ghats Agumbe** (`FOR_01_WGHATS`, tile `43PER`).
6. **MP Kanha Deciduous (`44QMK`)**: The actual scene is in **Validation**, but the narrative placed it in Train under tile `44QKG`.
7. **Sundarbans Estuary (`45QXD`)**: The actual scene is in **Validation**, but the narrative placed it in Train under tile `45QXF`.
8. **Assam Kaziranga Buffer (`46REQ`)**: The actual scene is in **Test**, but the narrative placed it in Validation under tile `46RCN`.
9. **Bengaluru (`43PGQ`)**: The actual scene is in **Train**, but the narrative moved it to Test under tile `43PFS`.
10. **Hyderabad (`43QHV`) & Mumbai MMR (`43QCB`)**: Omitted entirely from the narrative table.

> [!CAUTION]
> **Root Cause**: The narrative in the previous report was synthesized without querying the authoritative JSON metadata files. The author attempted to construct an artificial 2-per-category Train split ($4 \times 2 = 8$ scenes) and 1-per-category Val and Test splits ($3$ scenes each) to invent a symmetric 38-ROI justification. **None of those modifications ever occurred in the underlying metadata files.**

---

## 4. Authoritative Verification of Counts & Balance

### Exact Scene and ROI Counts

$$\text{Total Scenes} = 6 \text{ (Train)} + 4 \text{ (Validation)} + 4 \text{ (Test)} = \mathbf{14 \text{ Scenes}}$$

$$\text{Total ROIs} = 14 \text{ Scenes} \times 3 \text{ ROIs/Scene} = \mathbf{42 \text{ ROIs}}$$

$$\text{Train ROIs} = 6 \times 3 = \mathbf{18 \text{ ROIs}}$$
$$\text{Validation ROIs} = 4 \times 3 = \mathbf{12 \text{ ROIs}}$$
$$\text{Test ROIs} = 4 \times 3 = \mathbf{12 \text{ ROIs}}$$
$$\text{Overall Sum Check} = 18 + 12 + 12 = \mathbf{42 \text{ ROIs}} \quad (\text{Exact Match})$$

### Category Distribution (Scenes and Planned ROIs)

| Category | Train Scenes (ROIs) | Validation Scenes (ROIs) | Test Scenes (ROIs) | Total Scenes | Total Planned ROIs | Representation % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Urban** | 2 (6) | 1 (3) | 1 (3) | **4** | **12** | $28.57\%$ |
| **Agricultural** | 2 (6) | 1 (3) | 1 (3) | **4** | **12** | $28.57\%$ |
| **Forest** | 1 (3) | 1 (3) | 1 (3) | **3** | **9** | $21.43\%$ |
| **Water & Mixed** | 1 (3) | 1 (3) | 1 (3) | **3** | **9** | $21.43\%$ |
| **Total** | **6 (18)** | **4 (12)** | **4 (12)** | **14** | **42** | **100.0%** |

### MGRS Tile Uniqueness & Spatial Disjointness

- **Total MGRS Tiles**: 14
- **Unique MGRS Tiles**: 14
- **Cross-Split Duplicate Tiles**: **0**
- **Minimum Geographic Separation Across Splits**:
  - Train $\leftrightarrow$ Validation: **$435.5\text{ km}$** (Western Ghats Agumbe to Mumbai MMR)
  - Train $\leftrightarrow$ Test: **$295.6\text{ km}$** (Bengaluru to TN Cauvery Delta)
  - Validation $\leftrightarrow$ Test: **$353.4\text{ km}$** (Godavari Delta to Hyderabad)
- **Spatial Overlap**: **$0.0\text{ m}^2$** across all scene pairs.

---

## 5. Scientific Assessment of the 42-to-38 ROI Proposal

### What Was the Proposed 38-ROI Plan?
The previous report proposed removing 4 ROIs (reducing from 42 to 38) under the claim that dropping 1 ROI from each of the 4 Train categories preserves "exact 5-ROI symmetry across all 4 classes".

### Why This Claim Is Factually False
1. **The Train Split Does NOT Have 8 Scenes**:  
   The actual Train split contains **6 scenes**, not 8.
   - Urban has 2 scenes ($6$ ROIs)
   - Agricultural has 2 scenes ($6$ ROIs)
   - Forest has 1 scene ($3$ ROIs)
   - Water & Mixed has 1 scene ($3$ ROIs)
2. **Dropping 1 ROI per Category Destroys Balance**:  
   If 1 ROI were dropped from each category in Train:
   - Urban becomes: $6 - 1 = 5$ ROIs
   - Agricultural becomes: $6 - 1 = 5$ ROIs
   - Forest becomes: $3 - 1 = \mathbf{2}$ ROIs
   - Water & Mixed becomes: $3 - 1 = \mathbf{2}$ ROIs
   This results in a heavily skewed **5 / 5 / 2 / 2** training distribution where Forest and Water & Mixed are severely under-represented ($2$ ROIs each, from a single geographic scene).
3. **Storage Reality vs. Motivation**:  
   The motivation for proposing 38 ROIs was fear of exceeding the 250 MB environment cap. However, actual measurements demonstrate:
   - Measured average per ROI: **$5.78\text{ MB}$** (GeoTIFF pair)
   - $42\text{ ROIs} \times 5.78\text{ MB} = \mathbf{242.6\text{ MB}}$
   - Headroom to $250.0\text{ MB}$: **$+7.4\text{ MB}$ ($3.0\%$)**
   While tight, 42 ROIs is physically compliant with the budget when stored with lossless DEFLATE compression.

### Scientific Verdict
The 38-ROI proposal is **REJECTED**. The benchmark must proceed with the authoritative **42-ROI dataset** ($18$ Train, $12$ Val, $12$ Test). Any future storage adjustments must be grounded in the real 6 / 4 / 4 split structure and must never be based on fabricated scenes or false symmetries.

---

## 6. Physical Extraction Status Summary

As of 2026-09-17, physical downloads have been strictly restricted to small validation batches to preserve the environment quota:

| Scene | Region ID | MGRS | Split | Extracted ROIs | Valid Surface % | SCL Status | Measured File Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delhi NCR** | `URB_01_DELHI` | `43RFN` | Train | ROI 1 (`row:1800, col:1800`) | $100.0\%$ | **PASSED** | $5.87\text{ MB}$ |
| **Punjab Ludhiana** | `AGR_01_PUNJAB` | `43RFP` | Train | ROI 1 (`row:1800, col:1800`) | $100.0\%$ | **PASSED** | $5.66\text{ MB}$ |
| **Cauvery Delta North** | `AGR_04_CAUVERY` | `44PKT` | Test | ROI 1 (`row:1800, col:8500`) | $100.0\%$ | **PASSED** | $5.79\text{ MB}$ |
| **Cauvery Delta Central** | `AGR_04_CAUVERY` | `44PKT` | Test | ROI 2 (`row:5000, col:7500`) | $100.0\%$ | **PASSED** | $5.77\text{ MB}$ |
| **Cauvery Delta South** | `AGR_04_CAUVERY` | `44PKT` | Test | ROI 3 (`row:8000, col:8000`) | $100.0\%$ | **PASSED** | $5.79\text{ MB}$ |

- **Total Physically Extracted & Validated ROIs**: **5 ROIs**
- **Pending Planned ROIs**: **37 ROIs** (Metadata verified; pixels not yet downloaded)
- **Total Dataset Size on Disk Currently**: $\approx 28.9\text{ MB}$ (5 valid ROI GeoTIFF pairs)

---

## 7. List of Unresolved Issues & Next Verification Gates

1. **Swath Margin Verification for Remaining 11 Scenes**:  
   As proven by Cauvery Delta (`44PKT`), descending orbital swaths cut diagonally across MGRS granules, resulting in nodata wedges at arbitrary corners. The default planned interior coordinates (`1800, 4978, 8156`) must be dynamically verified against valid swath boundaries before bulk extraction to avoid nodata rejections.
2. **Dynamic In-Memory Batching**:  
   To prevent uncompressed float32 scratch tensors ($16.0\text{ MB}$ per ROI $\times 42 = 672\text{ MB}$) from accumulating on disk during model execution, dataloaders must stream patches directly from compressed GeoTIFFs into RAM without intermediate caching.
3. **Mandatory Protocol Safeguards**:  
   - Baseline SEN2SRLite architecture, weights, and evaluation code remain strictly pristine.
   - OCR-GSR module remains completely disconnected until dataset validation is fully complete.
   - No model training has been initiated.
