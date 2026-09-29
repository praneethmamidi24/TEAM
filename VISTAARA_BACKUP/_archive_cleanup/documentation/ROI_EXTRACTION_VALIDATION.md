# Small Real ROI Extraction Validation Report

**Generated**: 2026-09-17 11:29:07 UTC
**Primary Data Source**: Element 84 Earth Search (AWS Sentinel-2 L2A COGs)

---

## Delhi NCR (Train — Urban)

- **Scene ID**: `S2B_43RFN_20250324_0_L2A`
- **MGRS Tile**: `MGRS-43RFN`
- **CRS**: `EPSG:32643`
- **10m Array Shape**: `[1024, 1024]` (Dtype: `uint16`)
- **SCL Array Shape**: `[512, 512]` (Native 20m, Dtype: `uint8`)
- **Spatial Alignment**: All 10m bands aligned = `True`, Bounds match = `True`
- **Valid Surface Pixels**: **100.0%**
- **Total Cloud & Shadow**: **0.0%** (Cirrus: 0.0%, Shadow: 0.0%)
- **Nodata Pixels**: **0.0%**
- **Quality Gate**: **PASSED**

### Measured File Sizes on Disk

| Format | Measured Size | Status |
| :--- | :---: | :--- |
| **Compressed UInt16 GeoTIFF (4-Band 10m)** | **5.86 MB** | DEFLATE compressed |
| **Compressed UInt8 GeoTIFF (SCL 20m)** | **0.01 MB** | DEFLATE compressed |
| **Total GeoTIFF Pair (4-Band + SCL)** | **5.87 MB** | Within analytical model |
| **Compressed NPZ Package** | **5.91 MB** | `np.savez_compressed` |

- **Diagnostic Visualization**: `data/sample_rois\diagnostic_delhi_ncr.png`

---

## Punjab Ludhiana (Train — Agricultural)

- **Scene ID**: `S2B_43RFP_20241207_0_L2A`
- **MGRS Tile**: `MGRS-43RFP`
- **CRS**: `EPSG:32643`
- **10m Array Shape**: `[1024, 1024]` (Dtype: `uint16`)
- **SCL Array Shape**: `[512, 512]` (Native 20m, Dtype: `uint8`)
- **Spatial Alignment**: All 10m bands aligned = `True`, Bounds match = `True`
- **Valid Surface Pixels**: **100.0%**
- **Total Cloud & Shadow**: **0.0%** (Cirrus: 0.0%, Shadow: 0.0%)
- **Nodata Pixels**: **0.0%**
- **Quality Gate**: **PASSED**

### Measured File Sizes on Disk

| Format | Measured Size | Status |
| :--- | :---: | :--- |
| **Compressed UInt16 GeoTIFF (4-Band 10m)** | **5.64 MB** | DEFLATE compressed |
| **Compressed UInt8 GeoTIFF (SCL 20m)** | **0.02 MB** | DEFLATE compressed |
| **Total GeoTIFF Pair (4-Band + SCL)** | **5.66 MB** | Within analytical model |
| **Compressed NPZ Package** | **5.62 MB** | `np.savez_compressed` |

- **Diagnostic Visualization**: `data/sample_rois\diagnostic_punjab_ludhiana.png`

---

## Tamil Nadu Cauvery Delta (Test — Agricultural)

- **Scene ID**: `S2B_44PKT_20250504_0_L2A`
- **MGRS Tile**: `MGRS-44PKT`
- **CRS**: `EPSG:32644`
- **10m Array Shape**: `[1024, 1024]` (Dtype: `uint16`)
- **SCL Array Shape**: `[512, 512]` (Native 20m, Dtype: `uint8`)
- **Spatial Alignment**: All 10m bands aligned = `True`, Bounds match = `True`
- **Valid Surface Pixels**: **0.0%**
- **Total Cloud & Shadow**: **0.0%** (Cirrus: 0.0%, Shadow: 0.0%)
- **Nodata Pixels**: **100.0%**
- **Quality Gate**: **FAILED**

### Measured File Sizes on Disk

| Format | Measured Size | Status |
| :--- | :---: | :--- |
| **Compressed UInt16 GeoTIFF (4-Band 10m)** | **0.05 MB** | DEFLATE compressed |
| **Compressed UInt8 GeoTIFF (SCL 20m)** | **0.0 MB** | DEFLATE compressed |
| **Total GeoTIFF Pair (4-Band + SCL)** | **0.05 MB** | Within analytical model |
| **Compressed NPZ Package** | **0.01 MB** | `np.savez_compressed` |

- **Diagnostic Visualization**: `data/sample_rois\diagnostic_tn_cauvery_delta.png`

---

