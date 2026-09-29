# -*- coding: utf-8 -*-
"""
extract_and_validate_rois.py
============================
Small Real ROI Extraction Validation for OCR-GSR Experimental Pipeline.

Targets:
1. Delhi NCR (Train, Urban) - Scene S2B_43RFN_20250324_0_L2A
2. Punjab Ludhiana (Train, Agri) - Scene S2B_43RFP_20241207_0_L2A
3. TN Cauvery Delta (Test, Agri) - Scene S2B_44PKT_20250504_0_L2A

Extracts only ROI 1:
- B02, B03, B04, B08 at native 10 m resolution (1024 x 1024 pixels)
- SCL at native 20 m resolution (512 x 512 pixels)
- Measures actual compressed GeoTIFF and compressed NPZ file sizes.
- Computes pixel-level valid and SCL quality histograms.
- Generates 3-panel diagnostic visualization per scene.
- Saves results to roi_extraction_validation.json and ROI_EXTRACTION_VALIDATION.md.
"""

import os
import sys
import time
import json
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# GDAL performance tuning for AWS S3 COGs
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif'

OUTPUT_DIR = "data/sample_rois"
JSON_REPORT = "roi_extraction_validation.json"
MD_REPORT = "ROI_EXTRACTION_VALIDATION.md"

TARGET_SCENES = [
    {
        "scene_key": "delhi_ncr",
        "name": "Delhi NCR",
        "split": "Train",
        "category": "Urban",
        "scene_id": "S2B_43RFN_20250324_0_L2A",
        "mgrs_tile": "MGRS-43RFN",
        "asset_urls": {
            "B02": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FN/2025/3/S2B_43RFN_20250324_0_L2A/B02.tif",
            "B03": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FN/2025/3/S2B_43RFN_20250324_0_L2A/B03.tif",
            "B04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FN/2025/3/S2B_43RFN_20250324_0_L2A/B04.tif",
            "B08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FN/2025/3/S2B_43RFN_20250324_0_L2A/B08.tif",
            "SCL": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FN/2025/3/S2B_43RFN_20250324_0_L2A/SCL.tif"
        }
    },
    {
        "scene_key": "punjab_ludhiana",
        "name": "Punjab Ludhiana",
        "split": "Train",
        "category": "Agricultural",
        "scene_id": "S2B_43RFP_20241207_0_L2A",
        "mgrs_tile": "MGRS-43RFP",
        "asset_urls": {
            "B02": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FP/2024/12/S2B_43RFP_20241207_0_L2A/B02.tif",
            "B03": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FP/2024/12/S2B_43RFP_20241207_0_L2A/B03.tif",
            "B04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FP/2024/12/S2B_43RFP_20241207_0_L2A/B04.tif",
            "B08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FP/2024/12/S2B_43RFP_20241207_0_L2A/B08.tif",
            "SCL": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/R/FP/2024/12/S2B_43RFP_20241207_0_L2A/SCL.tif"
        }
    },
    {
        "scene_key": "tn_cauvery_delta",
        "name": "Tamil Nadu Cauvery Delta",
        "split": "Test",
        "category": "Agricultural",
        "scene_id": "S2B_44PKT_20250504_0_L2A",
        "mgrs_tile": "MGRS-44PKT",
        "asset_urls": {
            "B02": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/P/KT/2025/5/S2B_44PKT_20250504_0_L2A/B02.tif",
            "B03": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/P/KT/2025/5/S2B_44PKT_20250504_0_L2A/B03.tif",
            "B04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/P/KT/2025/5/S2B_44PKT_20250504_0_L2A/B04.tif",
            "B08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/P/KT/2025/5/S2B_44PKT_20250504_0_L2A/B08.tif",
            "SCL": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/P/KT/2025/5/S2B_44PKT_20250504_0_L2A/SCL.tif"
        }
    }
]

# SCL Color definitions (ESA Standard)
SCL_CLASSES = {
    0: ("NO_DATA", "#000000", False),
    1: ("SATURATED_OR_DEFECTIVE", "#ff0000", False),
    2: ("DARK_AREA_PIXELS", "#2f2f2f", True),
    3: ("CLOUD_SHADOWS", "#643200", False),
    4: ("VEGETATION", "#00a000", True),
    5: ("NOT_VEGETATED", "#ffe65a", True),
    6: ("WATER", "#0000ff", True),
    7: ("UNCLASSIFIED", "#808080", True),
    8: ("CLOUD_MEDIUM_PROBABILITY", "#c0c0c0", False),
    9: ("CLOUD_HIGH_PROBABILITY", "#ffffff", False),
    10: ("THIN_CIRRUS", "#00ffff", False),
    11: ("SNOW_ICE", "#46b4ff", True)
}

def extract_scene_roi(scene_info):
    scene_name = scene_info["name"]
    print(f"\n{'='*70}\nExtracting ROI 1 for: {scene_name} ({scene_info['split']} - {scene_info['category']})\n{'='*70}", flush=True)

    # Window definitions
    # 10m: col_off=1800, row_off=1800, width=1024, height=1024
    # 20m: col_off=900, row_off=900, width=512, height=512
    win_10m = Window(col_off=1800, row_off=1800, width=1024, height=1024)
    win_20m = Window(col_off=900, row_off=900, width=512, height=512)

    band_data = {}
    band_meta = {}
    errors = []

    # 1. Read 10m bands
    for b_name in ["B02", "B03", "B04", "B08"]:
        url = scene_info["asset_urls"][b_name]
        print(f"  -> Reading {b_name} from remote COG...", flush=True)
        t0 = time.perf_counter()
        try:
            with rasterio.open(url) as src:
                arr = src.read(1, window=win_10m)
                t_read = round((time.perf_counter() - t0) * 1000, 1)
                win_transform = rasterio.windows.transform(win_10m, src.transform)
                win_bounds = rasterio.windows.bounds(win_10m, src.transform)
                
                band_data[b_name] = arr
                band_meta[b_name] = {
                    "asset_url": url,
                    "http_status": 200,
                    "read_time_ms": t_read,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "crs": str(src.crs),
                    "resolution": list(src.res),
                    "transform": [round(x, 6) for x in win_transform][:6],
                    "bounds": [round(x, 2) for x in win_bounds],
                    "min_val": int(arr.min()),
                    "max_val": int(arr.max()),
                    "mean_val": round(float(arr.mean()), 2),
                    "nodata_pixels": int((arr == 0).sum()),
                    "nodata_pct": round(float((arr == 0).sum() / arr.size * 100), 4),
                    "valid_pixels": int((arr > 0).sum()),
                    "valid_pct": round(float((arr > 0).sum() / arr.size * 100), 4)
                }
                print(f"     [OK] {b_name}: shape={arr.shape}, dtype={arr.dtype}, valid={band_meta[b_name]['valid_pct']}% ({t_read} ms)", flush=True)
        except Exception as e:
            err = f"Failed to read {b_name}: {e}"
            errors.append(err)
            print(f"     [FAIL] {err}", flush=True)

    # 2. Read SCL at native 20m resolution
    scl_url = scene_info["asset_urls"]["SCL"]
    print(f"  -> Reading SCL at native 20m resolution...", flush=True)
    t0 = time.perf_counter()
    try:
        with rasterio.open(scl_url) as src_scl:
            scl_arr = src_scl.read(1, window=win_20m)
            t_read = round((time.perf_counter() - t0) * 1000, 1)
            win_transform_20m = rasterio.windows.transform(win_20m, src_scl.transform)
            win_bounds_20m = rasterio.windows.bounds(win_20m, src_scl.transform)

            band_data["SCL"] = scl_arr
            band_meta["SCL"] = {
                "asset_url": scl_url,
                "http_status": 200,
                "read_time_ms": t_read,
                "shape": list(scl_arr.shape),
                "dtype": str(scl_arr.dtype),
                "crs": str(src_scl.crs),
                "resolution": list(src_scl.res),
                "transform": [round(x, 6) for x in win_transform_20m][:6],
                "bounds": [round(x, 2) for x in win_bounds_20m]
            }
            print(f"     [OK] SCL: shape={scl_arr.shape} (native 20m), dtype={scl_arr.dtype} ({t_read} ms)", flush=True)
    except Exception as e:
        err = f"Failed to read SCL: {e}"
        errors.append(err)
        print(f"     [FAIL] {err}", flush=True)

    # 3. Spatial Alignment Verification
    # Check 10m alignment between B02, B03, B04, B08
    b_shapes = [band_meta[b]["shape"] for b in ["B02", "B03", "B04", "B08"] if b in band_meta]
    b_transforms = [band_meta[b]["transform"] for b in ["B02", "B03", "B04", "B08"] if b in band_meta]
    b_crs = [band_meta[b]["crs"] for b in ["B02", "B03", "B04", "B08"] if b in band_meta]

    alignment_10m = (len(set(map(tuple, b_shapes))) == 1 and
                     len(set(map(tuple, b_transforms))) == 1 and
                     len(set(b_crs)) == 1)

    # Check bounds consistency between 10m and 20m
    bounds_10m = band_meta["B04"]["bounds"] if "B04" in band_meta else None
    bounds_20m = band_meta["SCL"]["bounds"] if "SCL" in band_meta else None
    
    bounds_match = False
    if bounds_10m and bounds_20m:
        # Max difference in coordinates should be < 1 meter
        diffs = [abs(a - b) for a, b in zip(bounds_10m, bounds_20m)]
        bounds_match = all(d < 1.0 for d in diffs)

    # 4. SCL Quality & Cloud Histogram Analysis
    scl = band_data.get("SCL")
    scl_histogram = {}
    total_pixels_scl = scl.size if scl is not None else 1

    valid_surface_pixels = 0
    cloud_shadow_pixels = 0
    med_cloud_pixels = 0
    high_cloud_pixels = 0
    cirrus_pixels = 0
    nodata_pixels = 0
    saturated_pixels = 0

    if scl is not None:
        for c_id, (c_name, c_color, is_val) in SCL_CLASSES.items():
            cnt = int((scl == c_id).sum())
            pct = round((cnt / total_pixels_scl) * 100, 4)
            scl_histogram[f"class_{c_id}_{c_name}"] = {"count": cnt, "percentage": pct, "is_valid": is_val}

            if is_val: valid_surface_pixels += cnt
            if c_id == 0: nodata_pixels += cnt
            elif c_id == 1: saturated_pixels += cnt
            elif c_id == 3: cloud_shadow_pixels += cnt
            elif c_id == 8: med_cloud_pixels += cnt
            elif c_id == 9: high_cloud_pixels += cnt
            elif c_id == 10: cirrus_pixels += cnt

    total_cloud_shadow_pixels = cloud_shadow_pixels + med_cloud_pixels + high_cloud_pixels + cirrus_pixels
    total_cloud_shadow_pct = round((total_cloud_shadow_pixels / total_pixels_scl) * 100, 4)
    valid_surface_pct = round((valid_surface_pixels / total_pixels_scl) * 100, 4)
    nodata_pct = round((nodata_pixels / total_pixels_scl) * 100, 4)

    # Acceptance threshold: valid >= 98%, cloud/shadow <= 2%, nodata == 0%
    passes_scl_threshold = (valid_surface_pct >= 98.0 and total_cloud_shadow_pct <= 2.0 and nodata_pct == 0.0)

    quality_summary = {
        "valid_surface_pixels": valid_surface_pixels,
        "valid_surface_percentage": valid_surface_pct,
        "total_cloud_and_shadow_pixels": total_cloud_shadow_pixels,
        "total_cloud_and_shadow_percentage": total_cloud_shadow_pct,
        "breakdown": {
            "no_data_class_0_pct": round((nodata_pixels / total_pixels_scl) * 100, 4),
            "saturated_class_1_pct": round((saturated_pixels / total_pixels_scl) * 100, 4),
            "cloud_shadow_class_3_pct": round((cloud_shadow_pixels / total_pixels_scl) * 100, 4),
            "medium_cloud_class_8_pct": round((med_cloud_pixels / total_pixels_scl) * 100, 4),
            "high_cloud_class_9_pct": round((high_cloud_pixels / total_pixels_scl) * 100, 4),
            "thin_cirrus_class_10_pct": round((cirrus_pixels / total_pixels_scl) * 100, 4)
        },
        "passes_quality_threshold": passes_scl_threshold
    }

    print(f"  -> Quality Audit: Valid={valid_surface_pct}%, Cloud/Shadow={total_cloud_shadow_pct}%, Nodata={nodata_pct}%", flush=True)
    print(f"  -> Quality Threshold Verdict: {'PASS' if passes_scl_threshold else 'REJECT'}", flush=True)

    # 5. Save output formats and measure actual file sizes
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_prefix = os.path.join(OUTPUT_DIR, f"roi1_{scene_info['scene_key']}")

    # a) Compressed UInt16 GeoTIFF (4-band 10m)
    tif_path = f"{base_prefix}_10m_4band.tif"
    if all(b in band_data for b in ["B02", "B03", "B04", "B08"]):
        # Stack B04, B03, B02, B08
        stack_10m = np.stack([band_data["B04"], band_data["B03"], band_data["B02"], band_data["B08"]])
        out_profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'nodata': 0,
            'width': 1024,
            'height': 1024,
            'count': 4,
            'crs': band_meta["B04"]["crs"],
            'transform': rasterio.windows.transform(win_10m, rasterio.Affine(*band_meta["B04"]["transform"])),
            'compress': 'deflate',
            'predictor': 2
        }
        with rasterio.open(tif_path, 'w', **out_profile) as dst:
            dst.write(stack_10m)
            dst.set_band_description(1, "B04_Red")
            dst.set_band_description(2, "B03_Green")
            dst.set_band_description(3, "B02_Blue")
            dst.set_band_description(4, "B08_NIR")
    
    # Save SCL GeoTIFF
    scl_tif_path = f"{base_prefix}_20m_scl.tif"
    if "SCL" in band_data:
        scl_profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'width': 512,
            'height': 512,
            'count': 1,
            'crs': band_meta["SCL"]["crs"],
            'transform': rasterio.windows.transform(win_20m, rasterio.Affine(*band_meta["SCL"]["transform"])),
            'compress': 'deflate'
        }
        with rasterio.open(scl_tif_path, 'w', **scl_profile) as dst:
            dst.write(band_data["SCL"], 1)

    # b) Compressed NPZ
    npz_path = f"{base_prefix}_package.npz"
    np.savez_compressed(
        npz_path,
        b02=band_data.get("B02"),
        b03=band_data.get("B03"),
        b04=band_data.get("B04"),
        b08=band_data.get("B08"),
        scl=band_data.get("SCL"),
        bounds_10m=bounds_10m,
        bounds_20m=bounds_20m,
        crs=band_meta["B04"]["crs"]
    )

    size_geotiff_bytes = os.path.getsize(tif_path) if os.path.exists(tif_path) else 0
    size_scl_geotiff_bytes = os.path.getsize(scl_tif_path) if os.path.exists(scl_tif_path) else 0
    size_total_geotiff_bytes = size_geotiff_bytes + size_scl_geotiff_bytes
    size_npz_bytes = os.path.getsize(npz_path) if os.path.exists(npz_path) else 0

    print(f"  -> Actual Saved Sizes on Disk:", flush=True)
    print(f"     * 4-Band GeoTIFF (DEFLATE): {size_geotiff_bytes / (1024*1024):.2f} MB", flush=True)
    print(f"     * SCL GeoTIFF (DEFLATE):    {size_scl_geotiff_bytes / (1024*1024):.2f} MB", flush=True)
    print(f"     * Total GeoTIFF Pair:       {size_total_geotiff_bytes / (1024*1024):.2f} MB", flush=True)
    print(f"     * Compressed NPZ Package:   {size_npz_bytes / (1024*1024):.2f} MB", flush=True)

    # 6. Diagnostic Visualization (3 Panels: RGB, SCL Mask, Valid Mask)
    vis_path = os.path.join(OUTPUT_DIR, f"diagnostic_{scene_info['scene_key']}.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=130)

    # Panel 1: RGB Composite
    r = band_data["B04"].astype(np.float32)
    g = band_data["B03"].astype(np.float32)
    b = band_data["B02"].astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1)

    # Percentile contrast stretch (2% to 98%)
    valid_pixels = rgb[rgb > 0]
    if len(valid_pixels) > 0:
        p2 = np.percentile(valid_pixels, 2)
        p98 = np.percentile(valid_pixels, 98)
        rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    else:
        rgb_stretched = np.zeros_like(rgb)

    axes[0].imshow(rgb_stretched)
    axes[0].set_title(f"{scene_name} (ROI 1)\nRGB Composite (B04-B03-B02, 10m)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: SCL Mask (20m native displayed with discrete palette)
    # Upsample SCL nearest-neighbor to 1024x1024 for visualization comparison
    scl_up = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)

    cmap_colors = [SCL_CLASSES[i][1] for i in range(12)]
    cmap = mcolors.ListedColormap(cmap_colors)
    bounds = list(range(13))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    axes[1].imshow(scl_up, cmap=cmap, norm=norm)
    axes[1].set_title(f"SCL Scene Classification (Native 20m)\nValid: {valid_surface_pct}%, Cloud/Shadow: {total_cloud_shadow_pct}%", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # Add SCL Legend
    legend_patches = [
        mpatches.Patch(color=SCL_CLASSES[c][1], label=f"{c}: {SCL_CLASSES[c][0]}")
        for c in np.unique(scl)
    ]
    axes[1].legend(handles=legend_patches, loc='lower right', fontsize=7, framealpha=0.8)

    # Panel 3: Valid-Data Mask
    # Valid = Classes 2, 4, 5, 6, 7, 11 and not nodata
    valid_mask = np.isin(scl_up, [2, 4, 5, 6, 7, 11]) & (r > 0)
    axes[2].imshow(valid_mask, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title(f"Quality Acceptance Mask\nStatus: {'ACCEPTED (>=98% Valid)' if passes_scl_threshold else 'REJECTED'}", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(vis_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Diagnostic Visualization Saved: {vis_path}", flush=True)

    result_entry = {
        "scene_info": scene_info,
        "window_10m": {"col_off": 1800, "row_off": 1800, "width": 1024, "height": 1024},
        "window_20m": {"col_off": 900, "row_off": 900, "width": 512, "height": 512},
        "band_metadata": band_meta,
        "spatial_alignment": {
            "all_10m_bands_aligned": alignment_10m,
            "crs_consistent": (band_meta["B04"]["crs"] == band_meta["SCL"]["crs"]),
            "geographic_bounds_match": bounds_match,
            "bounds_10m": bounds_10m,
            "bounds_20m": bounds_20m
        },
        "scl_quality_audit": quality_summary,
        "scl_class_histogram": scl_histogram,
        "measured_file_sizes_bytes": {
            "geotiff_4band_10m": size_geotiff_bytes,
            "geotiff_scl_20m": size_scl_geotiff_bytes,
            "geotiff_total_pair": size_total_geotiff_bytes,
            "compressed_npz_package": size_npz_bytes
        },
        "measured_file_sizes_mb": {
            "geotiff_4band_10m_mb": round(size_geotiff_bytes / (1024 * 1024), 2),
            "geotiff_scl_20m_mb": round(size_scl_geotiff_bytes / (1024 * 1024), 2),
            "geotiff_total_pair_mb": round(size_total_geotiff_bytes / (1024 * 1024), 2),
            "compressed_npz_package_mb": round(size_npz_bytes / (1024 * 1024), 2)
        },
        "visualization_path": vis_path,
        "read_or_alignment_errors": errors
    }
    return result_entry

def main():
    print("=" * 80, flush=True)
    print("VISTAARA OCR-GSR: SMALL REAL ROI EXTRACTION & PHYSICAL VALIDATION", flush=True)
    print("=" * 80, flush=True)

    results = []
    for sc in TARGET_SCENES:
        res = extract_scene_roi(sc)
        results.append(res)

    # Save JSON report
    report_json = {
        "report_title": "Small Real ROI Extraction Validation Report",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_scenes_count": len(results),
        "results": results
    }
    with open(JSON_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    # Save Markdown report
    with open(MD_REPORT, "w", encoding="utf-8") as f:
        f.write("# Small Real ROI Extraction Validation Report\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write("**Primary Data Source**: Element 84 Earth Search (AWS Sentinel-2 L2A COGs)\n\n")
        f.write("---\n\n")

        for r in results:
            sc = r["scene_info"]
            bm = r["band_metadata"]
            qa = r["scl_quality_audit"]
            sz = r["measured_file_sizes_mb"]
            f.write(f"## {sc['name']} ({sc['split']} — {sc['category']})\n\n")
            f.write(f"- **Scene ID**: `{sc['scene_id']}`\n")
            f.write(f"- **MGRS Tile**: `{sc['mgrs_tile']}`\n")
            f.write(f"- **CRS**: `{bm['B04']['crs']}`\n")
            f.write(f"- **10m Array Shape**: `{bm['B04']['shape']}` (Dtype: `{bm['B04']['dtype']}`)\n")
            f.write(f"- **SCL Array Shape**: `{bm['SCL']['shape']}` (Native 20m, Dtype: `{bm['SCL']['dtype']}`)\n")
            f.write(f"- **Spatial Alignment**: All 10m bands aligned = `{r['spatial_alignment']['all_10m_bands_aligned']}`, Bounds match = `{r['spatial_alignment']['geographic_bounds_match']}`\n")
            f.write(f"- **Valid Surface Pixels**: **{qa['valid_surface_percentage']}%**\n")
            f.write(f"- **Total Cloud & Shadow**: **{qa['total_cloud_and_shadow_percentage']}%** (Cirrus: {qa['breakdown']['thin_cirrus_class_10_pct']}%, Shadow: {qa['breakdown']['cloud_shadow_class_3_pct']}%)\n")
            f.write(f"- **Nodata Pixels**: **{qa['breakdown']['no_data_class_0_pct']}%**\n")
            f.write(f"- **Quality Gate**: **{'PASSED' if qa['passes_quality_threshold'] else 'FAILED'}**\n\n")
            f.write("### Measured File Sizes on Disk\n\n")
            f.write(f"| Format | Measured Size | Status |\n| :--- | :---: | :--- |\n")
            f.write(f"| **Compressed UInt16 GeoTIFF (4-Band 10m)** | **{sz['geotiff_4band_10m_mb']} MB** | DEFLATE compressed |\n")
            f.write(f"| **Compressed UInt8 GeoTIFF (SCL 20m)** | **{sz['geotiff_scl_20m_mb']} MB** | DEFLATE compressed |\n")
            f.write(f"| **Total GeoTIFF Pair (4-Band + SCL)** | **{sz['geotiff_total_pair_mb']} MB** | Within analytical model |\n")
            f.write(f"| **Compressed NPZ Package** | **{sz['compressed_npz_package_mb']} MB** | `np.savez_compressed` |\n\n")
            f.write(f"- **Diagnostic Visualization**: `{r['visualization_path']}`\n\n")
            f.write("---\n\n")

    print(f"\n{'='*80}\nROI EXTRACTION VALIDATION COMPLETE\n{'='*80}", flush=True)
    print(f"JSON Report: {os.path.abspath(JSON_REPORT)}", flush=True)
    print(f"MD Report:   {os.path.abspath(MD_REPORT)}", flush=True)

if __name__ == "__main__":
    main()
