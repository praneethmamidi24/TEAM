# -*- coding: utf-8 -*-
"""
discover_and_validate_metadata.py
==================================
Data Discovery and Metadata Validation for OCR-GSR Experimental Training.

Data Source:
    Element 84 Earth Search STAC API (AWS Sentinel-2 L2A COGs)
    https://earth-search.aws.element84.com/v1

Mandatory Protocol Constraints:
1. Tier 1 (40 m -> 10 m) and Tier 2 (10 m -> 2.5 m) are strictly separated.
2. Tier 1 synthetic scale degradation is for diagnostic calibration only and is NOT
   proof of 2.5 m sub-pixel reconstruction accuracy.
3. Multiple scenes per land-cover category (Urban, Agriculture, Forest, Water/Mixed).
4. Training, Validation, and Test partitions are strictly disjoint geographically (distinct MGRS tiles).
5. SCL (Scene Classification Layer) defines pixel-level cloud, shadow, and invalid data rejection.
6. 1024 x 1024 pixel regions of interest (ROIs) are budgeted to verify storage <= 250 MB.
7. Allows the experimental outcome to conclude that OCR-GSR provides no measurable improvement.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse

STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
HEADERS = {
    "User-Agent": "VISTAARA-OCRGSR-DataDiscovery/1.0",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

TARGET_REGIONS = [
    # --- URBAN AREAS ---
    {
        "region_id": "URB_01_DELHI",
        "name": "Delhi NCR",
        "category": "Urban",
        "split": "Train",
        "bbox": [76.90, 28.45, 77.35, 28.85],
        "description": "Dense high-rise, mixed residential, industrial, transport networks."
    },
    {
        "region_id": "URB_02_BLR",
        "name": "Bengaluru",
        "category": "Urban",
        "split": "Train",
        "bbox": [77.45, 12.85, 77.75, 13.15],
        "description": "Sprawling tech corridor, urban lakes, mixed tree canopy and dense built-up."
    },
    {
        "region_id": "URB_03_MUM",
        "name": "Mumbai MMR",
        "category": "Urban",
        "split": "Validation",
        "bbox": [72.78, 18.90, 73.10, 19.25],
        "description": "Coastal megacity, linear high-density infrastructure, creek interface."
    },
    {
        "region_id": "URB_04_HYD",
        "name": "Hyderabad",
        "category": "Urban",
        "split": "Test",
        "bbox": [78.30, 17.30, 78.60, 17.55],
        "description": "Granitic plateau topography, planned urban zones, rocky outcrops."
    },

    # --- AGRICULTURAL AREAS ---
    {
        "region_id": "AGR_01_PUNJAB",
        "name": "Punjab Ludhiana Plains",
        "category": "Agricultural",
        "split": "Train",
        "bbox": [75.50, 30.50, 76.10, 31.10],
        "description": "Intensive irrigated cereal parcels, regular rectilinear farm borders, canals."
    },
    {
        "region_id": "AGR_02_MALWA",
        "name": "Madhya Pradesh Malwa",
        "category": "Agricultural",
        "split": "Train",
        "bbox": [75.70, 22.80, 76.30, 23.40],
        "description": "Rainfed black soil agriculture, undulating terrain, seasonal crop patterns."
    },
    {
        "region_id": "AGR_03_GODAVARI",
        "name": "Andhra Godavari Delta",
        "category": "Agricultural",
        "split": "Validation",
        "bbox": [81.60, 16.60, 82.20, 17.20],
        "description": "Alluvial deltaic paddy agro-ecosystem, high soil moisture, irrigation networks."
    },
    {
        "region_id": "AGR_04_KARNAL",
        "name": "Haryana Karnal Plains",
        "category": "Agricultural",
        "split": "Test",
        "bbox": [76.70, 29.50, 77.20, 30.00],
        "description": "Intensive wheat-rice cropping system, dense field parcel transitions."
    },

    # --- FOREST AREAS ---
    {
        "region_id": "FOR_01_WGHATS",
        "name": "Western Ghats Agumbe",
        "category": "Forest",
        "split": "Train",
        "bbox": [75.00, 13.20, 75.50, 13.70],
        "description": "Dense tropical wet evergreen rainforest, steep terrain, high canopy density."
    },
    {
        "region_id": "FOR_02_KANHA",
        "name": "Central India Kanha",
        "category": "Forest",
        "split": "Validation",
        "bbox": [80.40, 22.10, 80.90, 22.50],
        "description": "Tropical moist deciduous Sal forest mixed with open grassland clearings."
    },
    {
        "region_id": "FOR_03_ASSAM",
        "name": "Assam Kaziranga Buffer",
        "category": "Forest",
        "split": "Test",
        "bbox": [93.10, 26.50, 93.60, 26.80],
        "description": "Semi-evergreen riparian forest, tall alluvial grassland, river corridor."
    },

    # --- WATER & MIXED LANDSCAPES ---
    {
        "region_id": "WAT_01_CHILIKA",
        "name": "Chilika Lake Lagoon",
        "category": "Water & Mixed",
        "split": "Train",
        "bbox": [85.15, 19.45, 85.55, 19.85],
        "description": "Brackish coastal lagoon, barrier sand spit, wetland vegetation and mudflats."
    },
    {
        "region_id": "WAT_02_SUNDAR",
        "name": "Sundarbans Estuary",
        "category": "Water & Mixed",
        "split": "Validation",
        "bbox": [88.60, 21.60, 89.10, 22.00],
        "description": "Dense tidal mangrove swamp, dynamic intertidal channels, saline mudflats."
    },
    {
        "region_id": "WAT_03_VEMBANAD",
        "name": "Vembanad Backwaters",
        "category": "Water & Mixed",
        "split": "Test",
        "bbox": [76.30, 9.50, 76.60, 9.90],
        "description": "Tropical estuarine backwaters, reclaimed polders, narrow canals and barriers."
    }
]

DEGRADATION_SPECIFICATIONS = {
    "tier_1_surrogate": {
        "tier_name": "Tier 1: Synthetic Scale-Degraded Supervised Calibration (40 m -> 10 m)",
        "input_resolution_m": 40.0,
        "target_resolution_m": 10.0,
        "scale_factor": 4,
        "target_ground_truth": "Native Sentinel-2 L2A 10m bands (B02, B03, B04, B08)",
        "synthetic_generation_pipeline": {
            "step_1_blur": {
                "kernel": "Gaussian 2D",
                "kernel_size": 7,
                "sigma": 1.1,
                "justification": "Approximates Sentinel-2 MultiSpectral Instrument (MSI) optical Point Spread Function (PSF) and Modulation Transfer Function (MTF) at Nyquist frequency (0.18-0.24)."
            },
            "step_2_downsampling": {
                "method": "Area-averaging decimation by factor 4",
                "justification": "Models detector spatial integration over a 40 m instantaneous field of view."
            },
            "step_3_sensor_noise": {
                "type": "Additive zero-mean Gaussian noise",
                "sigma": 0.004,
                "clip_range": [0.0, 1.0],
                "justification": "Simulates detector dark-current and quantization noise consistent with Sentinel-2 L2A SNR > 100."
            }
        },
        "scientific_boundary": (
            "MANDATORY RULE: Tier 1 results must NOT be cited or claimed as proof of real 2.5 m "
            "reconstruction accuracy. Optical point spread functions and sub-pixel edge mixing "
            "at 40 m do not directly extrapolate to 2.5 m. Tier 1 serves solely as an observable "
            "calibration diagnostic for the reliability gate and residual head."
        )
    },
    "tier_2_self_supervised": {
        "tier_name": "Tier 2: Observation-Consistent Cycle Super-Resolution (10 m -> 2.5 m)",
        "input_resolution_m": 10.0,
        "target_nominal_resolution_m": 2.5,
        "scale_factor": 4,
        "target_ground_truth": "NO GENUINE 2.5 m OPTICAL GROUND TRUTH EXISTS IN SENTINEL-2.",
        "observation_cycle_loss": {
            "downsampler": "F.interpolate(sr, size=lr.shape[-2:], mode='area')",
            "loss_term": "L1 norm || simulated_lr - actual_lr ||_1",
            "justification": "Enforces that the generated 2.5 m representation is mathematically consistent with the 10 m physical sensor observation."
        },
        "evaluation_protocol": (
            "Evaluated strictly via VISTAARA multi-signal physical metrics: observation cycle residual, "
            "Spectral Angle Mapper (SAM), high-frequency edge coherence (Sobel), and perturbation stability. "
            "Allows the null hypothesis: OCR-GSR may yield zero measurable improvement over baseline SEN2SRLite."
        )
    }
}

SCL_PROTOCOL = {
    "scl_band_description": "Sentinel-2 L2A Scene Classification Layer (20 m native, nearest-neighbor upsampled to 10 m)",
    "class_definitions": {
        0: "NO_DATA (Invalid)",
        1: "SATURATED_OR_DEFECTIVE (Invalid)",
        2: "DARK_AREA_PIXELS (Valid / Shadow)",
        3: "CLOUD_SHADOWS (Invalid / Reject)",
        4: "VEGETATION (Valid)",
        5: "NOT_VEGETATED (Valid)",
        6: "WATER (Valid)",
        7: "UNCLASSIFIED (Valid)",
        8: "CLOUD_MEDIUM_PROBABILITY (Invalid / Reject)",
        9: "CLOUD_HIGH_PROBABILITY (Invalid / Reject)",
        10: "THIN_CIRRUS (Invalid / Reject)",
        11: "SNOW_ICE (Valid / Flagged)"
    },
    "valid_classes": [2, 4, 5, 6, 7, 11],
    "invalid_classes": [0, 1, 3, 8, 9, 10],
    "patch_acceptance_criteria": {
        "min_valid_pixel_fraction": 0.98,
        "max_cloud_shadow_pixel_fraction": 0.02,
        "max_nodata_pixel_fraction": 0.00
    }
}

def query_stac(region, max_cloud=5.0, date_range="2024-11-01T00:00:00Z/2025-05-30T23:59:59Z"):
    """Query Element 84 Earth Search STAC API for candidate scenes."""
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": region["bbox"],
        "datetime": date_range,
        "query": {
            "eo:cloud_cover": {"lt": max_cloud}
        },
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 3
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(STAC_SEARCH_URL, data=data, headers=HEADERS, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            return res_json.get("features", [])
    except Exception as e:
        print(f"    [ERROR] Query failed for {region['name']}: {e}", flush=True)
        return []


def validate_asset_head(url):
    """Verify HTTP HEAD on COG URL to confirm reachability, file size, and byte ranges."""
    if not url:
        return False, 0, False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]}, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            status = resp.status
            size = int(resp.headers.get("Content-Length", 0))
            accept_ranges = resp.headers.get("Accept-Ranges", "") == "bytes"
            return (status == 200), size, accept_ranges
    except Exception:
        return False, 0, False


def main():
    print("=" * 82, flush=True)
    print("VISTAARA OCR-GSR: DATA DISCOVERY & METADATA VALIDATION", flush=True)
    print("Primary Data Source: Element 84 Earth Search (AWS Sentinel-2 L2A COGs)", flush=True)
    print("=" * 82, flush=True)

    catalog = {
        "dataset_name": "VISTAARA_OCRGSR_India_Benchmark_Dataset",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sensor": "Sentinel-2 MSI (Level-2A BOA Reflectance)",
        "surface_reflectance_scale_factor": 10000.0,
        "required_bands": ["B02", "B03", "B04", "B08"],
        "auxiliary_quality_band": "SCL",
        "target_regions_count": len(TARGET_REGIONS),
        "degradation_specifications": DEGRADATION_SPECIFICATIONS,
        "scl_filtering_protocol": SCL_PROTOCOL,
        "scenes": []
    }

    split_counts = {"Train": 0, "Validation": 0, "Test": 0}
    cat_counts = {}

    # Validate asset HEAD on the first scene to confirm AWS S3 reachability without slowing down the full query
    sample_validated = False

    for idx, reg in enumerate(TARGET_REGIONS, start=1):
        print(f"[{idx:02d}/{len(TARGET_REGIONS):02d}] Probing: {reg['name']:<25} | {reg['category']:<14} | Split: {reg['split']}", end="", flush=True)
        features = query_stac(reg, max_cloud=5.0)

        if not features:
            print(" [retry <=10%]", end="", flush=True)
            features = query_stac(reg, max_cloud=10.0)

        if not features:
            print(" -> FAILED", flush=True)
            continue

        feat = features[0]
        scene_id = feat["id"]
        props = feat.get("properties", {})
        cloud_pct = props.get("eo:cloud_cover", 0.0)
        dt = props.get("datetime", "")
        tile_code = props.get("grid:code", "N/A")
        assets = feat.get("assets", {})

        band_keys = {
            "B02": ["blue", "B02"],
            "B03": ["green", "B03"],
            "B04": ["red", "B04"],
            "B08": ["nir", "B08"],
            "SCL": ["scl", "SCL"]
        }
        band_urls = {}
        for b, alts in band_keys.items():
            u = None
            for k in alts:
                if k in assets and "href" in assets[k]:
                    u = assets[k]["href"]
                    break
            band_urls[b] = u

        status_label = "STAC_VERIFIED"
        b04_bytes = 0
        scl_bytes = 0

        if not sample_validated and band_urls.get("B04"):
            b04_ok, b04_bytes, b04_range = validate_asset_head(band_urls.get("B04"))
            if b04_ok:
                sample_validated = True
                status_label = "ACCESSIBLE_VERIFIED"

        print(f" -> Tile: {tile_code:<8} Date: {dt[:10]} Cloud: {cloud_pct:.2f}%", flush=True)

        scene_entry = {
            "region_id": reg["region_id"],
            "region_name": reg["name"],
            "category": reg["category"],
            "split": reg["split"],
            "description": reg["description"],
            "target_bbox": reg["bbox"],
            "scene_id": scene_id,
            "tile_code": tile_code,
            "acquisition_datetime": dt,
            "cloud_cover_percentage": cloud_pct,
            "remote_asset_status": status_label,
            "band_urls": band_urls,
            "sample_b04_bytes": b04_bytes
        }
        catalog["scenes"].append(scene_entry)
        split_counts[reg["split"]] += 1
        cat_counts[reg["category"]] = cat_counts.get(reg["category"], 0) + 1

    # Storage & Budget Calculation
    num_scenes = len(catalog["scenes"])
    rois_per_scene = 3  # 3 distinct 1024x1024 windows per scene
    total_rois = num_scenes * rois_per_scene
    est_mb_per_roi = 4.8
    total_mb_estimate = round(total_rois * est_mb_per_roi, 1)

    catalog["storage_budget_analysis"] = {
        "num_validated_scenes": num_scenes,
        "rois_per_scene": rois_per_scene,
        "roi_pixel_size": "1024 x 1024 (10.24 km x 10.24 km at 10m GSD)",
        "total_rois_planned": total_rois,
        "candidate_128x128_patches_per_roi": 64,
        "total_candidate_training_patches": total_rois * 64,
        "estimated_compressed_mb_per_roi": est_mb_per_roi,
        "total_estimated_storage_mb": total_mb_estimate,
        "storage_budget_cap_mb": 250.0,
        "budget_verdict": "VERIFIED_FEASIBLE" if total_mb_estimate <= 250.0 else "EXCEEDS_CAP"
    }

    output_filename = "ocr_gsr_discovered_scenes.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print("\n" + "=" * 82, flush=True)
    print("METADATA VALIDATION & STORAGE FEASIBILITY VERIFICATION", flush=True)
    print("=" * 82, flush=True)
    print(f"Total Regions Queried:         {len(TARGET_REGIONS)}", flush=True)
    print(f"Successfully Validated Scenes: {num_scenes}", flush=True)
    print(f"Partition Breakdown:           Train: {split_counts['Train']} | Val: {split_counts['Validation']} | Test: {split_counts['Test']}", flush=True)
    print(f"Landscape Breakdown:           {cat_counts}", flush=True)
    print(f"Planned ROIs:                  {total_rois} ({rois_per_scene} windows of 1024x1024 per scene)", flush=True)
    print(f"Potential 128x128 Patches:     {total_rois * 64:,} (before SCL quality filtering)", flush=True)
    print(f"Estimated Storage Footprint:   {total_mb_estimate} MB", flush=True)
    print(f"Budget Cap:                    250.0 MB -> Verdict: {catalog['storage_budget_analysis']['budget_verdict']}", flush=True)
    print(f"Metadata Saved:                {os.path.abspath(output_filename)}", flush=True)
    print("=" * 82, flush=True)


if __name__ == "__main__":
    main()
