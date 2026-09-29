"""Smoke-test and inference ablation harness for OCR-GSR.
This does not claim trained performance; it validates tensor flow and diagnostics.
"""
import argparse, json, time
from pathlib import Path
import torch
from ocr_gsr_model import OCRGSR
from ocr_gsr_losses import ocr_gsr_loss


def run(h=128, w=128, bands=4):
    torch.manual_seed(7)
    lr = torch.rand(1, bands, h, w)
    sr = torch.rand(1, bands, h * 4, w * 4)
    model = OCRGSR(bands=bands).eval()
    start = time.perf_counter()
    with torch.no_grad():
        out = model(lr, sr)
        losses = ocr_gsr_loss(out, lr)
    elapsed = time.perf_counter() - start
    result = {
        "status": "inference_smoke_test_only",
        "runtime_seconds": elapsed,
        "shapes": {k: list(v.shape) for k, v in out.items() if torch.is_tensor(v)},
        "reliability_min": float(out["reliability"].min()),
        "reliability_max": float(out["reliability"].max()),
        "residual_mean_abs": float(out["residual"].abs().mean()),
        "cycle_loss": float(losses["cycle"]),
        "finite": all(bool(torch.isfinite(v).all()) for v in out.values() if torch.is_tensor(v)),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ocr_gsr_results/smoke_test.json")
    args = parser.parse_args()
    result = run()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
