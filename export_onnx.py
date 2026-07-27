"""
Run after generate_manifest.py. For every combo listed in manifest.json,
loads the trained PyTorch checkpoint, exports it to ONNX (fp32), quantizes
to INT8, and verifies numerical parity between the PyTorch and fp32-ONNX
outputs (a correctness check on the export itself).

This does NOT decide fp32 vs int8 for serving, nor CPU vs OpenVINO — that's
benchmark_runtimes.py's job, since it requires real recall numbers on the
held-out test set, not just "did the export succeed".

Usage:
    python export_onnx.py
    python export_onnx.py --manifest saved_models/manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from module.classifiers import get_classifier
from module.config import DEVICE
from module.training import export_to_onnx, extract_exportable_module, robust_quantize, verify_onnx_parity

SAVE_DIR = ROOT / "saved_models"
PARITY_TOLERANCE = 1e-3  # generous: fp32 ONNX vs PyTorch should agree far tighter than this in practice


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default=str(SAVE_DIR / "manifest.json"))
    p.add_argument("--save_dir", type=str, default=str(SAVE_DIR))
    return p.parse_args()


def export_one(combo: dict, save_dir: Path) -> bool:
    combo_id = combo["combo_id"]
    print(f"\n{'=' * 70}\n{combo['display_name']} ({combo_id})\n{'=' * 70}")

    weights_path = save_dir / combo["pytorch_weights_file"]
    if not weights_path.exists():
        print(f"  SKIPPED: checkpoint not found at {weights_path}")
        return False

    ClassifierClass = get_classifier(combo["classifier_type"])
    clf = ClassifierClass(
        model_name=combo["model_name"], num_classes=len(combo["class_names"]), device=DEVICE
    )
    clf.load(str(weights_path))
    module = extract_exportable_module(clf)

    fp32_path = save_dir / combo["onnx_fp32_file"]
    int8_path = save_dir / combo["onnx_int8_file"]
    img_size = combo["img_size"]

    print(f"  Exporting fp32 ONNX -> {fp32_path.name}")
    export_to_onnx(module, img_size, fp32_path)

    max_diff = verify_onnx_parity(module, fp32_path, img_size)
    status = "OK" if max_diff < PARITY_TOLERANCE else "WARNING — EXCEEDS TOLERANCE"
    print(f"  Parity check: max|pytorch - onnx| = {max_diff:.2e}  ({status})")
    if max_diff >= PARITY_TOLERANCE:
        print(
            f"  Refusing to proceed to quantization for {combo_id}: fp32 export itself "
            f"doesn't match the PyTorch model closely enough to trust. Inspect the graph "
            f"(e.g. `python -c \"import onnx; onnx.load('{fp32_path}')\"`) before retrying."
        )
        return False

    print(f"  Quantizing to INT8 -> {int8_path.name}")
    strategy = robust_quantize(fp32_path, int8_path)
    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6
    print(f"  Done via '{strategy}' path: {fp32_mb:.1f}MB -> {int8_mb:.1f}MB ({100 * int8_mb / fp32_mb:.0f}%)")
    return True


def main():
    args = _parse_args()
    save_dir = Path(args.save_dir)
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run generate_manifest.py first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    results = {combo["combo_id"]: export_one(combo, save_dir) for combo in manifest["combos"]}

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for combo_id, ok in results.items():
        print(f"  {combo_id}: {'exported' if ok else 'SKIPPED/FAILED'}")
    if not all(results.values()):
        print(
            "\nSome combos were not exported (see above). This is fine if you're still "
            "training the rest — re-run this script once every checkpoint exists. "
            "Next step once all four succeed: benchmark_runtimes.py"
        )


if __name__ == "__main__":
    main()
