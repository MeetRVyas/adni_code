"""
Run this after all four combos have been trained (train_swin.py,
train_efficientnet.py, train_vit_evidential.py, train_resnext_baseline.py).

Re-evaluates each saved checkpoint on its held-out test split and writes:
    saved_models/manifest.json
    saved_models/results.json

Upload both files, alongside the four *_best.pth weight files, to your HF
Hub artifact repo (see PROJECT_NOTES.md) — that's what the serving app
downloads at startup. Run export_onnx.py + benchmark_runtimes.py afterward
to also produce/select the ONNX files this manifest already references.

Usage:
    python generate_manifest.py
    python generate_manifest.py --data_dir /path/to/images
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from module.config import (
    DATA_DIR, DEVICE, BATCH_SIZE, TEST_SPLIT,
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS,
)
from module.training import ComboConfig, evaluate_combo, build_manifest_and_results, write_json

# combo #1 was trained by the original train_swin.py, which predates the
# ComboConfig abstraction — declared here (not imported) purely so this
# script can treat all four combos uniformly.
SWIN_COMBO = ComboConfig(
    combo_id="swin_progressive",
    display_name="Swin-B + Progressive",
    model_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    classifier_type="progressive",
    weights_filename="swin_progressive_best.pth",
    class_names_filename="swin_class_names.txt",
)

# The other three: import the exact ComboConfig each training script already
# defines, rather than redeclaring (and risking drift from) the same metadata
# a second time here.
from train_efficientnet import COMBO as EFFNET_COMBO
from train_vit_evidential import COMBO as VIT_COMBO
from train_resnext_baseline import COMBO as RESNEXT_COMBO

ALL_COMBOS = [SWIN_COMBO, EFFNET_COMBO, VIT_COMBO, RESNEXT_COMBO]
SAVE_DIR = ROOT / "saved_models"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--save_dir", type=str, default=str(SAVE_DIR))
    return p.parse_args()


def main():
    args = _parse_args()
    save_dir = Path(args.save_dir)

    eval_results = []
    failures = []
    for combo in ALL_COMBOS:
        print(f"\n{'=' * 70}\nEvaluating {combo.display_name} ({combo.combo_id})\n{'=' * 70}")
        try:
            result = evaluate_combo(
                combo,
                data_dir=args.data_dir,
                save_dir=save_dir,
                device=DEVICE,
                test_split=TEST_SPLIT,
                batch_size=args.batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS,
            )
            eval_results.append(result)
            m = result.metrics
            print(
                f"  accuracy={m['accuracy']:.2f}%  recall={m['recall']:.4f}  "
                f"precision={m['precision']:.4f}  f1={m['f1']:.4f}"
            )
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            failures.append(combo.combo_id)

    if not eval_results:
        print("\nNo trained checkpoints found — nothing to write. Train at least one combo first.")
        sys.exit(1)

    manifest, results = build_manifest_and_results(eval_results)
    write_json(save_dir / "manifest.json", manifest)
    write_json(save_dir / "results.json", results)

    print(f"\nWrote {save_dir / 'manifest.json'}")
    print(f"Wrote {save_dir / 'results.json'}")
    if failures:
        print(
            f"\nNote: {len(failures)}/{len(ALL_COMBOS)} combo(s) were skipped "
            f"(checkpoint not found) and are NOT in this manifest: {failures}. "
            f"The app's startup check will fail loudly until all four are present "
            f"(PRD NFR — Reliability), so re-run this script once the remaining "
            f"combo(s) finish training."
        )


if __name__ == "__main__":
    main()
