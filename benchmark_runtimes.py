"""
Run after export_onnx.py. For every combo:
  1. Validates INT8 recall parity against fp32 on the real held-out test set
     (PRD risk mitigation — falls back to fp32 if INT8 loses too much recall).
  2. Benchmarks CPUExecutionProvider vs OpenVINOExecutionProvider on whichever
     precision was just selected (PRD §8.3).
  3. Writes both decisions back into manifest.json's
     selected_precision / preferred_execution_provider fields — this is the
     only script that changes those two fields from their safe placeholders.

Usage:
    python benchmark_runtimes.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from module.config import DATA_DIR, DEVICE, TEST_SPLIT, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS
from module.training import build_loader, choose_execution_provider, choose_precision, load_and_split

SAVE_DIR = ROOT / "saved_models"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default=str(SAVE_DIR / "manifest.json"))
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def benchmark_one(combo: dict, save_dir: Path, data_dir: str, batch_size: int) -> dict | None:
    combo_id = combo["combo_id"]
    print(f"\n{'=' * 70}\n{combo['display_name']} ({combo_id})\n{'=' * 70}")

    fp32_path = save_dir / combo["onnx_fp32_file"]
    int8_path = save_dir / combo["onnx_int8_file"]
    if not (fp32_path.exists() and int8_path.exists()):
        print(f"  SKIPPED: run export_onnx.py first (missing {fp32_path.name} or {int8_path.name})")
        return None

    split = load_and_split(data_dir, combo["img_size"], TEST_SPLIT, DEVICE)
    test_loader = build_loader(
        split.full_dataset, split.test_idx, batch_size, False, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS
    )

    print("  Validating INT8 recall parity against fp32 on held-out test set...")
    precision_decision = choose_precision(fp32_path, int8_path, test_loader)
    print(
        f"    fp32 macro recall = {precision_decision.recall_fp32:.4f}   "
        f"int8 macro recall = {precision_decision.recall_int8:.4f}   "
        f"drop = {precision_decision.recall_drop:+.4f}"
    )
    print(f"    -> selected_precision = '{precision_decision.selected_precision}'")

    chosen_path = int8_path if precision_decision.selected_precision == "int8" else fp32_path
    print(f"  Timing execution providers on {chosen_path.name}...")
    latency_decision = choose_execution_provider(chosen_path, combo["img_size"])
    for provider, ms in latency_decision.latency_ms.items():
        print(f"    {provider:28s} {ms:6.2f} ms/inference")
    print(f"    -> preferred_execution_provider = '{latency_decision.preferred_execution_provider}'")

    return {
        "selected_precision": precision_decision.selected_precision,
        "preferred_execution_provider": latency_decision.preferred_execution_provider,
        "_recall_fp32": precision_decision.recall_fp32,
        "_recall_int8": precision_decision.recall_int8,
        "_latency_ms": latency_decision.latency_ms,
    }


def main():
    args = _parse_args()
    manifest_path = Path(args.manifest)
    save_dir = manifest_path.parent

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run generate_manifest.py and export_onnx.py first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    report = {}

    for combo in manifest["combos"]:
        decision = benchmark_one(combo, save_dir, args.data_dir, args.batch_size)
        if decision is not None:
            combo["selected_precision"] = decision["selected_precision"]
            combo["preferred_execution_provider"] = decision["preferred_execution_provider"]
            report[combo["combo_id"]] = decision

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n{'=' * 70}\nUpdated {manifest_path}\n{'=' * 70}")
    for combo_id, d in report.items():
        print(f"  {combo_id:20s} precision={d['selected_precision']:5s} ep={d['preferred_execution_provider']}")

    (save_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nFull report (including timings/recall numbers) -> {save_dir / 'benchmark_report.json'}")


if __name__ == "__main__":
    main()
