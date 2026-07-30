"""
Training script for resnext50_32x4d.a1h_in1k + BaselineClassifier
(combo #4 in the deployment PRD — deliberate control group: a vanilla
fine-tune with no special technique, so the dashboard can substantiate a
"technique X improved recall by Y points over baseline" claim).

To switch to the alternative noted in PRD §6 (resnext50_32x4d +
ProgressiveEvidentialClassifier, if you'd rather not ship a deliberately
weaker baseline in the live demo), change CLASSIFIER_TYPE below to
"progressive_evidential" — check its exact registry key in
module/classifiers/__init__.py first. Note this alternative would also
route through ArchitectureLayerGroups.get_resnet_groups (already verified
in this same patch, see PROJECT_NOTES.md), so no additional pre-training
fix is needed either way.

Usage:
    python train_resnext_baseline.py
    python train_resnext_baseline.py --epochs 20 --nfolds 3     # quick smoke run

Outputs (into saved_models/):
    resnext_baseline_best.pth
    resnext_class_names.txt

Tracking: set MLFLOW_TRACKING_URI to log to MLflow; leave unset to log to
saved_models/resnext_baseline_metrics.jsonl instead (module/training/tracking.py).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from module.config import (
    DATA_DIR, DEVICE, EPOCHS, NFOLDS, BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS,
    TEST_SPLIT, PATIENCE, MIN_DELTA_METRIC, OPTIMIZE_METRIC, LR,
)
from module.utils import Logger
from module.training import ComboConfig, train_combo, build_tracker

COMBO = ComboConfig(
    combo_id="resnext_baseline",
    display_name="ResNeXt50 + Baseline",
    model_name="resnext50_32x4d.a1h_in1k",
    classifier_type="baseline",
    weights_filename="resnext_baseline_best.pth",
    class_names_filename="resnext_class_names.txt",
)
SAVE_DIR = ROOT / "saved_models"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--nfolds", type=int, default=NFOLDS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--patience", type=float, default=PATIENCE)
    p.add_argument("--sam", type=float, default=False)
    p.add_argument("--landscape", type=float, default=False)
    return p.parse_args()


def main():
    args = _parse_args()
    logger = Logger(COMBO.combo_id, file_name=COMBO.combo_id)
    tracker = build_tracker(COMBO.combo_id, SAVE_DIR)

    result = train_combo(
        COMBO,
        data_dir=args.data_dir,
        save_dir=SAVE_DIR,
        device=DEVICE,
        epochs=args.epochs,
        nfolds=args.nfolds,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        min_delta=MIN_DELTA_METRIC,
        use_sam=args.sam,
        landscape=args.landscape,
        primary_metric=OPTIMIZE_METRIC,
        test_split=TEST_SPLIT,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        logger=logger,
        tracker=tracker,
    )

    logger.info(f"\nWeights saved -> {result.weights_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
