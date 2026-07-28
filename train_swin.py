"""
Training script for swin_base_patch4_window7_224.ms_in22k_ft_in1k +
ProgressiveClassifier (combo #3 in the deployment PRD — introduces Dirichlet
uncertainty quantification; this backbone is not used by any other combo;
feeds the explain-path uncertainty readout in §8.4).

Usage:
    python train_swin.py
    python train_swin.py --epochs 20 --nfolds 3     # quick smoke run

Outputs (into saved_models/):
    swin_best.pth
    swin_class_names.txt

Tracking: set MLFLOW_TRACKING_URI to log to MLflow; leave unset to log to
saved_models/vit_evidential_metrics.jsonl instead (module/training/tracking.py).
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
    combo_id="swin_progressive",
    display_name="Swin-Base + Progressive",
    model_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    classifier_type="progressive",
    weights_filename="swin_best.pth",
    class_names_filename="swin_class_names.txt",
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
