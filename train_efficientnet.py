"""
Training script for tf_efficientnet_b4.ns_jft_in1k + ProgressiveClassifier
(combo #2 in the deployment PRD — second-best empirical result from the
research phase, flagship CNN counterpart to the swin combo).

Depends on the §9 fix in module/classifiers/progressive_classifier.py
(get_efficientnet_groups now has the same coverage-check safeguard
get_swin_groups has) — run this only against that patched file.

Usage:
    python train_efficientnet.py
    python train_efficientnet.py --epochs 20 --nfolds 3     # quick smoke run
    python train_efficientnet.py --data_dir /path/to/images

Outputs (into saved_models/):
    effnet_progressive_best.pth
    effnet_class_names.txt

Tracking: set MLFLOW_TRACKING_URI to log to MLflow; leave unset to log to
saved_models/effnet_progressive_metrics.jsonl instead (module/training/tracking.py).
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
    combo_id="effnet_progressive",
    display_name="EfficientNet-B4 + Progressive",
    model_name="tf_efficientnet_b4.ns_jft_in1k",
    classifier_type="progressive",
    weights_filename="effnet_progressive_best.pth",
    class_names_filename="effnet_class_names.txt",
)
SAVE_DIR = ROOT / "saved_models"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--nfolds", type=int, default=NFOLDS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
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
        patience=PATIENCE,
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
