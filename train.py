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

    p.add_argument("--combo_id", type=str, required = True)
    p.add_argument("--model_name", type=str, required = True)
    p.add_argument("--classifier_type", type=str, required = True)
    p.add_argument("--display_name", type=str, default = None)
    p.add_argument("--weights_filename", type=str, default = None)
    p.add_argument("--class_names_filename", type=str, default = None)
    return p.parse_args()


def _generate_config(args) :
    return ComboConfig(
        combo_id=args.combo_id,
        display_name=args.display_name or f"{args.model_name.replace("_", " ").title()} {args.classifier_type.title()}",
        model_name=args.model_name,
        classifier_type=args.classifier_type,
        weights_filename=args.wrights_filename or f"{args.model_name.split('.')[0]}_{args.classifier_type}_best.pth",
        class_names_filename=args.class_names_filename or f"{args.model_name.split('.')[0]}_class_names.txt",
    )

def main():
    args = _parse_args()

    COMBO = _generate_config(args)
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
