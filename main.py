import argparse
import json


def _parse_args():
    parser = argparse.ArgumentParser(description="ADNI Training & Testing Pipeline")

    # --- Core ---
    parser.add_argument("--models",         type=str, required=True,
                        help="JSON list of model names, e.g. '[\"resnet18\"]'")
    parser.add_argument("--classifier_map", type=str, default="{}",
                        help="JSON dict mapping model_name -> classifier_type")

    # --- Mode ---
    parser.add_argument("--mode", type=str, default="cv", choices=["cv", "test"],
                        help="'cv' for cross-validation (default), 'test' for quick testing")
    parser.add_argument("--test_all", action="store_true", default=False,
                        help="(test mode only) Run all classifiers instead of using classifier_map")

    # --- Paths ---
    parser.add_argument("--data_dir",   type=str, default=None,
                        help="Path to dataset root (overrides config.DATA_DIR)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to output root (overrides config.OUTPUT_DIR)")

    # --- Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config.EPOCHS)")
    parser.add_argument("--folds",  type=int, default=None,
                        help="Number of CV folds (overrides config.NFOLDS)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Number of epochs for Early Stopping (overrides config.PATIENCE)")

    # --- BDA Pipeline ---
    parser.add_argument("--disease", type=str, default=None,
                        help="Disease tag for BDA pipeline (overrides config.DISEASE_ID). "
                             "E.g. 'adni', 'chestxray14', 'isic2024', 'retinamnist'.")
    parser.add_argument("--layout", type=str, default=None,
                        choices=["imagefolder", "chestxray14", "isic2024",
                                 "medmnist_png", "medmnist_hf"],
                        help="Dataset layout for registry build (overrides config.DATASET_LAYOUT)")
    parser.add_argument("--registry_dir", type=str, default=None,
                        help="BDA registry directory (overrides config.REGISTRY_DIR)")
    parser.add_argument("--shards_dir", type=str, default=None,
                        help="BDA shards directory (overrides config.SHARDS_DIR)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Annotation CSV path for chestxray14/isic2024 layouts "
                             "(overrides config.DATASET_CSV)")
    parser.add_argument("--no_bda", action="store_true", default=False,
                        help="Disable BDA pipeline; use legacy FullDataset/ImageFolder mode.")
    parser.add_argument("--no_mlflow", action="store_true", default=False,
                        help="Disable MLflow tracking; use CSV-only logging.")
    parser.add_argument("--run_pipeline_only", action="store_true", default=False,
                        help="Run the BDA ETL pipeline for the given disease and exit "
                             "(no model training).")

    return parser.parse_args()


def run_batch(models, classifier_map, logger):
    from module import Cross_Validator
    logger.info(f"Starting cross-validation for {models}")
    logger.info(f"Classifier mapping: {classifier_map}")

    validator = Cross_Validator(
        models,
        logger,
        model_classifier_map=classifier_map,
    )
    validator.run()
    logger.info("Cross-validation complete")


def run_test(models, classifier_map, test_all, logger):
    from module.classifiers.testing import test_all_classifiers_on_model, test_single_classifier
    from module.utils import get_base_transformations
    from module.models import get_img_size
    from module.config import (
        DATA_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TEST_SPLIT,
        DEVICE, OPTIMIZE_METRIC, EPOCHS, LR,
        USE_BDA_PIPELINE, DISEASE_ID, REGISTRY_DIR, SHARDS_DIR,
    )
    from module.utils import FullDataset
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from sklearn.model_selection import train_test_split
    import numpy as np
    import torch

    logger.info(f"Starting test mode for {models}")
    logger.info(f"Test-all classifiers: {test_all}")

    # ── Resolve class info and weights ────────────────────────────────────────
    if USE_BDA_PIPELINE:
        from data_pipeline.registry.disease_registry import DiseaseRegistry
        reg = DiseaseRegistry(REGISTRY_DIR, use_dask=False)
        classes = reg.class_names(DISEASE_ID)
        class_weights_np = reg.class_weights(DISEASE_ID, split="train")
        class_weights_tensor = torch.FloatTensor(class_weights_np).to(DEVICE)
        targets = None   # resolved per model via BDA loader
    else:
        temp_transform = transforms.Compose([transforms.ToTensor()])
        base_dataset = FullDataset(DATA_DIR, temp_transform)
        targets = np.array(base_dataset.targets)
        classes = base_dataset.classes
        class_counts = np.bincount(targets)
        total_samples = len(targets)
        class_weights = total_samples / (len(classes) * class_counts)
        class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

        train_val_indices, test_indices = train_test_split(
            np.arange(len(targets)),
            test_size=TEST_SPLIT,
            stratify=targets,
            random_state=42,
        )
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.2,
            stratify=targets[train_val_indices],
            random_state=42,
        )

    for model_name in models:
        logger.info(f"\nSetting up data for model: {model_name}")
        img_size = get_img_size(model_name)

        if USE_BDA_PIPELINE:
            from data_pipeline.loaders.webdataset_loader import get_dataloader
            train_loader = get_dataloader(
                disease=DISEASE_ID, fold=0, split="train",
                img_size=img_size, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS, registry_dir=REGISTRY_DIR,
                shards_dir=SHARDS_DIR, pin_memory=PIN_MEMORY,
            )
            val_loader = get_dataloader(
                disease=DISEASE_ID, fold=0, split="val",
                img_size=img_size, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS, registry_dir=REGISTRY_DIR,
                shards_dir=SHARDS_DIR, pin_memory=PIN_MEMORY,
            )
            test_loader = get_dataloader(
                disease=DISEASE_ID, fold=-1, split="test",
                img_size=img_size, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS, registry_dir=REGISTRY_DIR,
                shards_dir=SHARDS_DIR, pin_memory=PIN_MEMORY,
            )
        else:
            transform = get_base_transformations(img_size)
            full_dataset = FullDataset(DATA_DIR, transform)
            train_loader = DataLoader(
                Subset(full_dataset, train_indices),
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            )
            val_loader = DataLoader(
                Subset(full_dataset, val_indices),
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            )
            test_loader = DataLoader(
                Subset(full_dataset, test_indices),
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            )

        if test_all:
            logger.info(f"Running test_all_classifiers_on_model for {model_name}")
            test_results = test_all_classifiers_on_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                classifiers='all',
                class_weights_tensor=class_weights_tensor,
                class_names=classes,
                epochs=EPOCHS,
                lr=LR,
                primary_metric=OPTIMIZE_METRIC,
                device=DEVICE,
            )
        else:
            classifier_type = classifier_map.get(model_name, 'baseline')
            logger.info(f"Running test_single_classifier for {model_name} with '{classifier_type}'")
            test_results, classifier = test_single_classifier(
                classifier_name=classifier_type,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_names=classes,
                class_weights_tensor=class_weights_tensor,
                epochs=EPOCHS,
                lr=LR,
                primary_metric=OPTIMIZE_METRIC,
                device=DEVICE,
            )
            logger.info(
                f"best_recall -> {classifier.best_recall} "
                f"best_acc -> {classifier.best_acc} "
                f"best_f1 -> {classifier.best_f1} "
                f"best_epoch -> {classifier.best_epoch}"
            )
        logger.info(f"Test mode complete for {model_name}")


def configure(
    data_dir:     str  = None,
    output_dir:   str  = None,
    epochs:       int  = None,
    nfolds:       int  = None,
    patience:     int  = None,
    disease:      str  = None,
    layout:       str  = None,
    registry_dir: str  = None,
    shards_dir:   str  = None,
    csv:          str  = None,
    use_bda:      bool = None,
    use_mlflow:   bool = None,
):
    """
    Override mutable config values at runtime.
    Only non-None arguments are applied.
    """
    import os
    import module.config as cfg

    if data_dir is not None:
        cfg.DATA_DIR = data_dir

    if output_dir is not None:
        cfg.OUTPUT_DIR  = output_dir
        cfg.RESULTS_DIR = os.path.join(output_dir, "results")
        cfg.PLOTS_DIR   = os.path.join(cfg.RESULTS_DIR, "plots")
        cfg.REPORTS_DIR = os.path.join(cfg.RESULTS_DIR, "reports")
        cfg.LOG_DIR     = os.path.join(output_dir, "logs")

    if epochs is not None:
        cfg.EPOCHS = epochs

    if nfolds is not None:
        cfg.NFOLDS = nfolds

    if patience is not None:
        cfg.PATIENCE = patience

    # BDA overrides
    if disease is not None:
        cfg.DISEASE_ID = disease

    if layout is not None:
        cfg.DATASET_LAYOUT = layout

    if registry_dir is not None:
        cfg.REGISTRY_DIR = registry_dir

    if shards_dir is not None:
        cfg.SHARDS_DIR = shards_dir

    if csv is not None:
        cfg.DATASET_CSV = csv

    if use_bda is not None:
        cfg.USE_BDA_PIPELINE = use_bda

    if use_mlflow is not None:
        cfg.USE_MLFLOW = use_mlflow

    # (Re-)create directories after any path changes
    os.makedirs(cfg.PLOTS_DIR,   exist_ok=True)
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,     exist_ok=True)


if __name__ == "__main__":
    args = _parse_args()

    configure(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        nfolds=args.folds,
        patience=args.patience,
        disease=args.disease,
        layout=args.layout,
        registry_dir=args.registry_dir,
        shards_dir=args.shards_dir,
        csv=args.csv,
        use_bda=False if args.no_bda else None,
        use_mlflow=False if args.no_mlflow else None,
    )

    # ── Pipeline-only mode: run ETL and exit ──────────────────────────────────
    if args.run_pipeline_only:
        import module.config as cfg
        from data_pipeline.pipeline import run_pipeline
        run_pipeline(
            disease=cfg.DISEASE_ID,
            source_dir=cfg.DATA_DIR,
            layout=cfg.DATASET_LAYOUT,
            source="local",
            nfolds=cfg.NFOLDS,
            registry_dir=cfg.REGISTRY_DIR,
            shards_dir=cfg.SHARDS_DIR,
            csv=cfg.DATASET_CSV,
            build_shards=True,
            compute_stats=True,
            validate=True,
            skip_if_exists=False,   # force rebuild when explicitly requested
        )
        print("\n[main] Pipeline run complete. Exiting.")
        import sys; sys.exit(0)

    from module import Logger

    models         = json.loads(args.models)
    classifier_map = json.loads(args.classifier_map)
    logger = Logger("batch_" + str(hash(str(models)))[:8])

    if args.mode == "cv":
        run_batch(models, classifier_map, logger)
    elif args.mode == "test":
        run_test(models, classifier_map, args.test_all, logger)