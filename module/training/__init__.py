"""
Shared infrastructure for the four deployment-combo training scripts
(train_efficientnet.py, train_vit_evidential.py, train_resnext_baseline.py,
and — if you choose to migrate it — train_swin.py).

Public API:
    ComboConfig, TrainingRunResult, train_combo   — module.training.combo_runner
    SplitDataset, load_and_split, build_loader     — module.training.data_split
    ExperimentTracker, build_tracker               — module.training.tracking
"""

from module.training.benchmarking import (
    LatencyDecision,
    PrecisionDecision,
    choose_execution_provider,
    choose_precision,
)
from module.training.combo_runner import ComboConfig, TrainingRunResult, train_combo
from module.training.data_split import SplitDataset, build_loader, load_and_split
from module.training.manifest import (
    ComboEvalResult,
    build_manifest_and_results,
    evaluate_combo,
    infer_architecture_family,
    write_json,
)
from module.training.onnx_export import (
    export_to_onnx,
    extract_exportable_module,
    robust_quantize,
    verify_onnx_parity,
)
from module.training.tracking import ExperimentTracker, build_tracker

__all__ = [
    "ComboConfig",
    "TrainingRunResult",
    "train_combo",
    "SplitDataset",
    "build_loader",
    "load_and_split",
    "ExperimentTracker",
    "build_tracker",
    "ComboEvalResult",
    "evaluate_combo",
    "build_manifest_and_results",
    "infer_architecture_family",
    "write_json",
    "extract_exportable_module",
    "export_to_onnx",
    "robust_quantize",
    "verify_onnx_parity",
    "PrecisionDecision",
    "LatencyDecision",
    "choose_precision",
    "choose_execution_provider",
]
