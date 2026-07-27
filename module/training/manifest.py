"""
Post-training artifact generation: re-evaluates each trained combo on its
held-out test split and assembles the two JSON files the serving app
consumes at startup (PRD §8.2, §8.5):

    manifest.json  - model registry: which weight/ONNX file goes with which
                      backbone/classifier/preprocessing, for every combo.
    results.json   - precomputed metrics (accuracy/recall/confusion matrix/etc)
                      for the dashboard's auto-rendered leaderboard + confusion
                      matrix grid. Computed once here, not recomputed live by
                      the app, per PRD §8.5.

Both files are uploaded to the HF Hub artifact repo alongside the weights;
see PROJECT_NOTES.md for the exact upload steps and the app/core/manifest.py
Pydantic models that parse these back out on the serving side (that file is
the authoritative schema reference — keep both in sync if you change either).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from module.classifiers import get_classifier
from module.models import get_img_size
from module.training.combo_runner import ComboConfig
from module.training.data_split import build_loader, load_and_split

# ImageNet normalization used by module.utils.get_base_transformations for
# every combo — duplicated here as an explicit constant (rather than
# imported) so the manifest always states the numbers it actually used,
# even if get_base_transformations changes later.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

MANIFEST_VERSION = "1.0"

# Ordered so 'resnext' is checked before a hypothetical generic 'resnet'
# entry would ever be added — mirrors the dispatch order already used by
# ArchitectureLayerGroups.get_layer_groups for consistency with the rest of
# the codebase, but only needs to distinguish family, not pick a function.
_ARCHITECTURE_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("swin", "swin"),
    ("efficientnet", "efficientnet"),
    ("vit", "vit"),
    ("resnext", "resnet"),
    ("resnet", "resnet"),
    ("mobilenet", "mobilenet"),
]


def infer_architecture_family(model_name: str) -> str:
    """
    Maps a timm model identifier to a coarse family used by the app's
    explainability adapter registry (transformer-style reshape_transform
    needed for swin/vit, plain conv target layer for everything else).
    Raises rather than guessing, so a genuinely new family gets a deliberate
    decision instead of silently falling into the wrong adapter.
    """
    base_name = model_name.split(".")[0].lower()
    for needle, family in _ARCHITECTURE_FAMILY_PATTERNS:
        if needle in base_name:
            return family
    raise ValueError(
        f"Cannot infer architecture_family for '{model_name}' — none of "
        f"{[p[0] for p in _ARCHITECTURE_FAMILY_PATTERNS]} matched. Add a mapping "
        f"in module/training/manifest.py before using this backbone."
    )


@dataclass
class ComboEvalResult:
    combo: ComboConfig
    architecture_family: str
    img_size: int
    class_names: list[str]
    metrics: dict[str, Any]
    weights_path: Path


def evaluate_combo(
    cfg: ComboConfig,
    *,
    data_dir: str,
    save_dir: Path,
    device: str,
    test_split: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> ComboEvalResult:
    """
    Loads a trained checkpoint and re-evaluates it on the deterministic
    held-out test split (same split every combo used during training — see
    module/training/data_split.py's docstring for why that's guaranteed even
    though img_size differs per combo).
    """
    img_size = get_img_size(cfg.model_name)
    weights_path = save_dir / cfg.weights_filename
    if not weights_path.exists():
        raise FileNotFoundError(
            f"[{cfg.combo_id}] Expected checkpoint not found at {weights_path}. "
            f"Train this combo first (see train_*.py for this combo)."
        )

    split = load_and_split(data_dir, img_size, test_split, device)
    test_loader = build_loader(
        split.full_dataset, split.test_idx, batch_size, False,
        num_workers, pin_memory, persistent_workers,
    )

    ClassifierClass = get_classifier(cfg.classifier_type)
    clf = ClassifierClass(
        model_name=cfg.model_name, num_classes=len(split.class_names), device=device, pretrained=False
    )
    clf.load(str(weights_path))

    metrics = clf.evaluate(test_loader, split.class_names)

    return ComboEvalResult(
        combo=cfg,
        architecture_family=infer_architecture_family(cfg.model_name),
        img_size=img_size,
        class_names=split.class_names,
        metrics=metrics,
        weights_path=weights_path,
    )


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """sklearn/numpy metrics -> plain JSON-serializable types."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            out[k] = [[int(x) for x in row] for row in v]
        elif k == "per_class_recall":
            out[k] = [float(x) for x in v]
        elif k in ("preds", "labels", "probs"):
            continue  # per-sample arrays: not needed for the dashboard, would bloat results.json
        elif hasattr(v, "item"):  # numpy scalar
            out[k] = v.item()
        else:
            out[k] = v
    return out


def build_manifest_and_results(
    eval_results: list[ComboEvalResult],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assembles the two JSON payloads from a list of per-combo evaluations."""
    now = datetime.now(timezone.utc).isoformat()

    manifest = {
        "version": MANIFEST_VERSION,
        "generated_at": now,
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
        "combos": [],
    }
    results = {
        "version": MANIFEST_VERSION,
        "generated_at": now,
        "combos": {},
    }

    for r in eval_results:
        onnx_stub = r.combo.combo_id  # export_onnx.py writes files named "<combo_id>_{fp32,int8}.onnx"
        manifest["combos"].append({
            "combo_id": r.combo.combo_id,
            "display_name": r.combo.display_name,
            "model_name": r.combo.model_name,
            "architecture_family": r.architecture_family,
            "classifier_type": r.combo.classifier_type,
            "is_evidential": r.combo.classifier_type == "evidential",
            "img_size": r.img_size,
            "class_names": r.class_names,
            "pytorch_weights_file": r.combo.weights_filename,
            "onnx_fp32_file": f"{onnx_stub}_fp32.onnx",
            "onnx_int8_file": f"{onnx_stub}_int8.onnx",
            # Both fields below are placeholders until benchmark_runtimes.py
            # runs: it validates INT8 recall parity against this same test
            # set before ever promoting selected_precision to "int8", and
            # times both execution providers before setting the preferred one.
            "selected_precision": "fp32",
            "preferred_execution_provider": "CPUExecutionProvider",
        })
        results["combos"][r.combo.combo_id] = _json_safe_metrics(r.metrics)

    return manifest, results


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
