"""
Shared logic for benchmark_runtimes.py.

Two independent decisions live here, matching two independent PRD concerns:

1. selected_precision (fp32 vs int8) — an ACCURACY decision. PRD Risk: INT8
   quantization could disproportionately hurt recall on minority dementia
   classes, so recall must be re-validated on the quantized model against
   the full-precision one before shipping. `evaluate_recall` computes recall
   from both ONNX graphs against the real held-out test set; `choose_precision`
   only promotes int8 if it doesn't lose more than RECALL_TOLERANCE recall.

2. preferred_execution_provider (CPU vs OpenVINO) — a SPEED decision, timed
   only on whichever precision was just selected, per PRD §8.3.

Note on argmax equivalence (why this file never special-cases the evidential
combo): EvidentialClassifier's raw output is Dirichlet "evidence", not
logits, but both the softmax used for standard combos and the
evidence -> alpha -> alpha/S transform used for evidential combos are
strictly monotonic per-class rescalings applied identically to every class
(softmax) or via the same positive scalar S (evidential) — neither changes
which class has the largest value. So argmax(raw_output) always equals the
classifier's actual predicted class regardless of classifier_type, and
recall can be computed uniformly here without reimplementing
get_predictions_and_uncertainty from module/classifiers/evidential.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader

RECALL_TOLERANCE = 0.02  # int8 may trail fp32 macro recall by at most 2 points and still be selected
WARMUP_RUNS = 3
TIMED_RUNS = 20


def _predict_all(sess: ort.InferenceSession, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Runs every batch in loader through an ONNX session, returns (preds, labels)."""
    all_preds, all_labels = [], []
    input_name = sess.get_inputs()[0].name
    for images, labels in loader:
        raw = sess.run(None, {input_name: images.numpy()})[0]
        all_preds.append(np.argmax(raw, axis=1))
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


@dataclass
class PrecisionDecision:
    selected_precision: str
    recall_fp32: float
    recall_int8: float
    recall_drop: float


def choose_precision(fp32_path: Path, int8_path: Path, test_loader: DataLoader) -> PrecisionDecision:
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    preds_fp32, labels = _predict_all(sess_fp32, test_loader)
    preds_int8, _ = _predict_all(sess_int8, test_loader)

    recall_fp32 = float(recall_score(labels, preds_fp32, average="macro", zero_division=0))
    recall_int8 = float(recall_score(labels, preds_int8, average="macro", zero_division=0))
    drop = recall_fp32 - recall_int8

    selected = "int8" if drop <= RECALL_TOLERANCE else "fp32"
    return PrecisionDecision(selected, recall_fp32, recall_int8, drop)


@dataclass
class LatencyDecision:
    preferred_execution_provider: str
    latency_ms: dict[str, float]  # provider -> mean ms/inference, only for providers that loaded


def choose_execution_provider(onnx_path: Path, img_size: int) -> LatencyDecision:
    """
    Times CPUExecutionProvider vs OpenVINOExecutionProvider on single-image
    inference (matching the app's actual per-request shape). Falls back to
    CPU-only if the OpenVINO EP isn't installed/available in this environment
    (it ships in the `onnxruntime-openvino` package — see requirements.txt —
    which replaces plain `onnxruntime`; if only plain onnxruntime is
    installed, OpenVINOExecutionProvider simply won't appear here and CPU
    wins by default rather than the benchmark erroring out).
    """
    import numpy as np

    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    input_name = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    candidates = [["OpenVINOExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    timings: dict[str, float] = {}

    for providers in candidates:
        provider_label = providers[0]
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            if provider_label not in sess.get_providers():
                continue  # requested EP not actually active (not installed) — skip, don't fake a number
            for _ in range(WARMUP_RUNS):
                sess.run(None, {input_name: dummy})
            t0 = time.perf_counter()
            for _ in range(TIMED_RUNS):
                sess.run(None, {input_name: dummy})
            timings[provider_label] = (time.perf_counter() - t0) / TIMED_RUNS * 1000
        except Exception as exc:
            print(f"    ({provider_label} unavailable: {exc})")

    if not timings:
        raise RuntimeError(f"No execution provider could run {onnx_path} — is onnxruntime installed correctly?")

    preferred = min(timings, key=timings.get)
    return LatencyDecision(preferred, timings)
