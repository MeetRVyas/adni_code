"""
PyTorch -> ONNX -> INT8 export pipeline shared by export_onnx.py.

Two empirical findings this module encodes (see PROJECT_NOTES.md for the
full investigation — both were verified against real exports of all four
target architectures, not assumed):

1. torch.onnx.export's newer dynamo-based path (dynamo=True, the current
   default) emits a small ONNX file plus a companion `.onnx.data` external-
   weights file once a model is large enough, and — more importantly for
   this pipeline — its graph shape for swin's window-attention pattern trips
   up onnxruntime.quantization's shape-inference step. The older
   `dynamo=False` exporter produces a single self-contained .onnx file
   (simpler for the manifest/HF Hub upload: one file per model, not a pair)
   and was the only path that quantized successfully for every one of the 4
   target backbones in testing. Used here for that reason.

2. onnxruntime.quantization.quantize_dynamic() succeeded directly for three
   of the four backbones (swin, efficientnet, vit) but raised on resnext50
   ("Expected onnx::Conv_518 to be an initializer") until the model was run
   through quant_pre_process() first — which in turn raised on the swin
   graph. Rather than hardcode a per-architecture branch (which breaks the
   moment a 5th backbone is added), robust_quantize() tries the cheap direct
   path first and falls back to preprocessing only if that fails — this
   handled all 4 known cases and should degrade gracefully for a new one too.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

QuantStrategy = Literal["direct", "preprocessed"]


def extract_exportable_module(classifier) -> nn.Module:
    """
    Every BaseClassifier subclass exposes `.model` as the actual forward()
    target (raw timm model for Baseline/Progressive, UniversalEvidentialModel
    wrapper for Evidential) — see module/classifiers/base_classifier.py. This
    is what gets exported, not the classifier wrapper itself (fit/evaluate/
    etc. have no place in an inference graph).
    """
    return classifier.model


def export_to_onnx(module: nn.Module, img_size: int, output_path: Path, opset: int = 17) -> None:
    module.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device = next(module.parameters()).device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["raw_output"],  # logits for Baseline/Progressive, evidence for Evidential
            dynamic_axes={"pixel_values": {0: "batch"}, "raw_output": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # see module docstring — required for reliable downstream quantization
        )
    onnx.checker.check_model(onnx.load(str(output_path)))


def robust_quantize(fp32_path: Path, int8_path: Path) -> QuantStrategy:
    """Dynamic INT8 weight quantization with an automatic fallback. See module docstring."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        return "direct"
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            prep_path = Path(td) / "preprocessed.onnx"
            quant_pre_process(str(fp32_path), str(prep_path), skip_optimization=False)
            quantize_dynamic(str(prep_path), str(int8_path), weight_type=QuantType.QInt8)
        return "preprocessed"


def verify_onnx_parity(
    pytorch_module: nn.Module, onnx_path: Path, img_size: int, num_samples: int = 4
) -> float:
    """
    Runs the same random input through the original PyTorch module and the
    exported ONNX graph, returns the max absolute difference. Call this right
    after export_to_onnx() (fp32 graph) — it is a correctness self-check on
    the export itself, independent of what the model was trained to predict,
    so it works even before real weights exist.
    """
    pytorch_module.eval()
    dummy = torch.randn(num_samples, 3, img_size, img_size, device = next(pytorch_module.parameters()).device)
    with torch.no_grad():
        torch_out = pytorch_module(dummy).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"pixel_values": dummy.cpu().numpy()})[0]

    return float(np.abs(torch_out - onnx_out).max())
