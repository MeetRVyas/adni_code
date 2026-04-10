"""
FastAPI inference server for the trained Swin model.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io, os, sys, json, zipfile, tempfile
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from module.models      import get_img_size
from module.classifiers import get_classifier

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME       = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
CLASSIFIER_TYPE  = "progressive"
SAVE_DIR         = ROOT / "saved_models"
WEIGHTS_PATH     = SAVE_DIR / "swin_progressive_best.pth"
CLASS_NAMES_PATH = SAVE_DIR / "swin_class_names.txt"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load model once at startup ──────────────────────────────────────────────
app = FastAPI(title="Alzheimer MRI Classifier")

_model = None
_class_names: List[str] = []
_transform = None


@app.on_event("startup")
def load_model():
    global _model, _class_names, _transform

    if not WEIGHTS_PATH.exists():
        print(f"[WARN] Weights not found at {WEIGHTS_PATH}. Run train_swin.py first.")
        return

    _class_names = CLASS_NAMES_PATH.read_text().strip().splitlines() if CLASS_NAMES_PATH.exists() \
        else ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

    ClassifierClass = get_classifier(CLASSIFIER_TYPE)
    clf = ClassifierClass(
        model_name=MODEL_NAME,
        num_classes=len(_class_names),
        device=DEVICE,
    )
    clf.load(str(WEIGHTS_PATH))
    clf.model.eval()
    _model = clf

    img_size = get_img_size(MODEL_NAME)
    _transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"[INFO] Model loaded. Classes: {_class_names}  Device: {DEVICE}")


# ─── Inference helpers ───────────────────────────────────────────────────────

def _predict_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        outputs = _model.forward(img_tensor.to(DEVICE))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = F.softmax(outputs, dim=1)
    return probs.cpu().numpy()


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0)


def _predict_image(pil_img: Image.Image) -> dict:
    tensor = _pil_to_tensor(pil_img)
    probs  = _predict_tensor(tensor)[0]          # shape (n_classes,)
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": _class_names[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "probabilities":   {c: float(p) for c, p in zip(_class_names, probs)},
    }


def _aggregate_max_prob(results: List[dict]) -> dict:
    """For each class, take the max probability across all images."""
    agg = {c: 0.0 for c in _class_names}
    for r in results:
        for c, p in r["probabilities"].items():
            if p > agg[c]:
                agg[c] = p
    pred_class = max(agg, key=agg.get)
    return {
        "mode":            "max_probability",
        "predicted_class": pred_class,
        "max_probabilities": agg,
        "num_images":      len(results),
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": DEVICE}


@app.get("/classes")
def classes():
    return {"classes": _class_names}


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    mode: str = Form("max_probability"),   # "max_probability" | "per_image"
):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run train_swin.py first.")

    results = []
    for uf in files:
        raw = await uf.read()
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Cannot open image: {uf.filename}")
        res = _predict_image(pil)
        res["filename"] = uf.filename
        results.append(res)

    if mode == "per_image":
        return JSONResponse({"mode": "per_image", "results": results})

    # default: max_probability aggregate
    agg = _aggregate_max_prob(results)
    agg["per_image"] = results
    return JSONResponse(agg)


# ─── Single-page UI (served at /) ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    return FileResponse("index.html")