import torch
from torchvision import transforms
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import os
import logging
import shutil
from .config import *
from .augmentation import CutMix, Mixup, get_gpu_augmentations


class FullDataset(torch.utils.data.Dataset):
    """
    Legacy ImageFolder-compatible dataset.
    Used when USE_BDA_PIPELINE=False or as a fallback inside the BDA loader.
    """
    def __init__(self, data_dir, transform):
        from torchvision.datasets import ImageFolder
        self.transform = transform
        data = ImageFolder(root=data_dir)
        self.samples = data.samples
        self.targets = data.targets
        self.classes = data.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def ensure_pipeline_ready(
    disease_id: str,
    data_dir: str,
    layout: str,
    registry_dir: str,
    shards_dir: str,
    nfolds: int,
    target_size: int = 224,
    csv_path=None,
    auto_run: bool = True,
) -> bool:
    """
    Check whether the BDA registry exists for the given disease.
    If it doesn't exist and auto_run=True, run the pipeline to create it.

    Returns True if the registry is ready, False otherwise.
    """
    import os
    from pathlib import Path

    parquet_path = Path(registry_dir) / f"{disease_id}.parquet"

    if parquet_path.exists():
        return True

    if not auto_run:
        print(
            f"[utils] BDA registry not found: {parquet_path}\n"
            f"  Run the data pipeline first:\n"
            f"    python -m data_pipeline.pipeline --disease {disease_id} "
            f"--source_dir {data_dir} --layout {layout}"
        )
        return False

    print(f"[utils] Registry not found for '{disease_id}'. Running pipeline automatically...")
    try:
        from data_pipeline.pipeline import run_pipeline
        run_pipeline(
            disease=disease_id,
            source_dir=data_dir,
            layout=layout,
            source="local",
            nfolds=nfolds,
            registry_dir=registry_dir,
            shards_dir=shards_dir,
            target_size=target_size,
            csv=csv_path,
            build_shards=True,
            compute_stats=True,
            validate=True,
            skip_if_exists=True,
        )
        return parquet_path.exists()
    except Exception as e:
        print(f"[utils] Pipeline auto-run failed: {e}")
        print("[utils] Falling back to legacy FullDataset.")
        return False


class Logger:
    def __init__(self, name: str = "Logger", file_name: str = "batch"):
        self.logger = logging.getLogger(name)
        self.current_log_dir = os.path.join(LOG_DIR, name)
        os.makedirs(self.current_log_dir, exist_ok=True)

        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)

            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
            console_formatter = logging.Formatter("[%(name)s] %(message)s")

            base_file_name = os.path.join(self.current_log_dir, file_name)

            info_path = f"{base_file_name}_debug.log"
            info_handler = logging.FileHandler(info_path, encoding="utf-8")
            info_handler.setLevel(logging.DEBUG)
            info_handler.setFormatter(formatter)

            error_path = f"{base_file_name}_error.log"
            error_handler = logging.FileHandler(error_path)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)

            self.logger.addHandler(info_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = True) -> None:
        self.logger.error(message, exc_info)

    def debug(self, message: str) -> None:
        self.logger.debug(message)


# Sharpness-Aware Minimization (SAM)
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer.

    PUBLISHED: "Sharpness-Aware Minimization" (ICLR 2021)
    USAGE: SOTA on ImageNet, medical imaging competitions
    """

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if isinstance(params, (list, tuple)) and isinstance(params[0], dict):
            param_groups = params
        else:
            param_groups = [{'params': params}]

        defaults = dict(rho=rho, **kwargs)
        super().__init__(param_groups, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def train_one_epoch_with_augments(model, loader, criterion, optimizer, scaler, strategy, scheduler=None):
    """Optimized training loop with proper memory management"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    mixup = Mixup(alpha=0.2) if strategy.use_mixup else None
    cutmix = CutMix(alpha=1.0) if strategy.use_cutmix else None
    simple_augments = get_gpu_augmentations(strategy.img_size) if strategy.use_simple_augments else None
    grad_accum_steps = strategy.gradient_accumulation_steps

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        if mixup and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = mixup(images, labels)
            use_mixed = True
        elif cutmix and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = cutmix(images, labels)
            use_mixed = True
        elif simple_augments and np.random.rand() < 0.5:
            images = simple_augments(images)
            use_mixed = False
        else:
            use_mixed = False

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
            outputs = model(images)

            if use_mixed:
                loss = (mixup or cutmix).loss_function(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler and isinstance(scheduler, (
                optim.lr_scheduler.OneCycleLR,
                optim.lr_scheduler.SequentialLR
            )):
                scheduler.step()

        running_loss += loss.item() * grad_accum_steps * images.size(0)

        if not use_mixed:
            _, predicted = outputs.detach().max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    if all_preds:
        acc = accuracy_score(all_labels, all_preds) * 100.0
    else:
        acc = 0.0

    return total_loss, acc


def train_one_epoch(model, loader, criterion, optimizer, scaler, gpu_augmenter=None, scheduler=None):
    """Optimized training loop with proper memory management"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        if gpu_augmenter:
            images = gpu_augmenter(images)

        optimizer.zero_grad(set_to_none=True)

        if isinstance(optimizer, SAM):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if scheduler and isinstance(scheduler, (
            optim.lr_scheduler.OneCycleLR,
            optim.lr_scheduler.SequentialLR
        )):
            scheduler.step()

        running_loss += loss.detach().item() * images.size(0)

        _, predicted = outputs.detach().max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds) * 100.0

    return total_loss, acc


def validate_one_epoch(model, loader, criterion):
    """Optimized validation loop"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    return total_loss, acc, prec, rec, f1


def get_base_transformations(img_size):
    """Standard preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _aggressive_empty_directory(folder_path):
    """Recursively empties a directory while preserving structure"""
    if not os.path.exists(folder_path):
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if (os.path.isfile(item_path) or os.path.islink(item_path)) and not item.endswith(".zip"):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
            except Exception:
                _aggressive_empty_directory(item_path)
                try:
                    os.rmdir(item_path)
                except Exception:
                    pass


def zip_and_empty(source_dir, output_zip):
    """
    Creates a zip archive and empties source directory.
    Used for saving results before cleanup.
    """
    import zipfile

    if not os.path.exists(source_dir):
        print(f"Directory '{source_dir}' does not exist. Skipping.")
        return

    print(f"📦 Zipping '{source_dir}' to '{output_zip}'...")

    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    parent_dir = os.path.dirname(source_dir)
                    arcname = os.path.relpath(file_path, parent_dir)
                    zipf.write(file_path, arcname=arcname)

        if os.path.exists(output_zip):
            print(f"✅ Zip created successfully: {output_zip}")
            print(f"🗑️ Emptying target folder: {source_dir}")
            _aggressive_empty_directory(source_dir)
            print("✅ Cleanup complete.")
        else:
            print("❌ Zip creation failed. Folder NOT emptied.")

    except Exception as e:
        print(f"❌ An error occurred during process: {e}")
