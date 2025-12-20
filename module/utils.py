import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import logging
import shutil
from .config import DEVICE, LOG_DIR


class FullDataset(torch.utils.data.Dataset):
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
        if self.transform: image = self.transform(image)
        return image, label

class Logger:
    def __init__(self, name: str = "Logger", file_name: str = "batch"):
        self.logger = logging.getLogger(name)
        self.current_log_dir = os.path.join(LOG_DIR, name)
        os.makedirs(self.current_log_dir, exist_ok = True)
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
            console_formatter = logging.Formatter("[%(name)s] %(message)s")

            base_file_name = os.path.join(self.current_log_dir, file_name)

            info_path = f"{base_file_name}_debug.log"
            info_handler = logging.FileHandler(info_path, encoding = "utf-8")
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

    def error(self, message: str) -> None:
        self.logger.error(message, exc_info=True)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)


def get_gpu_augmentations(img_size):
    """ 
    Defines augmentations that will be applied on the GPU to a batch of tensors.
    This is extremely fast.
    """
    # Note: These transforms expect a batch of tensors, not PIL Images.
    return torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ).to("cuda") # Move the augmentation module itself to the GPU

def train_one_epoch(model, loader, criterion, optimizer, scaler, gpu_augmenter = None, scheduler = None):
    # This function is unchanged
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.enable_grad() :
        for images, labels in tqdm(loader, desc="   Training", leave=False):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            if gpu_augmenter:
                images = gpu_augmenter(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward(), scaler.step(optimizer), scaler.update()

            if scheduler :
                scheduler.step()

            running_loss += loss.item() * images.size(0)
            
            # Collect stats
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds) * 100.0
    
    return total_loss, acc

def validate_one_epoch(model, loader, criterion):
    # This function is unchanged
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="   Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Collect stats
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc, loss = 100.0 * accuracy_score(all_labels, all_preds), running_loss / len(loader)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return loss, acc, prec, rec, f1


def get_base_transformations(img_size):
    """ The standard, non-augmented transformations. Converts PIL Image to Tensor. """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), # CRITICAL: Converts PIL Image to a [0, 1] float tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _aggressive_empty_directory(folder_path):
    if not os.path.exists(folder_path):
        return

    print(f"🧹 Cleaning inside: {folder_path}")

    # List everything in the current folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # CASE A: It is a File
        if (os.path.isfile(item_path) or os.path.islink(item_path)) and not item.endswith(".zip")  :
            os.remove(item_path)

        elif os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
            except Exception:
                print(f"   ⚠️ Could not remove dir '{item}', emptying its contents instead...")
                _aggressive_empty_directory(item_path)
                try:
                    os.rmdir(item_path)
                except:
                    print(f"   Note: Directory '{item}' remains (but is empty).")

def zip_and_empty(source_dir, output_zip):
    import zipfile
    if not os.path.exists(source_dir):
        print(f"Directory '{source_dir}' does not exist. Skipping.")
        return

    print(f"📦 Zipping '{source_dir}' to '{output_zip}'...")

    try:
        # --- 1. Create the Zip File ---
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create structure: 'results/subdir/file.txt'
                    parent_dir = os.path.dirname(source_dir)
                    arcname = os.path.relpath(file_path, parent_dir)
                    zipf.write(file_path, arcname=arcname)
                    
        # --- 2. Verify & Empty ---
        if os.path.exists(output_zip):
            print(f"✅ Zip created successfully: {output_zip}")
            print(f"🗑️  Emptying target folder: {source_dir}")
            
            # Call the recursive emptying function
            _aggressive_empty_directory(source_dir)
            
            print("✅ Cleanup complete.")
        else:
            print("❌ Zip creation failed. Folder NOT emptied.")

    except Exception as e:
        print(f"❌ An error occurred during process: {e}")