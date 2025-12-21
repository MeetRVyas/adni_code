# ADNI Medical Image Classification Pipeline

## 🎯 Overview

Production-ready PyTorch pipeline for cross-validation of CNN and Transformer architectures on the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. Supports 25+ pretrained models with comprehensive evaluation metrics and explainability analysis.

## 📁 Project Structure

```
project/
├── module/                  # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration & hyperparameters
│   ├── models.py           # Model architecture definitions
│   ├── utils.py            # Training utilities & data loading
│   ├── cross_validation.py # K-fold validation orchestrator
│   ├── test.py             # Test set evaluation
│   └── visualization.py    # Plotting & XAI analysis
├── abcd.ipynb              # Batch processing orchestrator
├── hf_manager.py           # Hugging Face backup manager
├── requirements.txt        # Python dependencies
├── OPTIMIZATION_REPORT.md  # Detailed optimization documentation
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your dataset in ImageFolder format:
```
OriginalDataset/
├── class_0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_1/
│   └── ...
└── class_N/
    └── ...
```

### 3. Configuration

Edit `module/config.py` to set your preferences:

```python
# Training hyperparameters
EPOCHS = 25
NFOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4

# Optimization settings
USE_AMP = True              # Automatic Mixed Precision (recommended)
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Data paths
DATA_DIR = "OriginalDataset"
OUTPUT_DIR = "output"
```

### 4. Run Training

```bash
jupyter notebook abcd.ipynb
# Execute all cells, or programmatically:
# run_queue(use_aug=False)  # Without augmentation
# run_queue(use_aug=True)   # With augmentation
```

## 🧠 Supported Models

### Heavy Models (Batch size: 3)
- Vision Transformers: `vit_base_patch16_224`, `vit_tiny_patch16_224`
- Swin Transformer: `swin_base_patch4_window7_224`
- MaxViT: `maxvit_tiny_224`
- EfficientNet: `tf_efficientnet_b4`
- ConvNeXt: `convnext_small`

### Medium Models (Batch size: 5)
- EfficientNetV2: `tf_efficientnetv2_s`
- ConvNeXt Tiny: `convnext_tiny`
- CoAtNet: `coatnet_0_rw_224`
- ResNet: `resnet50`, `resnext50_32x4d`
- DenseNet: `densenet121`
- Inception: `inception_v3`
- Xception: `xception`
- VGG: `vgg16_bn`

### Light Models (Batch size: 8)
- MobileViT: `mobilevit_s`
- EfficientFormer: `efficientformer_l1`
- PoolFormer: `poolformer_s12`
- ResNet18: `resnet18`
- EfficientNet: `efficientnet_b0`
- MobileNetV3: `mobilenetv3_large_100`
- GhostNet: `ghostnet_100`

## 📊 Outputs

### 1. Training Metrics
- **Master CSV**: `output/master_results.csv`
  - Per-model accuracy, precision, recall, F1
  - Cross-validation statistics (mean, std)
  - Training configuration

### 2. Model Checkpoints
- **Location**: `output/best_models/`
- **Format**: `{model_name}_pre={pretrained}_aug={use_aug}_best_model.pth`

### 3. Visualizations (per experiment)
- Training history curves (loss, accuracy, F1, precision, recall)
- Confusion matrix (normalized)
- Per-class performance heatmap
- ROC curves (multi-class)
- Precision-Recall curves
- Confidence distribution analysis
- Cumulative gain curves
- GradCAM explainability visualizations

### 4. Text Reports
- **Location**: `output/results/reports/`
- Comprehensive metrics breakdown
- Per-class specificity, TP, FP, FN, TN

### 5. Logs
- **Location**: `output/logs/`
- Debug logs: Full execution trace
- Error logs: Exception stack traces

## ⚙️ Advanced Configuration

### Custom Model Addition

```python
# In abcd.ipynb
custom_models = ['your_model_name_from_timm']
batches.append(custom_models)
```

### Hyperparameter Tuning

```python
# In module/config.py
EPOCHS = 50                    # Increase training duration
BATCH_SIZE = 64                # Larger batches (requires more VRAM)
LEARNING_RATE = 1e-3           # Modify in cross_validation.py

# Learning rate scheduler options:
# - OneCycleLR (current, recommended)
# - CosineAnnealingLR
# - ReduceLROnPlateau
```

### GPU Augmentation

```python
# In module/utils.py - get_gpu_augmentations()
return torch.nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add this
).to(DEVICE)
```

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

1. **Reduce batch size**:
   ```python
   BATCH_SIZE = 16  # or 8
   ```

2. **Enable gradient checkpointing** (for very large models):
   ```python
   model.set_grad_checkpointing(enable=True)
   ```

3. **Lower image resolution**:
   ```python
   # In models.py - RECOMMENDED_IMG_SIZES
   "your_model": 192,  # Instead of 224
   ```

### Slow Data Loading

1. **Reduce workers**:
   ```python
   NUM_WORKERS = 2
   ```

2. **Check I/O bottleneck**:
   - Move dataset to SSD
   - Reduce image compression

### Model Loading Failures

```python
# Check available models
import timm
print(timm.list_models('*efficientnet*'))
```

## 📈 Performance Tips

1. **Use AMP**: Already enabled by default (`USE_AMP = True`)
2. **Pin Memory**: Enabled for faster GPU transfers
3. **Persistent Workers**: Reduces worker respawn overhead
4. **Non-blocking Transfers**: Already implemented in training loop

## 🧪 Testing & Validation

### Quick Smoke Test

```python
# Edit config.py
EPOCHS = 2
NFOLDS = 2

# Run single model
run_subprocess(['resnet18'], use_aug=False)
```

### Memory Profiling

```bash
# Install pytorch profiler
pip install torch-tb-profiler

# Add to training loop
with torch.profiler.profile(...) as prof:
    train_one_epoch(...)
```
<!-- 
## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{adni_pipeline_2025,
  title={ADNI Medical Image Classification Pipeline},
  author={Meet Vyas},
  year={2024},
  url={https://github.com/yourusername/adni-pipeline}
}
``` -->

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

For issues, questions, or suggestions:
- **GitHub Issues**: [Project Issues](https://github.com/MeetRVyas/adni_code/issues)
- **Email**: vyasmeet2006@gmail.com

## 🔄 Changelog

### v2.0.0 (December 2025) - Optimization Release
- Fixed critical subprocess import bug
- Eliminated memory leaks in training loop
- Implemented proper scheduler stepping
- Optimized DataLoader configuration
- Added comprehensive error handling
- Improved XAI performance (20x faster)
- Enhanced logging and reporting

### v1.0.0 (Initial Release)
- Basic cross-validation pipeline
- Multi-model support
- Visualization suite