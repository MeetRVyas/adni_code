import os
import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory structure
OUTPUT_DIR = "output"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DATA_DIR = "OriginalDataset"

# Classifier Configuration
OPTIMIZE_METRIC = 'recall'  # Primary metric: 'recall', 'accuracy', 'f1', 'precision'
MIN_DELTA_METRIC = 0.001  # Minimum improvement threshold for early stopping

# Training hyperparameters
EPOCHS = 30  # Total epochs (will be distributed in progressive training)
NFOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4
PRETRAINED = True
NUM_SAMPLES_TO_ANALYSE = 5  # For GradCAM/XAI visualization
TEST_SPLIT = 0.2
PATIENCE = 10
MIN_DELTA = 0.3  # For legacy compatibility
LR = 1e-4

# Optimization settings
USE_AMP = True  # Automatic Mixed Precision
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Memory management
EMPTY_CACHE_FREQUENCY = 1
SAVE_BEST_ONLY = True

# Timeout settings (in seconds)
SUBPROCESS_TIMEOUT = 8 * 3600

# Create necessary directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)