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

# Training hyperparameters
EPOCHS = 25
NFOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4  # Reduced from 8 to prevent CPU bottleneck
PRETRAINED = True
NUM_SAMPLES_TO_ANALYSE = 5
TEST_SPLIT = 0.2
PATIENCE = 10
MIN_DELTA = 0.3
LR = 1e-4

# Optimization settings
USE_AMP = True  # Automatic Mixed Precision
PIN_MEMORY = True
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Memory management
EMPTY_CACHE_FREQUENCY = 1  # Clear cache every N epochs
SAVE_BEST_ONLY = True  # Only save best checkpoint, not every improvement

# Timeout settings (in seconds)
SUBPROCESS_TIMEOUT = 8 * 3600

# Create necessary directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
