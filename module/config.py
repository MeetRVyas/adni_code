import os

DEVICE = "cuda"
OUTPUT_DIR = "output"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DATA_DIR = "OriginalDataset"
EPOCHS = 25
NFOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 8
PRETRAINED = True
NUM_SAMPLES_TO_ANALYSE = 5
TEST_SPLIT = 0.2

os.makedirs(PLOTS_DIR, exist_ok = True)
os.makedirs(REPORTS_DIR, exist_ok = True)
os.makedirs(LOG_DIR, exist_ok = True)