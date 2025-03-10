from pathlib import Path

# Task-specific parameters

IMG_SIZE = 224
NUM_CLASSES = 20
VOC_CLASSES = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}
BATCH_SIZES = [16, 32, 64]

# Directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
TRAIN_TXT = DATA_DIR / "train.txt"
TEST_TXT = DATA_DIR / "test.txt"
RAW_DATA_DIR = DATA_DIR / "00_raw"
HISTORIES_DIR = DATA_DIR / "01_histories"
RESULTS_DIR = DATA_DIR / "02_results"
LABELS_DIR = DATA_DIR / "03_labels"
TRUE_LABELS_CSV = LABELS_DIR / "true.csv"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
