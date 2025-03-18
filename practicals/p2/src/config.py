from pathlib import Path
import torch
from urllib.parse import urlparse

# Fashionpedia dataset URLs
FASHIONPEDIA_URLS = {
    "train_images": "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip",
    "val_test_images": "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip",
    "train_instances": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json",
    "val_instances": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json",
    "test_info": "https://s3.amazonaws.com/ifashionist-dataset/annotations/info_test2020.json",
    "train_attributes": "https://s3.amazonaws.com/ifashionist-dataset/annotations/attributes_train2020.json",
    "val_attributes": "https://s3.amazonaws.com/ifashionist-dataset/annotations/attributes_val2020.json",
}

# Directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "00_raw"
RUNS_DIR = DATA_DIR / "01_runs"
METRICS_DIR = DATA_DIR / "02_metrics"

# Raw data subdirectories
IMAGES_DIR = RAW_DATA_DIR / "images"
TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
VAL_IMAGES_DIR = IMAGES_DIR / "val"
ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"
TRAIN_ANNOTATIONS_JSON = (
    ANNOTATIONS_DIR / urlparse(FASHIONPEDIA_URLS["train_instances"]).path.split("/")[-1]
)
VAL_ANNOTATIONS_JSON = (
    ANNOTATIONS_DIR / urlparse(FASHIONPEDIA_URLS["val_instances"]).path.split("/")[-1]
)

# List of all required directories
REQUIRED_DIRS = [
    DATA_DIR,
    RAW_DATA_DIR,
    RUNS_DIR,
    METRICS_DIR,
    IMAGES_DIR,
    ANNOTATIONS_DIR,
]

# CUDA settings and mini-run
USING_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USING_CUDA else "cpu")
# Set to True to run a mini-run with less data and fewer epochs
MINI_RUN = not USING_CUDA

NUM_EPOCHS = 1 if MINI_RUN else 4
