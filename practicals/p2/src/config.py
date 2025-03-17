from pathlib import Path
import torch

# Task-specific parameters
# ...

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
MODELS_DIR = PROJECT_DIR / "models"

# Raw data subdirectories
IMAGES_DIR = RAW_DATA_DIR / "images"
ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"

# List of all required directories
REQUIRED_DIRS = [
    DATA_DIR,
    RAW_DATA_DIR,
    MODELS_DIR,
    IMAGES_DIR,
    ANNOTATIONS_DIR,
]

USING_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USING_CUDA else "cpu")