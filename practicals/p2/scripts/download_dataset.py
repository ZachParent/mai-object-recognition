#!/usr/bin/env python3
"""
Script to download and extract the Fashionpedia dataset.
"""

import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import os
# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import (
    FASHIONPEDIA_URLS,
    IMAGES_DIR,
    ANNOTATIONS_DIR,
    REQUIRED_DIRS,
)


def download_file(url: str, dest_path: Path) -> None:
    """
    Download a file from a URL to a destination path with a progress bar.

    Args:
        url: URL to download from
        dest_path: Path where the file should be saved
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    desc = f"Downloading {dest_path.name}"
    with open(dest_path, "wb") as f, tqdm(
        desc=desc,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def extract_zip(zip_path: Path, extract_path: Path) -> None:
    """
    Extract a ZIP file to a specified path.

    Args:
        zip_path: Path to the ZIP file
        extract_path: Path where contents should be extracted
    """
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Finished extracting {zip_path.name}")


def main():
    # Create all required directories
    print("Creating required directories...")
    for dir_path in REQUIRED_DIRS:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Download and extract image files
    image_files = {
        "train_images": IMAGES_DIR / "train2020.zip",
        "val_test_images": IMAGES_DIR / "val_test2020.zip",
    }

    for key, dest_path in image_files.items():
        if not dest_path.exists():
            print(f"\nDownloading {key}...")
            download_file(FASHIONPEDIA_URLS[key], dest_path)
            extract_zip(dest_path, IMAGES_DIR)
        else:
            print(f"\nSkipping {key} - file already exists")
    if (IMAGES_DIR / "test").exists():
        print("Renaming test/ to val/...")
        os.rename(IMAGES_DIR / "test", IMAGES_DIR / "val")
    else:
        print("test/ does not exist, skipping renaming")

    # Download annotation files
    annotation_files = {
        "train_instances": "instances_attributes_train2020.json",
        "val_instances": "instances_attributes_val2020.json",
        "test_info": "info_test2020.json",
        "train_attributes": "attributes_train2020.json",
        "val_attributes": "attributes_val2020.json",
    }

    for key, filename in annotation_files.items():
        dest_path = ANNOTATIONS_DIR / filename
        if not dest_path.exists():
            print(f"\nDownloading {key}...")
            download_file(FASHIONPEDIA_URLS[key], dest_path)
        else:
            print(f"\nSkipping {key} - file already exists")


if __name__ == "__main__":
    main()
