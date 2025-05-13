# %%
import zipfile
import os
import sys

sys.path.append("..")
from src.config import DATA_DIR, RAW_DATA_ZIP_PATH, RAW_DATA_DIR

# %%
DOWNLOAD_URL = "https://cvcuab-my.sharepoint.com/:u:/g/personal/mmadadi_cvc_uab_cat/EaJUHQv5N2dEjvA51WbGLdIB5aVjZfQraF0Fa0tprVMBYA?e=rJv9sZ"

if RAW_DATA_DIR.exists():
    print(f"Dataset already exists in {RAW_DATA_DIR}")
elif RAW_DATA_ZIP_PATH.exists():
    print(f"Unzipping dataset from {RAW_DATA_ZIP_PATH} to {DATA_DIR}")
    with zipfile.ZipFile(RAW_DATA_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
    print(f"Removing zip file {RAW_DATA_ZIP_PATH}")
    os.remove(RAW_DATA_ZIP_PATH)
else:
    print(f"Dataset not found in {DATA_DIR}")
    print(f"Please download the dataset from {DOWNLOAD_URL}, and place the zip file in {DATA_DIR}")

# %%
# preprocess the dataset