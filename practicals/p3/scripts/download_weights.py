# practicals/p3/scripts/download_weights.py

import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Add src to Python path to import config
# This assumes the script is run from project_root or practicals/p3/scripts
# and that src is a sibling of practicals or can be found by going up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.config import R50_VIT_B16_PRETRAINED_PATH, VIT_CHECKPOINT_DIR
except ImportError as e:
    print(f"Error: Could not import configuration from src.config. Make sure PYTHONPATH is set correctly or run from project root.")
    print(f"PROJECT_ROOT determined as: {PROJECT_ROOT}")
    print(f"sys.path: {sys.path}")
    raise e


def download_file(url, destination_path: Path):
    """Downloads a file from a URL to a destination path with a progress bar."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=60) # Increased timeout
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 

        print(f"Downloading {url.split('/')[-1]} to {destination_path}")
        with open(destination_path, 'wb') as file, tqdm(
            desc=destination_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR: Download incomplete.")
            if destination_path.exists():
                destination_path.unlink()
            return False
        print("Download complete.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if destination_path.exists():
            try:
                destination_path.unlink()
            except OSError:
                pass
        return False

def main():
    weights_url = "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"
    
    # The destination path is now directly from src.config
    destination_file_path = R50_VIT_B16_PRETRAINED_PATH 
    
    if destination_file_path.exists():
        print(f"Weights file already exists at: {destination_file_path}")
        # You could add a size check here if desired:
        # expected_size_bytes = ... # Known size of R50+ViT-B_16.npz
        # if destination_file_path.stat().st_size == expected_size_bytes:
        #     print("File exists and size matches.")
        #     return
        # else:
        #     print("File exists but size mismatch. Re-downloading.")
        return # Simple check: if exists, do nothing
        
    print(f"Target location for weights: {destination_file_path}")
    
    # Ensure parent directory exists (also done in download_file)
    destination_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not download_file(weights_url, destination_file_path):
        print("Failed to download weights.")

if __name__ == "__main__":
    main()