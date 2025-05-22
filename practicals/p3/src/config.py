# src/config.py

from pathlib import Path

# Directories
PROJECT_DIR = Path(__file__).resolve().parent.parent # project_root
DATA_DIR = PROJECT_DIR / "data"
MODEL_WEIGHTS_DIR = DATA_DIR / "model_weights" # New constant for all model weights
RAW_DATA_ZIP_PATH = DATA_DIR / "cloth3d++_subset.zip"
RAW_DATA_DIR = DATA_DIR / "cloth3d++_subset"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed_dataset"
SMPL_DIR = DATA_DIR / "smpl"
RESULTS_DIR = DATA_DIR / "01_results"
CHECKPOINTS_DIR = DATA_DIR / "02_checkpoints"
LOGS_DIR = DATA_DIR / "03_logs"
REPORT_DIR = PROJECT_DIR / "report" # Assuming report is at project_root/report
FIGURES_DIR = REPORT_DIR / "figures"
VISUALIZATIONS_DIR = FIGURES_DIR / "visualizations"

# You can also define the specific ViT checkpoint path here if you want
VIT_CHECKPOINT_DIR = MODEL_WEIGHTS_DIR / "vit_checkpoint" / "imagenet21k"
R50_VIT_B16_PRETRAINED_PATH = VIT_CHECKPOINT_DIR / "R50+ViT-B_16.npz"