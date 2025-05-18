from pathlib import Path

# Directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_ZIP_PATH = DATA_DIR / "cloth3d++_subset.zip"
RAW_DATA_DIR = DATA_DIR / "cloth3d++_subset"
RESULTS_DIR = DATA_DIR / "01_results"
CHECKPOINTS_DIR = DATA_DIR / "02_checkpoints"
LOGS_DIR = DATA_DIR / "03_logs"
REPORT_DIR = PROJECT_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"
