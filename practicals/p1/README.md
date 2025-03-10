# Practical 1

This repository contains the code and analysis for Practical 1. The main implementation can be found in the `src/` directory, while the central notebook for reviewing and reproducing results is `main.ipynb`.

## Project Structure

```
├── data/             # Data files
│   ├── 00_raw/       # Raw data
│   ├── 01_histories/ # Training histories
│   ├── 02_results/   # Results
│   └── 03_labels/    # Labels
├── notes/            # Analysis notebooks
│   ├── 0.3_dataset_analysis_2025-02-27.ipynb  # Dataset exploration
│   └── 1.1_results_analysis.ipynb  # Detailed analysis of results
├── report/figures/   # Report figures
├── scripts/          # Utility scripts
├── src/              # Source code for the implementation
│   ├── main.ipynb    # Main notebook
│   └── ...           # Supporting py files
├── requirements.txt  # Dependencies
└── requirements_cuda.txt # Dependencies for GPU
```

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
# For systems without GPU:
pip install -r requirements.txt

# For systems with GPU:
pip install -r requirements_cuda.txt
```

## Running the Code

The main entry point for reviewing and reproducing the results is `main.ipynb`. This notebook contains:
- Data loading and preprocessing
- Model training and evaluation

## Additional Analysis

Detailed analyses can be found in the `notes/` directory:
- `notes/0.3_dataset_analysis_2025-02-27.ipynb`: Explores the dataset characteristics and preprocessing steps
- `notes/1.1_results_analysis.ipynb`: Contains in-depth analysis of the experimental results

## Makefile Commands

The project includes a Makefile for common operations. View available commands with:
```bash
make help
```

You can set up the raw data by running:
```bash
make setup_raw_data
```

This will download the raw data and extract it into the `data/00_raw` directory.

You can also create a zip of the entire project by running:
```bash
make zip
```

This will create a zip of the entire project and save it as `AgundezLangParentRecaldeSanchez_OR_P1.zip` in the root directory.

