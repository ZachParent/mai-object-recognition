# Practical 2

## Requirements

- Python 3.11

## Installation

- create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

- install the requirements

```bash
pip install -r requirements.txt
```

## Setup scripts

- download the dataset and place it in `data/cloth3d++_subset/`

- preprocess the dataset

```bash
python -m scripts.unified_preprocessing.preprocessing
```

- download TransUNet weights

```bash
python -m scripts.download_weights
```

## Run experiments

- run all experiments

```bash
python -m src.main
```
- run just one test experiment

```bash
python -m src.trainer --test
```

## Post-processing

- run the post-processing script

```bash
python -m scripts.get_individual_metrics
```

## Visualize the results
- visualize the results with streamlit

```bash
streamlit run notebooks/streamlit_app.py
```

A subset of the functionality is available publicly at https://mai-or-p3.streamlit.app/

Full functionality requires the large dataset locally.

## Important files

- `src/main.py` - main script to run the experiments
- `src/trainer.py` - script to train the model
- `src/run_configs.py` - configuration for the experiments
- `src/datasets/cloth3d.py` - dataset class for the cloth3d++ dataset
- `src/models/unet2d.py` - UNet model
- `src/models/transUnet.py` - TransUNet model
- `notebooks/streamlit_app.py` - streamlit app to visualize the results

- `data/01_results/`
  - `frame_metrics.csv` - metrics for each frame for our 3 comparative models
  - `run_<id>/` - results for each experiment
    - `train.csv` - metrics for each epoch
    - `val.csv` - metrics for each epoch
    - `test.csv` - final test metrics
  - `run_configs.csv` - table of the run configurations
