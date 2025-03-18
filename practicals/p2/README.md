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

- download the dataset

```bash
python scripts/download_dataset.py
```

- run the experiments

```bash
python src/main.py
```
- `MINI_RUN`
  - by default, this will run in mini-run mode if CUDA is not available, which will use a smaller dataset and fewer epochs
  - you can disable this by setting `MINI_RUN` to `False` in `src/config.py`
  - or you can force mini-run mode by setting `MINI_RUN` to `True` in `src/config.py`

- monitor the experiments with tensorboard

```bash
tensorboard --logdir=data/01_runs
```
