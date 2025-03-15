# zach's TODO for setting up p2

## src/
- `main.py`
  - [ ] use `# %%` to separate cells
  - [ ] add main() to run as a script
  - [ ] use argparse to parse arguments
    - [ ] add `micro_run` flag, which sets epochs to 1, steps to 2
- `config.py`
  - [ ] add epochs, steps
- `experiment_configs.py`
  - [ ] add UnresolvedExperimentConfig class
    - allows for referencing a different config by name
  - [ ] add ExperimentConfig class
  - [ ] add ExperimentSet class
- `run_experiment.py`
  - [ ] add main() to run as a script
    - this should run just 1 experiment by id, specified by an ExperimentConfig
  - [ ] use tqdm to show progress?
  - [ ] log to a file?
  - [ ] log to tensorboard?
- `metrics.py`
  - [ ] add MetricFn class
  - [ ] add several classic metric functions
  - [ ] add a demo custom metric function
  - [ ] put these in a list of metrics
- `dataset.py`
  - [ ] use huggingface's `datasets` library to load the dataset
  - [ ] add a basic preprocessing step
  - [ ] add this to a list of preprocessing steps
  - [ ] leave a todo for adding more preprocessing steps
- `models.py`
  - [ ] set up 1 model from torchvision.models
  - [ ] set up 1 model from huggingface's model hub
  - [ ] set up 1 custom model
  - [ ] put these in a list of models

## scripts/
- `setup_hooks.py`
  - [ ] add a script to set up the hooks

## documentation
- `README.md`
  - [ ] update README.md for team
