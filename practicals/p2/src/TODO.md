# zach's TODO for setting up p2

## src/
- `main.py`
  - [ ] use `# %%` to separate cells
  - [x] add main() to run as a script
  - [x] add `mini_run` config var, which sets epochs to 1, loads a smaller dataset
- `config.py`
  - [x] add epochs
- `experiment_configs.py`
  - [ ] add UnresolvedExperimentConfig class
    - allows for referencing a different config by name
  - [x] add ExperimentConfig class
  - [x] add ExperimentSet class
- `run_experiment.py`
  - [x] add main() to run as a script
    - this should run just 1 experiment by id, specified by an ExperimentConfig
  - [x] use tqdm to show progress?
  - [x] log to a file?
  - [x] log to tensorboard?
- `metrics.py`
  - [x] add MetricFn class
  - [x] add several classic metric functions
  - [x] add a demo custom metric function
  - [x] put these in a list of metrics
- `dataset.py`
  - [x] add a basic preprocessing step
  - [x] add this to a list of preprocessing steps
  - [x] leave a todo for adding more preprocessing steps
- `models.py`
  - [x] set up 1 model from torchvision.models
  - [x] put these in a list of models

## scripts/
- `setup_hooks.py`
  - [x] add a script to set up the hooks

## documentation
- `README.md`
  - [x] update README.md for team
