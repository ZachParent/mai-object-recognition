from experiment_config import EXPERIMENT_SETS, best_model_experiment
from run_experiment import run_experiment
from pprint import pprint
from config import MINI_RUN
from metrics import compile_best_runs_csv

import torch.multiprocessing as mp

# Set spawn method at module level before any other multiprocessing operations
mp.set_start_method("spawn", force=True)


def main():
    # if MINI_RUN:
    #     print()
    #     print("==================================================")
    #     print("Running in mini-run mode")
    #     print("==================================================")
    #     print()

    # for experiment_set_getter in EXPERIMENT_SETS:
    #     experiment_set = experiment_set_getter()
    #     print("==================================================")
    #     print(f"\tRunning experiment set: {experiment_set.title}")
    #     print("==================================================")
    #     print()
    #     for experiment in experiment_set.configs:
    #         print(f"\t\tRunning experiment: {experiment.model_name}")
    #         pprint(experiment)
    #         print("==================================================")
    #         run_experiment(experiment)
    #     # Compile best_runs.csv with best runs of experiment set for each model
    #     compile_best_runs_csv(experiment_set)
    
    print("==================================================")
    print("Running best model with best hyperparameters")
    print("==================================================")
    
    print(f"\t\tRunning best model: {best_model_experiment.model_name}")
    pprint(best_model_experiment)
    print("==================================================")
    run_experiment(best_model_experiment, save_weights=True, epochs=8)


if __name__ == "__main__":
    main()
