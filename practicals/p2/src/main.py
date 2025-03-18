from experiment_config import EXPERIMENT_SETS
from run_experiment import run_experiment
from pprint import pprint
from config import MINI_RUN


def main():
    if MINI_RUN:
        print()
        print("==================================================")
        print("Running in mini-run mode")
        print("==================================================")
        print()

    for experiment_set in EXPERIMENT_SETS:
        print("==================================================")
        print(f"\tRunning experiment set: {experiment_set.title}")
        print("==================================================")
        print()
        for experiment in experiment_set.configs:
            print(f"\t\tRunning experiment: {experiment.model_name}")
            pprint(experiment)
            print("==================================================")
            run_experiment(experiment)


if __name__ == "__main__":
    main()
