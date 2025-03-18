from config import RAW_DATA_DIR
from experiment_config import ExperimentConfig, EXPERIMENT_SETS
from dataset import get_dataloaders, load_category_mappings
from pathlib import Path
from config import TRAIN_ANNOTATIONS_JSON
from visualize import visualize_segmentation
from run_experiment import run_experiment
from pprint import pprint


def main():

    for experiment_set in EXPERIMENT_SETS:
        print("==================================================")
        print(f"\tRunning experiment set: {experiment_set.title}")
        print("==================================================")
        for experiment in experiment_set.configs:
            print(f"\t\tRunning experiment: {experiment.model_name}")
            pprint(experiment)
            print("==================================================")
            run_experiment(experiment)


if __name__ == "__main__":
    main()
