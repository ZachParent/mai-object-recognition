import pydantic
from typing import Literal
import pandas as pd
from config import MODELS
from metrics import METRICS_DIR

ModelName = Literal["deeplab", "segformer", "lraspp"]


def get_best_run_hyperparameter(experiment_set_title, model_name, hyperparameter):
    try:
        best_runs_df = pd.read_csv(f"{METRICS_DIR}/best_runs.csv")
        # get best run by experiment_set.title and model name
        best_run = best_runs_df[
            (best_runs_df["experiment_set"] == experiment_set_title)
            & (best_runs_df["model_name"] == model_name)
        ]
        if best_run.empty:
            raise Exception("No best runs found, please run experiments first")
        best_run_hyperparameter_value = best_run[hyperparameter].values[0]
        return best_run_hyperparameter_value
    except FileNotFoundError:
        raise Exception("No best runs found, please run experiments first")


class ExperimentConfig(pydantic.BaseModel):
    id: int
    model_name: ModelName
    learning_rate: float
    batch_size: int
    img_size: int


class ExperimentSet(pydantic.BaseModel):
    title: str
    configs: list[ExperimentConfig]


def get_learning_rate_experiments():
    learning_rates = [0.0005, 0.0001, 0.00005]
    id = 0
    experiment_configs = []
    for model in MODELS:
        for lr in learning_rates:
            experiment_configs.append(
                ExperimentConfig(
                    id=id,
                    model_name=model,
                    learning_rate=lr,
                    batch_size=4,
                    img_size=192,
                )
            )
            id += 1
    return ExperimentSet(title="Learning Rate Experiments", configs=experiment_configs)


def get_batch_size_experiments():
    batch_sizes = [2, 4, 8]
    id = 9
    experiment_configs = []
    for model in MODELS:
        for batch_size in batch_sizes:
            learning_rate = get_best_run_hyperparameter(
                "Learning Rate Experiments", model, "learning_rate"
            )
            experiment_configs.append(
                ExperimentConfig(
                    id=id,
                    model_name=model,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    img_size=192,
                )
            )
            id += 1
    return ExperimentSet(title="Batch Size Experiments", configs=experiment_configs)


def get_augmentation_experiments():
    id = 18
    experiment_configs = []
    for model in MODELS:
        learning_rate = get_best_run_hyperparameter(
            "Learning Rate Experiments", model, "learning_rate"
        )
        batch_size = get_best_run_hyperparameter(
            "Batch Size Experiments", model, "batch_size"
        )
        experiment_configs.append(
            ExperimentConfig(
                id=id,
                model_name=model,
                learning_rate=learning_rate,
                batch_size=batch_size,
                img_size=192,
            )
        )
        id += 1
    return ExperimentSet(title="Augmentation Experiments", configs=experiment_configs)


def get_resolution_experiments():
    id = 21
    experiment_configs = []
    for model in MODELS:
        learning_rate = get_best_run_hyperparameter(
            "Learning Rate Experiments", model, "learning_rate"
        )
        batch_size = get_best_run_hyperparameter(
            "Batch Size Experiments", model, "batch_size"
        )
        experiment_configs.append(
            ExperimentConfig(
                id=id,
                model_name=model,
                learning_rate=learning_rate,
                batch_size=batch_size,
                img_size=384,
            )
        )
        id += 1
    return ExperimentSet(title="Resolution Experiments", configs=experiment_configs)


EXPERIMENT_SETS = [
    get_learning_rate_experiments,
    get_batch_size_experiments,
    get_augmentation_experiments,
    get_resolution_experiments,
]
