import pydantic
from typing import Literal
import pandas as pd
from metrics import METRICS_DIR

ModelName = Literal["deeplab", "segformer", "lraspp"]

# Model names
MODELS = ["deeplab", "segformer", "lraspp"]


class ExperimentConfig(pydantic.BaseModel):
    id: int
    model_name: ModelName
    learning_rate: float
    batch_size: int
    augmentation: bool = False
    img_size: int
    visualize: bool = False  # TODO Add visualize=True for running best model


class ExperimentSet(pydantic.BaseModel):
    title: str
    configs: list[ExperimentConfig]


LEARNING_RATE_EXPERIMENTS_NAME = "Learning Rate Experiments"


def get_learning_rate_experiments() -> ExperimentSet:
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
                    augmentation=False,
                    img_size=192,
                )
            )
            id += 1
    return ExperimentSet(
        title=LEARNING_RATE_EXPERIMENTS_NAME, configs=experiment_configs
    )


BATCH_SIZE_EXPERIMENTS_NAME = "Batch Size Experiments"


def get_batch_size_experiments() -> ExperimentSet:
    batch_sizes = [4, 8, 16]
    id = 9
    experiment_configs = []
    for model in MODELS:
        for batch_size in batch_sizes:
            learning_rate = get_best_run_hyperparameter(
                LEARNING_RATE_EXPERIMENTS_NAME, model, "learning_rate"
            )
            experiment_configs.append(
                ExperimentConfig(
                    id=id,
                    model_name=model,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    augmentation=False,
                    img_size=192,
                )
            )
            id += 1
    return ExperimentSet(title=BATCH_SIZE_EXPERIMENTS_NAME, configs=experiment_configs)


AUGMENTATION_EXPERIMENTS_NAME = "Augmentation Experiments"


def get_augmentation_experiments() -> ExperimentSet:
    id = 18
    experiment_configs = []
    for model in MODELS:
        learning_rate = get_best_run_hyperparameter(
            LEARNING_RATE_EXPERIMENTS_NAME, model, "learning_rate"
        )
        batch_size = get_best_run_hyperparameter(
            BATCH_SIZE_EXPERIMENTS_NAME, model, "batch_size"
        )
        experiment_configs.append(
            ExperimentConfig(
                id=id,
                model_name=model,
                learning_rate=learning_rate,
                batch_size=batch_size,
                augmentation=True,
                img_size=192,
            )
        )
        id += 1
    return ExperimentSet(
        title=AUGMENTATION_EXPERIMENTS_NAME, configs=experiment_configs
    )


RESOLUTION_EXPERIMENTS_NAME = "Resolution Experiments"


def get_resolution_experiments() -> ExperimentSet:
    id = 21
    experiment_configs = []
    for model in MODELS:
        learning_rate = get_best_run_hyperparameter(
            LEARNING_RATE_EXPERIMENTS_NAME, model, "learning_rate"
        )
        batch_size = get_best_run_hyperparameter(
            BATCH_SIZE_EXPERIMENTS_NAME, model, "batch_size"
        )
        augmentation = get_best_run_hyperparameter(
            AUGMENTATION_EXPERIMENTS_NAME, model, "augmentation"
        )
        experiment_configs.append(
            ExperimentConfig(
                id=id,
                model_name=model,
                learning_rate=learning_rate,
                batch_size=batch_size,
                augmentation=augmentation,
                img_size=384,
            )
        )
        id += 1
    return ExperimentSet(title=RESOLUTION_EXPERIMENTS_NAME, configs=experiment_configs)


def get_best_run_hyperparameter(
    experiment_set_title: str, model_name: str, hyperparameter: str
) -> float | bool:
    try:
        best_runs_df = pd.read_csv(f"{METRICS_DIR}/best_runs.csv")

        if hyperparameter == "augmentation":
            # Filter dataframe for the given model
            model_df = best_runs_df[best_runs_df["model_name"] == model_name]

            # Get the row with the highest dice score
            best_run = model_df.loc[model_df["dice"].idxmax()]

            if best_run.empty:
                raise ValueError("No best runs found, please run experiments first")

            return best_run["augmentation"]

        # get best run by experiment_set.title and model name
        best_run = best_runs_df[
            (best_runs_df["experiment_set"] == experiment_set_title)
            & (best_runs_df["model_name"] == model_name)
        ]
        if best_run.empty:
            raise ValueError("No best runs found, please run experiments first")
        best_run_hyperparameter_value = best_run[hyperparameter].values[0]
        return best_run_hyperparameter_value
    except FileNotFoundError:
        raise ValueError("No best runs found, please run experiments first")


best_model_experiment = ExperimentConfig(
    id=99,
    model_name="deeplab",
    batch_size=16,
    learning_rate=0.0001,
    augmentation=False,
    img_size=384
)


EXPERIMENT_SETS = [
    get_learning_rate_experiments,
    get_batch_size_experiments,
    get_augmentation_experiments,
    get_resolution_experiments,
]
