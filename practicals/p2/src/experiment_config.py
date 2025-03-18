import pydantic


class ExperimentConfig(pydantic.BaseModel):
    id: int
    model_name: str
    learning_rate: float
    batch_size: int
    epochs: int = 4


class ExperimentSet(pydantic.BaseModel):
    title: str
    configs: list[ExperimentConfig]


hyperparameter_search = ExperimentSet(
    title="Hyperparameter Search",
    configs=[
        ExperimentConfig(
            id=0, model_name="resnet18", learning_rate=0.001, batch_size=32
        ),
        ExperimentConfig(
            id=1, model_name="resnet18", learning_rate=0.0001, batch_size=32
        ),
    ],
)

EXPERIMENT_SETS = [hyperparameter_search]
