import pydantic
from typing import Literal

type ModelName = Literal["resnet18", "resnet50"]


class ExperimentConfig(pydantic.BaseModel):
    id: int
    model_name: ModelName
    learning_rate: float
    batch_size: int
    img_size: int


class ExperimentSet(pydantic.BaseModel):
    title: str
    configs: list[ExperimentConfig]


hyperparameter_search = ExperimentSet(
    title="Hyperparameter Search",
    configs=[
        ExperimentConfig(
            id=0,
            model_name="resnet18",
            learning_rate=0.001,
            batch_size=32,
            img_size=224,
        ),
        ExperimentConfig(
            id=1,
            model_name="resnet18",
            learning_rate=0.0001,
            batch_size=32,
            img_size=224,
        ),
    ],
)

EXPERIMENT_SETS = [hyperparameter_search]
