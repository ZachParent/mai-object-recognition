import pydantic
from typing import Literal

ModelName = Literal["deeplab", "segformer", "lraspp"]

class ExperimentConfig(pydantic.BaseModel):
    id: int
    model_name: ModelName
    learning_rate: float
    batch_size: int
    img_size: int

class ExperimentSet(pydantic.BaseModel):
    title: str
    configs: list[ExperimentConfig]

learning_rate_experiments = ExperimentSet(
    title="Learning Rate Experiments",
    configs=[
        ExperimentConfig(id=0, model_name="deeplab", learning_rate=0.0001, batch_size=2, img_size=192),
        ExperimentConfig(id=1, model_name="deeplab", learning_rate=0.001, batch_size=2, img_size=192),
        ExperimentConfig(id=2, model_name="deeplab", learning_rate=0.01, batch_size=2, img_size=192),
        ExperimentConfig(id=3, model_name="segformer", learning_rate=0.0001, batch_size=2, img_size=192),
        ExperimentConfig(id=2, model_name="segformer", learning_rate=0.001, batch_size=2, img_size=192),
        ExperimentConfig(id=5, model_name="segformer", learning_rate=0.01, batch_size=2, img_size=192),
        ExperimentConfig(id=6, model_name="lraspp", learning_rate=0.0001, batch_size=2, img_size=192),
        ExperimentConfig(id=7, model_name="lraspp", learning_rate=0.001, batch_size=2, img_size=192),
        ExperimentConfig(id=8, model_name="lraspp", learning_rate=0.01, batch_size=2, img_size=192),
    ],
)

batch_size_experiments = ExperimentSet(
    title="Batch Size Experiments",
    configs=[
        ExperimentConfig(id=9, model_name="deeplab", learning_rate=0.0001, batch_size=16, img_size=192),
        ExperimentConfig(id=10, model_name="deeplab", learning_rate=0.0001, batch_size=32, img_size=192),
        ExperimentConfig(id=11, model_name="deeplab", learning_rate=0.0001, batch_size=64, img_size=192),
        ExperimentConfig(id=12, model_name="segformer", learning_rate=0.0001, batch_size=16, img_size=192),
        ExperimentConfig(id=13, model_name="segformer", learning_rate=0.0001, batch_size=32, img_size=192),
        ExperimentConfig(id=14, model_name="segformer", learning_rate=0.0001, batch_size=64, img_size=192),
        ExperimentConfig(id=15, model_name="lraspp", learning_rate=0.0001, batch_size=16, img_size=192),
        ExperimentConfig(id=16, model_name="lraspp", learning_rate=0.0001, batch_size=32, img_size=192),
        ExperimentConfig(id=17, model_name="lraspp", learning_rate=0.0001, batch_size=64, img_size=192),
    ],
)

augmentation_experiments = ExperimentSet(
    title="Augmentation Experiments",
    configs=[
        ExperimentConfig(id=18, model_name="deeplab", learning_rate=0.0001, batch_size=16, img_size=192),
        ExperimentConfig(id=19, model_name="segformer", learning_rate=0.0001, batch_size=16, img_size=192),
        ExperimentConfig(id=20, model_name="lraspp", learning_rate=0.0001, batch_size=16, img_size=192),
    ],
)

resolution_experiments = ExperimentSet(
    title="Resolution Experiments",
    configs=[
        ExperimentConfig(id=21, model_name="deeplab", learning_rate=0.0001, batch_size=16, img_size=384),
        ExperimentConfig(id=22, model_name="segformer", learning_rate=0.0001, batch_size=16, img_size=384),
        ExperimentConfig(id=23, model_name="lraspp", learning_rate=0.0001, batch_size=16, img_size=384),
    ],
)

EXPERIMENT_SETS = [learning_rate_experiments, batch_size_experiments, augmentation_experiments, resolution_experiments]