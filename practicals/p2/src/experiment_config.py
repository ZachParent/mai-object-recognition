import pydantic

class ExperimentConfig(pydantic.BaseModel):
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
        ExperimentConfig(model_name="resnet18", learning_rate=0.001, batch_size=32),
        ExperimentConfig(model_name="resnet18", learning_rate=0.0001, batch_size=32),
    ],
)

experiments = [hyperparameter_search]
