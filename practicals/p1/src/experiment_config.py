import pydantic
import itertools


class ExperimentConfig(pydantic.BaseModel):
    id: int
    title: str
    net_name: list[str] = ["resnet50", "ResNet50"]
    train_from_scratch: bool = False
    warm_up: bool = True
    batch_size: int = 32
    n_epochs: int = 1
    last_layer_activation: str = "sigmoid"
    learning_rate: float = 0.001
    loss: str = "binary_crossentropy"
    classifier_head: str = "default"


experiments = {
    "model-experiments": [
        {
            "id": 0,
            "title": "resnet50 no-pretraining no-warmup",
            "net_name": ["resnet50", "ResNet50"],
            "train_from_scratch": True,
            "warm_up": False,
        },
        {
            "id": 1,
            "title": "resnet50 pretraining no-warmup",
            "net_name": ["resnet50", "ResNet50"],
            "train_from_scratch": False,
            "warm_up": False,
        },
        {
            "id": 2,
            "title": "resnet50 pretraining warmup",
            "net_name": ["resnet50", "ResNet50"],
            "train_from_scratch": False,
            "warm_up": True,
        },
        {
            "id": 3,
            "title": "inception_v3 no-pretraining no-warmup",
            "net_name": ["inception_v3", "InceptionV3"],
            "train_from_scratch": True,
            "warm_up": False,
        },
        {
            "id": 4,
            "title": "inception_v3 pretraining no-warmup",
            "net_name": ["inception_v3", "InceptionV3"],
            "train_from_scratch": False,
            "warm_up": False,
        },
        {
            "id": 5,
            "title": "inception_v3 pretraining warmup",
            "net_name": ["inception_v3", "InceptionV3"],
            "train_from_scratch": False,
            "warm_up": True,
        },
        {
            "id": 6,
            "title": "mobilenet_v2 no-pretraining no-warmup",
            "net_name": ["mobilenet_v2", "MobileNetV2"],
            "train_from_scratch": True,
            "warm_up": False,
        },
        {
            "id": 7,
            "title": "mobilenet_v2 pretraining no-warmup",
            "net_name": ["mobilenet_v2", "MobileNetV2"],
            "train_from_scratch": False,
            "warm_up": False,
        },
        {
            "id": 8,
            "title": "mobilenet_v2 pretraining warmup",
            "net_name": ["mobilenet_v2", "MobileNetV2"],
            "train_from_scratch": False,
            "warm_up": True,
        },
    ],
    "hyperparameter-experiments": [
        {
            "id": 9 + i,
            "title": f"batch_size: {batch_size}, learning_rate: {learning_rate}",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        for i, (batch_size, learning_rate) in enumerate(
            itertools.product([16, 32, 64], [0.001, 0.01, 0.1])
        )
    ],
    "augmentation-experiments": [],  # Add to the id of classifier_head_experiments the number of experiments performed here
    "classfier_head-experiments": [
        {
            "id": 18 + i,
            "title": f"classifier_head: {classifier_head}",
            "classifier_head": classifier_head,
        }
        for i, classifier_head in enumerate(["ensemble", "attention"])
    ],
}

# validate the experiment configs compile
experiments = {
    exp_name: [ExperimentConfig(**exp) for exp in experiments[exp_name]]
    for exp_name in experiments
}
