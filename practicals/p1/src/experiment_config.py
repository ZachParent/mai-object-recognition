import pydantic

class ExperimentConfig(pydantic.BaseModel):
    id: int
    title: str
    net_name: list[str]
    train_from_scratch: bool
    warm_up: bool
    batch_size: int = 32
    n_epochs: int = 1
    last_layer_activation: str = "sigmoid"
    learning_rate: float = 0.001
    loss: str = "binary_crossentropy"


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
    # "hyperparameter-experiments": [
    #     {
    #         "id": 9,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 16,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.01,
    #         "loss": "binary_crossentropy",
    #     },
    #     {
    #         "id": 10,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 16,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.001,
    #         "loss": "binary_crossentropy",
    #     },
    #     {
    #         "id": 11,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 16,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.0001,
    #         "loss": "binary_crossentropy",
    #     },
    #     {
    #         "id": 12,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 32,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.01,
    #         "loss": "binary_crossentropy",
    #     },
    #     {
    #         "id": 13,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 32,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.001,
    #         "loss": "binary_crossentropy",
    #     },
    #     {
    #         "id": 14,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 32,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.0001,
    #         "loss": "binary_crossentropy",
    #     },
    #     {
    #         "id": 15,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 64,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.01,
    #         "loss": "binary_crossentropy",
    #     },
    #     {
    #         "id": 16,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 64,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.001,
    #         "loss": "binary_crossentropy",
    #     },
    #     {
    #         "id": 17,
    #         "title": "resnet50 no-pretraining no-warmup",
    #         "net_name": ["resnet50", "ResNet50"],
    #         "train_from_scratch": True,
    #         "warm_up": False,
    #         "batch_size": 64,
    #         "n_epochs": 1,
    #         "last_layer_activation": "sigmoid",
    #         "learning_rate": 0.0001,
    #         "loss": "binary_crossentropy",
    #     },
    # ],
    # "augmentation_experiments": [],
    # "classfier_head_experiments": [],
}

experiments = {
    exp_name: [ExperimentConfig(**exp) for exp in experiments[exp_name]]
    for exp_name in experiments
}
