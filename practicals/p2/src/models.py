from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models
from transformers import SegformerForSemanticSegmentation
from typing import Literal, Optional
from config import DEVICE, MODELS_DIR
from experiment_config import ModelName, ExperimentConfig
from dataset import MAIN_ITEM_NAMES

def get_model(model_name: ModelName, num_classes: int = 28, img_size: int = 224):
    """
    Get a model for semantic segmentation.

    Args:
        model_name: Name of the model architecture to use
        num_classes: Number of segmentation classes (including background)
        img_size: Size of input/output images (assumes square images)

    Returns:
        A PyTorch model for semantic segmentation
    """
    # Ensure num_classes is an integer
    if not isinstance(num_classes, int):
        try:
            num_classes = int(num_classes)
        except (ValueError, TypeError):
            raise TypeError(f"num_classes must be an integer, got {type(num_classes)}")

    # Ensure img_size is an integer
    if not isinstance(img_size, int):
        try:
            img_size = int(img_size)
        except (ValueError, TypeError):
            raise TypeError(f"img_size must be an integer, got {type(img_size)}")

    img_size = img_size // 4 if model_name == "segformer" else img_size

    # Print model parameters for debugging
    print(
        f"Creating {model_name} model with {num_classes} classes and {img_size}x{img_size} output size"
    )

    if model_name == "deeplab":
        # Load the pre-trained DeepLabv3 model
        model = models.segmentation.deeplabv3_resnet101(
            weights_backbone="ResNet101_Weights.DEFAULT", num_classes=num_classes
        )
    elif model_name == "segformer":
        # Load the pre-trained SegFormer model
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    elif model_name == "lraspp":
        # Load the pre-trained LR-ASPP model
        model = models.segmentation.lraspp_mobilenet_v3_large(
            weights_backbone="MobileNet_V3_Large_Weights.DEFAULT",
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Model {model_name} not found")

    return model.to(DEVICE)

def load_model_from_weights(experiment: ExperimentConfig, num_classes = len(MAIN_ITEM_NAMES)+1):
    path = f"{MODELS_DIR}/experimentId{experiment.id}_modelName{experiment.model_name}_lr{experiment.learning_rate}_img{experiment.img_size}.pt"
    model = get_model(model_name=experiment.model_name, img_size=experiment.img_size, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    return model