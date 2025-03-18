
from torch import nn
from torchvision import models
from typing import Literal


type ModelName = Literal["resnet18", "resnet50"]

def get_model(model_name: ModelName):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model {model_name} not found")
    
    model.fc = nn.Linear(model.fc.in_features, 1000)
    return model
