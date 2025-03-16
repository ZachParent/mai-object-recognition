
from torch import nn
from torchvision import models
from typing import Literal

type ModelName = Literal["resnet18", "resnet50"]

def get_model(model_name: ModelName):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        model =  models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not found")
    
    model.fc = nn.Linear(model.fc.in_features, 1000)
    return model
