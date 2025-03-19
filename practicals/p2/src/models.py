from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models
from typing import Literal, Optional
from config import DEVICE


type ModelName = Literal["resnet18", "resnet50"]


class SegmentationHead(nn.Module):
    """
    Segmentation head that upsamples feature maps to produce segmentation masks.
    Uses a flexible approach that can handle any input image size.
    """

    def __init__(self, in_channels: int, num_classes: int, img_size: int):
        super().__init__()

        # Calculate the feature map size after ResNet backbone
        # For ResNet, features are downsampled by a factor of 32
        self.feature_size = img_size // 32
        self.img_size = img_size

        # Create a more flexible decoder that doesn't rely on exact upsampling dimensions
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Series of upsampling blocks
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Final layer to get to num_classes
        self.final = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Input x comes from the backbone and has shape [batch_size, in_channels, H/32, W/32]
        # Reshape to expected input format if needed (e.g., if coming from a flattened layer)
        if len(x.shape) == 2:
            # If x is flattened, reshape it to [batch_size, channels, H, W]
            batch_size = x.shape[0]
            x = x.view(batch_size, -1, self.feature_size, self.feature_size)

        # Apply the decoder layers
        x = self.initial_conv(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.final(x)

        # Use interpolation for the final sizing to ensure we get exactly the right output size
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        return x


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

    # Print model parameters for debugging
    print(
        f"Creating {model_name} model with {num_classes} classes and {img_size}x{img_size} output size"
    )

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features  # Typically 512 for ResNet18
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features  # Typically 2048 for ResNet50
    else:
        raise ValueError(f"Model {model_name} not found")

    # Modify the ResNet backbone for segmentation
    # Remove average pooling and final fully connected layer
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()

    # Create a new segmentation model using the modified backbone
    segmentation_model = SegmentationResNet(
        backbone=model,
        in_features=in_features,
        num_classes=num_classes,
        img_size=img_size,
    )

    return segmentation_model.to(DEVICE)


class SegmentationResNet(nn.Module):
    """
    Wrapper class that combines ResNet backbone with segmentation head.
    """

    def __init__(self, backbone, in_features, num_classes, img_size):
        super().__init__()
        self.backbone = backbone
        self.segmentation_head = SegmentationHead(
            in_channels=in_features, num_classes=num_classes, img_size=img_size
        )

    def forward(self, x):
        # Extract features with backbone
        features = self.backbone(x)

        # Generate segmentation masks
        masks = self.segmentation_head(features)

        return masks
