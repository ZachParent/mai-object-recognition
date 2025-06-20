import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import DEVICE, VISUALIZATIONS_DIR
import torchmetrics
from typing import Tuple, List


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
) -> None:

    save_dir = VISUALIZATIONS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    i = 0

    for image, target in dataloader:
        # Move tensors to the correct device
        image = image.to(DEVICE)
        true_mask = target["labels"].to(DEVICE)

        # Get prediction
        with torch.no_grad():
            output = model(image)
            if isinstance(output, dict):
                if "out" in output:
                    output = output["out"]
                elif "logits" in output:
                    output = output["logits"]
            pred_mask = torch.argmax(output, dim=1).squeeze(0)

        # Convert tensors to numpy for visualization
        img_np = image.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
        true_mask_np = true_mask.squeeze(0).cpu().detach().numpy()
        pred_mask_np = pred_mask.cpu().detach().numpy()

        # Denormalize image if necessary (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Plot original image with true segmentation
        axes[0].imshow(img_np)
        im0 = axes[0].imshow(
            true_mask_np,
            alpha=0.5,
            cmap="viridis",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[0].set_title("Ground Truth Segmentation")
        axes[0].axis("off")

        # Plot original image with predicted segmentation
        axes[1].imshow(img_np)
        im1 = axes[1].imshow(
            pred_mask_np,
            alpha=0.5,
            cmap="viridis",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[1].set_title("Model Prediction")
        axes[1].axis("off")

        # Add colorbars with consistent limits
        fig.colorbar(
            im0,
            ax=axes[0],
            fraction=0.046,
            pad=0.04,
            ticks=range(num_classes),
        )
        fig.colorbar(
            im1,
            ax=axes[1],
            fraction=0.046,
            pad=0.04,
            ticks=range(num_classes),
        )

        plt.tight_layout()

        # Save figure
        output_path = save_dir / f"{i}_prediction.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        i += 1
        if i == 5:
            break

    print(f"Saved visualization images ({model._get_name()}) to {output_path}")
