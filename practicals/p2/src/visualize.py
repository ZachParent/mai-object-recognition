import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import DEVICE, VISUALIZATIONS_DIR
from dataset import get_aux_dataloader
from models import load_model_from_weights
from experiment_config import best_model_experiment, balancing_experiment
import torchmetrics
from typing import Tuple, List


def get_best_and_worst_images(
    model: torch.nn.Module, dataloader: DataLoader, num_classes: int
) -> Tuple[List[int], List[int]]:

    model.eval()

    dice_scores = []  # To store Dice scores for each image

    with torch.no_grad():
        for img_idx, (image, target) in enumerate(dataloader):
            # Move tensors to the correct device
            image = image.to(DEVICE)
            mask = target["labels"].to(DEVICE)

            # Forward pass
            outputs = model(image)  # Shape: [batch_size, num_classes, H, W]

            if "out" in outputs:  # DeepLabv3 and LR-ASPP
                outputs = outputs["out"]
            else:  # SegFormer
                outputs = outputs["logits"]
                mask = (
                    torch.nn.functional.interpolate(
                        mask.unsqueeze(1).float(),
                        scale_factor=1 / 4,
                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                )

            # Get predictions
            preds = outputs.argmax(dim=1)

            # Compute Dice scores for each image in the batch
            dice_metric = torchmetrics.segmentation.DiceScore(
                input_format="index",
                num_classes=num_classes,
                include_background=False,
                average="macro",
            )
            for i in range(image.size(0)):
                dice_score = dice_metric(
                    preds[i].unsqueeze(0), mask[i].unsqueeze(0)
                ).item()
                dice_scores.append((dice_score, img_idx))

    # Sort Dice scores to find the 5 worst-performing images
    dice_scores.sort(key=lambda x: x[0])  # Sort by Dice score (ascending)
    best_images = [
        index for _, index in dice_scores[-5:]
    ]  # Extract the indices of the 5 highest scores
    worst_images = [
        index for _, index in dice_scores[:5]
    ]  # Extract the indices of the 5 lowest scores
    return best_images, worst_images


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    worst_img_idxs: List[int],
    best_img_idxs: List[int],
    num_classes: int,
) -> None:
    dataset = dataloader.dataset

    # Create separate directories for worst and best visualizations
    worst_dir = VISUALIZATIONS_DIR / "worst"
    best_dir = VISUALIZATIONS_DIR / "best"
    worst_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    visualize = {
        "worst": (worst_img_idxs, worst_dir),
        "best": (best_img_idxs, best_dir),
    }

    for set_name, (img_idxs, save_dir) in visualize.items():
        for idx in img_idxs:
            # Get the image and target by index
            img, target = dataset[idx]

            # Move tensors to the correct device
            image = img.to(DEVICE)
            true_mask = target["labels"].to(DEVICE)

            # Get prediction
            with torch.no_grad():
                output = model(image.unsqueeze(0))
                if isinstance(output, dict):
                    if "out" in output:
                        output = output["out"]
                    elif "logits" in output:
                        output = output["logits"]
                pred_mask = torch.argmax(output, dim=1).squeeze(0)

            # Resize predicted mask to match the true mask's shape
            pred_mask_resized = (
                torch.nn.functional.interpolate(
                    pred_mask.unsqueeze(0)
                    .unsqueeze(0)
                    .float(),  # Add batch and channel dimensions
                    size=true_mask.shape,  # Resize to match true mask shape
                    mode="nearest",  # Use nearest neighbor interpolation for segmentation masks
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )  # Remove batch and channel dimensions

            # Convert tensors to numpy for visualization
            img_np = img.cpu().detach().permute(1, 2, 0).numpy()
            true_mask_np = true_mask.cpu().detach().numpy()
            pred_mask_np = pred_mask_resized.cpu().detach().numpy()

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
            output_path = save_dir / f"prediction_idx_{idx}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(
                f"Saved {set_name} visualization for image index {idx} to {output_path}"
            )

def main():
    num_classes = 28
    model = load_model_from_weights(best_model_experiment)
    aux_dataloader = get_aux_dataloader(best_model_experiment)
    worst_performing_images, best_performing_images = get_best_and_worst_images(
        model, aux_dataloader, num_classes
    )
    visualize_predictions(
        model=model,
        dataloader=aux_dataloader,
        worst_img_idxs=worst_performing_images,
        best_img_idxs=best_performing_images,
        num_classes=num_classes,
    )

if __name__ == "__main__":
    main()