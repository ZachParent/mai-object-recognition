from experiment_config import ExperimentConfig
from models import get_model
from dataset import get_dataloaders, NUM_CLASSES
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DEVICE
from metrics import MetricLogger, get_metric_collection
import torchmetrics
from config import MODELS_DIR
import numpy as np
import matplotlib.pyplot as plt
import os


class TrainingProgress:
    def __init__(
        self,
        dataloader: DataLoader,
        loss: float = 0.0,
        desc: str = "",
    ) -> None:
        self.pbar = tqdm(total=len(dataloader), desc=desc)
        self.loss = loss
        self.desc = desc.ljust(10)

        # Format description
        self._update_description()
        self.pbar.update(0)  # Don't increment on init

    def _update_description(self) -> None:
        self.pbar.set_description(f"{self.desc} | loss: {self.loss:.4f}")

    def update(self, loss: float) -> None:
        self.loss = loss
        self._update_description()
        self.pbar.update(1)

    def close(self) -> None:
        self.pbar.close()


class Trainer:
    def __init__(
        self,
        experiment: ExperimentConfig,
        train_metrics_collection: torchmetrics.MetricCollection,
        val_metrics_collection: torchmetrics.MetricCollection,
    ) -> None:
        self.experiment = experiment
        # Get a sample from the dataloader to determine num_classes
        train_dataloader, _ = get_dataloaders(experiment)
        sample_image, sample_target = next(iter(train_dataloader))

        # Fix: Convert tensor to integer for num_classes
        num_classes = sample_target["num_classes"]
        if isinstance(num_classes, torch.Tensor):
            # If it's a tensor, take the first value and convert to int
            num_classes = int(num_classes[0].item())
        elif not isinstance(num_classes, int):
            # If it's not an int or tensor, try to convert it
            num_classes = int(num_classes)

        print(f"Using {num_classes} classes for segmentation model")

        # Create model with correct parameters
        self.model = get_model(
            model_name=experiment.model_name,
            num_classes=num_classes,
            img_size=experiment.img_size,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=experiment.learning_rate
        )
        # CrossEntropyLoss for multi-class segmentation
        self.criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
        self.train_metrics_collection = train_metrics_collection.to(DEVICE)
        self.val_metrics_collection = val_metrics_collection.to(DEVICE)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        progress = TrainingProgress(
            dataloader,
            desc="Training",
        )
        self.train_metrics_collection.reset()

        total_loss = 0.0
        num_batches = 0

        for image, target in dataloader:
            # Move tensors to the correct device
            image = image.to(DEVICE)
            mask = target["labels"].to(DEVICE)

            # Clear gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(image)  # Shape: [batch_size, num_classes, H, W]

            if "out" in outputs:  # DeepLabv3 and LR-ASPP
                outputs = outputs["out"]
            else:  # SegFormer
                outputs = outputs["logits"]
                mask = (
                    torch.nn.functional.interpolate(
                        mask.unsqueeze(1).float(), scale_factor=1 / 4, mode="nearest"
                    )
                    .squeeze(1)
                    .long()
                )

            # Calculate loss - CrossEntropyLoss expects [B, C, H, W] outputs and [B, H, W] targets
            loss = self.criterion(outputs, mask)

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            preds = outputs.argmax(dim=1)
            self.train_metrics_collection.update(preds, mask)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Update progress with current batch metrics
            progress.update(loss.item())

        progress.close()

        # Return average loss for the epoch
        return total_loss / num_batches

    def evaluate(self, dataloader: DataLoader) -> tuple[float, list]:
        self.model.eval()
        progress = TrainingProgress(dataloader, desc="Evaluating")
        self.val_metrics_collection.reset()

        total_loss = 0.0
        num_batches = 0

        dice_scores = []  # To store Dice scores for each image
        worst_images = []  # To store information about the worst-performing images

        with torch.no_grad():
            for img_idx, (image, target) in enumerate(dataloader):
                # Move tensors to the correct device
                image = image.to(DEVICE)
                mask = target["labels"].to(DEVICE)

                # Forward pass
                outputs = self.model(image)  # Shape: [batch_size, num_classes, H, W]

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

                # Calculate loss - CrossEntropyLoss expects [B, C, H, W] outputs and [B, H, W] targets
                loss = self.criterion(outputs, mask)

                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1

                preds = outputs.argmax(dim=1)
                self.val_metrics_collection.update(preds, mask)

                dice_metric = torchmetrics.segmentation.DiceScore(
                    input_format="index", num_classes=NUM_CLASSES
                )
                dice_score = dice_metric(preds.unsqueeze(0), mask.unsqueeze(0)).item()
                dice_scores.append((dice_score, img_idx))

                # Update progress with current batch metrics
                progress.update(loss.item())

        progress.close()

        dice_scores.sort(key=lambda x: x[0])  # Sort by Dice score (ascending)
        worst_images = [
            index for _, index in dice_scores[:5]
        ]  # Extract the indices of the 5 lowest scores

        print("Indices of the 5 worst-performing images:", worst_images)

        # Return average loss for the epoch
        return total_loss / num_batches, worst_images

    def visualize_lowest_dice_predictions(
        self, dataloader=None, output_dir="visualizations", worst_img_idxs=None
    ) -> None:

        # Access the dataset directly from the dataloader
        dataset = dataloader.dataset

        for idx in worst_img_idxs:

            # Get the image and target by index
            img, target = dataset[idx]

            # Move tensors to the correct device
            image = img.to(DEVICE)
            true_mask = target["labels"].to(DEVICE)

            # Get prediction
            with torch.no_grad():
                output = self.model(image.unsqueeze(0))
                if isinstance(output, dict):
                    if "out" in output:
                        output = output["out"]
                    elif "logits" in output:
                        output = output["logits"]
                pred_mask = torch.argmax(output, dim=1).squeeze(0)

            # Convert tensors to numpy for visualization
            img_np = img.cpu().detach().permute(1, 2, 0).numpy()
            true_mask_np = true_mask.cpu().detach().numpy()
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
            im0 = axes[0].imshow(true_mask_np, alpha=0.5, cmap="viridis")
            axes[0].set_title("Ground Truth Segmentation")
            axes[0].axis("off")

            # Plot original image with predicted segmentation
            axes[1].imshow(img_np)
            im1 = axes[1].imshow(pred_mask_np, alpha=0.5, cmap="viridis")
            axes[1].set_title("Model Prediction")
            axes[1].axis("off")

            # Add colorbars
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()

            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"prediction_idx_{idx}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

    def save_model(self) -> None:
        """Save the model weights to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = f"{MODELS_DIR}/{self.experiment.model_name}_lr{self.experiment.learning_rate}_img{self.experiment.img_size}.pt"

        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


def run_experiment(experiment: ExperimentConfig) -> None:
    train_dataloader, val_dataloader = get_dataloaders(experiment)
    train_metrics_collection = get_metric_collection(NUM_CLASSES)
    val_metrics_collection = get_metric_collection(NUM_CLASSES)

    trainer = Trainer(experiment, train_metrics_collection, val_metrics_collection)
    metrics_logger = MetricLogger(
        experiment.id, trainer.train_metrics_collection, trainer.val_metrics_collection
    )

    for epoch in range(experiment.epochs):
        width = 90
        print("\n" + "=" * width)
        print(f"EPOCH {epoch+1} / {experiment.epochs}".center(width))
        print("-" * width)
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss, worst_img_idxs = trainer.evaluate(val_dataloader)

        # Log metrics to TensorBoard and CSV (will also print epoch summary)
        metrics_logger.update_metrics(train_loss, val_loss)
        metrics_logger.log_metrics()
        if epoch == experiment.epochs - 1 and experiment.visualize:
            trainer.visualize_lowest_dice_predictions(
                dataloader=val_dataloader, worst_img_idxs=worst_img_idxs
            )

    metrics_logger.save_val_confusion_matrix()
    metrics_logger.close()

    if experiment.save_weights:
        trainer.save_model()


# Use this to run a quick test
if __name__ == "__main__":
    experiment = ExperimentConfig(
        id=0,
        model_name="segformer",
        learning_rate=0.001,
        batch_size=2,
        img_size=100,
    )
    run_experiment(experiment)
