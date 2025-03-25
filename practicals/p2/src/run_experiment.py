from experiment_config import ExperimentConfig
from models import get_model
from dataset import get_dataloaders, NUM_CLASSES
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from config import NUM_EPOCHS, DEVICE
from metrics import MetricLogger, get_metric_collection
import torchmetrics


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
        self.train_metrics_collection = train_metrics_collection
        self.val_metrics_collection = val_metrics_collection

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        progress = TrainingProgress(
            dataloader,
            desc="Training",
        )
        self.train_metrics_collection.reset()

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

            preds = outputs.argmax(dim=1)
            self.train_metrics_collection.update(preds, mask)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Update progress with current batch metrics
            progress.update(loss.item())

        progress.close()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        progress = TrainingProgress(dataloader, desc="Evaluating")
        self.val_metrics_collection.reset()

        with torch.no_grad():
            for image, target in dataloader:
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

                preds = outputs.argmax(dim=1)
                self.val_metrics_collection.update(preds, mask)

                # Update progress with current batch metrics
                progress.update(loss.item())

        progress.close()


def run_experiment(experiment: ExperimentConfig) -> None:
    train_dataloader, val_dataloader = get_dataloaders(experiment)
    train_metrics_collection = get_metric_collection(NUM_CLASSES)
    val_metrics_collection = get_metric_collection(NUM_CLASSES)

    trainer = Trainer(experiment, train_metrics_collection, val_metrics_collection)
    metrics_logger = MetricLogger(
        experiment.id, trainer.train_metrics_collection, trainer.val_metrics_collection
    )
    for epoch in range(NUM_EPOCHS):
        width = 90
        print("\n" + "=" * width)
        print(f"EPOCH {epoch+1} / {NUM_EPOCHS}".center(width))
        print("-" * width)
        trainer.train_epoch(train_dataloader)
        trainer.evaluate(val_dataloader)

        # Log metrics to TensorBoard and CSV (will also print epoch summary)
        metrics_logger.update_metrics()
        metrics_logger.log_metrics()

    metrics_logger.close()


# Use this to run a quick test
if __name__ == "__main__":
    experiment = ExperimentConfig(
        id=0,
        model_name="segformer",
        learning_rate=0.001,
        batch_size=4,
        img_size=224,
    )
    run_experiment(experiment)
