from experiment_config import ExperimentConfig
from models import get_model
from dataset import get_dataloaders
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
from config import NUM_EPOCHS, DEVICE
import numpy as np
from metrics import ALL_METRICS, MetricsLogger


class TrainingProgress:
    def __init__(
        self,
        dataloader: DataLoader,
        desc: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        self.pbar = tqdm(total=len(dataloader), desc=desc)
        self.metrics = metrics or {}  # Dict to track moving averages
        self.desc = desc.ljust(10)

        # Format description
        if self.metrics:
            self._set_description(self.metrics)
        self.pbar.update(0)  # Don't increment on init

    def _set_description(self, metrics: Dict[str, float]) -> None:
        metrics_str = " | ".join(
            f"{name}: {value:.4f}" for name, value in metrics.items()
        )
        self.pbar.set_description(f"{self.desc} | {metrics_str}")

    def update(self, metrics: Dict[str, float]) -> None:
        # Replace metrics instead of updating
        self.metrics = metrics

        # Update progress bar description
        self._set_description(self.metrics)
        self.pbar.update(1)

    def close(self) -> None:
        self.pbar.close()


class Trainer:
    def __init__(self, experiment: ExperimentConfig) -> None:
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

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        progress = TrainingProgress(dataloader, desc="Training")

        # Initialize accumulators for all metrics
        total_metrics = {
            "loss": 0.0,
        }
        # Initialize accumulators for all metrics from ALL_METRICS
        for metric in ALL_METRICS:
            total_metrics[metric.name] = 0.0

        # Keep track of batch count
        batch_count = 0

        # Track predictions and targets for accumulating metrics that need the entire dataset
        all_outputs = []
        all_targets = []

        for image, target in dataloader:
            # Move tensors to the correct device
            image = image.to(DEVICE)
            mask = target["labels"].to(DEVICE)

            # Clear gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(image)  # Shape: [batch_size, num_classes, H, W]

            # Calculate loss - CrossEntropyLoss expects [B, C, H, W] outputs and [B, H, W] targets
            loss = self.criterion(outputs, mask)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Store batch outputs and targets for metrics calculation
            all_outputs.append(outputs.detach())
            all_targets.append(mask.detach())

            # Update accumulators
            batch_loss = loss.item()
            total_metrics["loss"] += batch_loss
            batch_count += 1

            # Calculate batch metrics for progress display
            batch_metrics = {
                "loss": batch_loss,
            }

            # Calculate metrics for this batch
            for metric in ALL_METRICS:
                metric_value = metric(outputs, mask)
                total_metrics[metric.name] += metric_value
                batch_metrics[metric.name] = metric_value

            # Update progress with current batch metrics
            progress.update(batch_metrics)

        # Calculate average metrics across all batches
        avg_metrics = {}
        for key, value in total_metrics.items():
            avg_metrics[key] = value / batch_count if batch_count > 0 else 0

        # Calculate any metrics that need the entire dataset's predictions and targets
        # (For now, we're not using any such metrics, but this is where they would go)

        progress.close()
        return avg_metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        progress = TrainingProgress(dataloader, desc="Evaluating")

        # Initialize accumulators for all metrics
        total_metrics = {
            "loss": 0.0,
        }
        # Initialize accumulators for all metrics from ALL_METRICS
        for metric in ALL_METRICS:
            total_metrics[metric.name] = 0.0

        # Keep track of batch count
        batch_count = 0

        # Track predictions and targets for accumulating metrics that need the entire dataset
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for image, target in dataloader:
                # Move tensors to the correct device
                image = image.to(DEVICE)
                mask = target["labels"].to(DEVICE)

                # Forward pass
                outputs = self.model(image)  # Shape: [batch_size, num_classes, H, W]

                # Calculate loss
                loss = self.criterion(outputs, mask)

                # Store batch outputs and targets for metrics calculation
                all_outputs.append(outputs.detach())
                all_targets.append(mask.detach())

                # Update accumulators
                batch_loss = loss.item()
                total_metrics["loss"] += batch_loss
                batch_count += 1

                # Calculate batch metrics for progress display
                batch_metrics = {
                    "loss": batch_loss,
                }

                # Calculate metrics for this batch
                for metric in ALL_METRICS:
                    metric_value = metric(outputs, mask)
                    total_metrics[metric.name] += metric_value
                    batch_metrics[metric.name] = metric_value

                # Update progress with current batch metrics
                progress.update(batch_metrics)

        # Calculate average metrics across all batches
        avg_metrics = {}
        for key, value in total_metrics.items():
            avg_metrics[key] = value / batch_count if batch_count > 0 else 0

        # Calculate any metrics that need the entire dataset's predictions and targets
        # (For now, we're not using any such metrics, but this is where they would go)

        progress.close()
        return avg_metrics


def run_experiment(experiment: ExperimentConfig) -> None:
    train_dataloader, val_dataloader = get_dataloaders(experiment)

    trainer = Trainer(experiment)
    metrics_logger = MetricsLogger(experiment.id)
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        train_metrics = trainer.train_epoch(train_dataloader)
        val_metrics = trainer.evaluate(val_dataloader)
        metrics_logger.log_metrics(train_metrics, val_metrics, epoch)
    metrics_logger.close()


# Use this to run a quick test
if __name__ == "__main__":
    experiment = ExperimentConfig(
        id=0,
        model_name="resnet18",
        learning_rate=0.001,
        batch_size=16,
        img_size=224,
    )
    run_experiment(experiment)
