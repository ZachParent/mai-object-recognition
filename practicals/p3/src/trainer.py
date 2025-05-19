import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CHECKPOINTS_DIR, RESULTS_DIR
from .datasets.cloth3d import Cloth3dDataset
from .metrics import (
    CombinedLoss,
    MetricCollection,
    MetricLogger,
    get_metric_collection,
)
from .models import get_model
from .models.unet2d import UNet2D
from .run_configs import ModelName, RunConfig, UNet2DConfig


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
        self._update_description()
        self.pbar.update(0)

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
        model: UNet2D,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_metric_collection: MetricCollection,
        val_metric_collection: MetricCollection,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_metric_collection = train_metric_collection
        self.val_metric_collection = val_metric_collection
        self.device = device

    def train_epoch(
        self, dataloader: DataLoader, metric_collection: MetricCollection
    ) -> float:
        self.model.train()
        progress = TrainingProgress(dataloader, desc="Training")

        total_loss = 0.0
        num_batches = 0
        metric_collection.reset()

        for image, target in dataloader:
            # Move tensors to the correct device
            image = image.to(self.device)
            depth = target.to(self.device)  # [B, 1, H, W]

            # Clear gradients
            self.optimizer.zero_grad()

            # Forward pass
            pred_depth = self.model(image)  # [B, 1, H, W]

            # Calculate loss
            loss = self.criterion(pred_depth, depth)

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Update metrics
            metric_collection.update(pred_depth.detach(), depth)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Update progress
            progress.update(total_loss / num_batches)

        progress.close()

        # Return average loss for the epoch
        return total_loss / num_batches

    def evaluate(
        self, dataloader: DataLoader, metric_collection: MetricCollection
    ) -> float:
        self.model.eval()
        progress = TrainingProgress(dataloader, desc="Evaluating")

        # Reset metrics
        metric_collection.reset()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for image, target in dataloader:
                # Move tensors to the correct device
                image = image.to(self.device)
                depth = target.to(self.device)  # [B, 1, H, W]

                # Forward pass
                pred_depth = self.model(image)  # [B, 1, H, W]

                # Calculate loss
                loss = self.criterion(pred_depth, depth)

                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1

                # Update metrics
                metric_collection.update(pred_depth.detach(), depth)

                # Update progress
                progress.update(total_loss / num_batches)

        progress.close()

        # Return average loss for the epoch
        return total_loss / num_batches

    def save_model(self, path: str) -> None:
        """Save the model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load the model weights from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    config: RunConfig,
) -> None:
    # Set random seed for reproducibility
    if config.seed is not None:
        set_seed(config.seed)

    # Initialize dataloaders
    train_dataloader = DataLoader(
        # TODO: enable augmentation
        dataset=Cloth3dDataset(start_idx=0, end_idx=128, enable_augmentation=False),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=Cloth3dDataset(
            start_idx=128, end_idx=128 + 16, enable_augmentation=False
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=Cloth3dDataset(
            start_idx=128 + 16, end_idx=None, enable_augmentation=False
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Initialize model
    model = get_model(config)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.perceptual_loss_weight is not None:
        criterion = CombinedLoss(
            discrepancy_error=config.perceptual_loss,
            weight=config.perceptual_loss_weight,
        )
    else:
        criterion = torch.nn.MSELoss()

    # Initialize metrics
    train_metric_collection = get_metric_collection(config)
    val_metric_collection = get_metric_collection(config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_metric_collection=train_metric_collection,
        val_metric_collection=val_metric_collection,
    )

    # Initialize metric logger
    metrics_logger = MetricLogger(
        config.id,
        train_metric_collection,
        val_metric_collection,
    )

    # Training loop
    for epoch in range(config.epochs):
        width = 90
        print("\n" + "=" * width)
        print(f"EPOCH {epoch+1} / {config.epochs}".center(width))
        print("-" * width)

        # Train and evaluate
        trainer.train_epoch(train_dataloader, train_metric_collection)
        trainer.evaluate(val_dataloader, val_metric_collection)

        # Log metrics
        metrics_logger.log_metrics()

    test_metric_collection = get_metric_collection(config)
    trainer.evaluate(test_dataloader, test_metric_collection)

    for metric_name, metric in test_metric_collection.metrics.items():
        print(f"Test {metric_name.upper()}: {metric.compute()}")

    metrics_logger.save_metrics(RESULTS_DIR / f"run_{config.id}")
    test_df = pd.DataFrame(test_metric_collection.compute(), index=[0])
    test_df.to_csv(RESULTS_DIR / f"run_{config.id}" / "test.csv", index=False)

    # Save model if path is provided
    if config.save_path:
        trainer.save_model(str(config.save_path / f"run_{config.id}.pt"))


if __name__ == "__main__":
    # Example usage
    config = RunConfig(
        id=0,
        name="demo",
        model_name=ModelName.UNET2D,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10,
        save_path=CHECKPOINTS_DIR,
        unet2d_config=UNet2DConfig(),
        seed=42,
        perceptual_loss="L2",
        perceptual_loss_weight=0.5,
    )

    run_experiment(config=config)
