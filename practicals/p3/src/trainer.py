import os
import random
from pathlib import Path

import numpy as np
import torch
from config import CHECKPOINTS_DIR
from datasets.dummy import get_dummy_dataloader
from metrics import MAE, MetricCollection, MetricLogger, PerceptualLoss
from models import get_model
from models.unet2d import UNet2D
from run_configs import ModelName, RunConfig, UNet2DConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


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
            progress.update(loss.item())

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
                progress.update(loss.item())

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
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
) -> None:
    # Set random seed for reproducibility
    if config.seed is not None:
        set_seed(config.seed)

    # Initialize model
    model = get_model(config)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()  # MSE loss for depth estimation

    # Initialize metrics
    train_metric_collection = MetricCollection(
        {"mae": MAE(), "perceptual": PerceptualLoss()}
    )
    val_metric_collection = MetricCollection(
        {"mae": MAE(), "perceptual": PerceptualLoss()}
    )

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
        Path(".tmp/metrics.csv"),
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

        # Save model if path is provided
        if config.save_path:
            save_dir = config.save_path / f"run_{config.id}"
            trainer.save_model(str(save_dir / f"epoch_{epoch+1}.pt"))

    test_metric_collection = MetricCollection(
        {"mae": MAE(), "perceptual": PerceptualLoss()}
    )
    trainer.evaluate(test_dataloader, test_metric_collection)

    print(f"Test MAE: {test_metric_collection.metrics['mae'].compute()}")
    print(f"Test Perceptual: {test_metric_collection.metrics['perceptual'].compute()}")

    metrics_logger.save_metrics()


if __name__ == "__main__":
    # Example usage
    config = RunConfig(
        id=0,
        model_name=ModelName.UNET2D,
        learning_rate=1e-2,
        batch_size=1,
        epochs=2,
        save_path=CHECKPOINTS_DIR,
        unet2d_config=UNet2DConfig(),
        seed=42,
    )

    # Create dummy dataloaders for testing
    torch.manual_seed(config.seed)
    train_dataloader = get_dummy_dataloader(config.batch_size)
    val_dataloader = get_dummy_dataloader(config.batch_size)
    test_dataloader = get_dummy_dataloader(config.batch_size)

    run_experiment(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
    )
