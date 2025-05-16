import os
from typing import Dict, Optional

import torch
from config import CHECKPOINTS_DIR
from metrics import MetricLogger, get_metric_collection
from models.unet2d import UNet2D
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
        train_metrics: Dict,
        val_metrics: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.device = device

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        progress = TrainingProgress(dataloader, desc="Training")

        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()

        total_loss = 0.0
        num_batches = 0

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
            for metric in self.train_metrics.values():
                metric.update(pred_depth, depth)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Update progress
            progress.update(loss.item())

        progress.close()

        # Return average loss for the epoch
        return total_loss / num_batches

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        progress = TrainingProgress(dataloader, desc="Evaluating")

        # Reset metrics
        for metric in self.val_metrics.values():
            metric.reset()

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
                for metric in self.val_metrics.values():
                    metric.update(pred_depth, depth)

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


def run_experiment(
    model: UNet2D,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    experiment_id: int,
    learning_rate: float = 1e-4,
    epochs: int = 100,
    save_path: Optional[str] = None,
) -> None:
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()  # MSE loss for depth estimation

    # Initialize metrics
    train_metrics = get_metric_collection()
    val_metrics = get_metric_collection()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )

    # Initialize metric logger
    metrics_logger = MetricLogger(experiment_id, train_metrics, val_metrics)

    # Training loop
    for epoch in range(epochs):
        width = 90
        print("\n" + "=" * width)
        print(f"EPOCH {epoch+1} / {epochs}".center(width))
        print("-" * width)

        # Train and evaluate
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss = trainer.evaluate(val_dataloader)

        # Log metrics
        metrics_logger.update_metrics(train_loss, val_loss)
        metrics_logger.log_metrics()

        # Save model if path is provided
        if save_path:
            trainer.save_model(f"{save_path}/epoch_{epoch+1}.pt")

    metrics_logger.close()


if __name__ == "__main__":
    # Example usage
    model = UNet2D(
        input_size=(256, 256, 3),
        filter_num=[64, 128, 256, 512],
        n_labels=1,  # Single channel for depth
    )

    # Create dummy dataloaders for testing
    train_dataloader = DataLoader(
        [(torch.randn(3, 256, 256), torch.randn(1, 256, 256))], batch_size=1
    )
    val_dataloader = DataLoader(
        [(torch.randn(3, 256, 256), torch.randn(1, 256, 256))], batch_size=1
    )

    run_experiment(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        experiment_id=0,
        learning_rate=1e-4,
        epochs=10,
        save_path=CHECKPOINTS_DIR / "demo_model",
    )
