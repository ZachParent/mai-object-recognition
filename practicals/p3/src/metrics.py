from typing import Dict

import torch


class MetricLogger:
    def __init__(self, experiment_id: int, train_metrics: Dict, val_metrics: Dict):
        self.experiment_id = experiment_id
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_perceptual_losses = []
        self.val_perceptual_losses = []

    def update_metrics(self, train_loss: float, val_loss: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Update MAE and perceptual loss from metrics
        self.train_maes.append(self.train_metrics["mae"].compute())
        self.val_maes.append(self.val_metrics["mae"].compute())
        self.train_perceptual_losses.append(self.train_metrics["perceptual"].compute())
        self.val_perceptual_losses.append(self.val_metrics["perceptual"].compute())

    def log_metrics(self):
        print(f"\nEpoch Summary:")
        print(f"Train Loss (MSE): {self.train_losses[-1]:.4f}")
        print(f"Val Loss (MSE): {self.val_losses[-1]:.4f}")
        print(f"Train MAE: {self.train_maes[-1]:.4f}")
        print(f"Val MAE: {self.val_maes[-1]:.4f}")
        print(f"Train Perceptual Loss: {self.train_perceptual_losses[-1]:.4f}")
        print(f"Val Perceptual Loss: {self.val_perceptual_losses[-1]:.4f}")

    def close(self):
        pass  # Add any cleanup if needed


def get_metric_collection() -> Dict:
    """Create a collection of metrics for depth estimation evaluation."""
    return {"mae": MAE(), "perceptual": PerceptualLoss()}


class MAE:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_abs_error = 0.0
        self.total_pixels = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target should be [B, 1, H, W] tensors
        abs_error = torch.abs(preds - target)
        self.total_abs_error += abs_error.sum().item()
        self.total_pixels += target.numel()

    def compute(self) -> float:
        return (
            self.total_abs_error / self.total_pixels if self.total_pixels > 0 else 0.0
        )


# TODO(@Bruno): Implement perceptual loss
class PerceptualLoss:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        # Garbage loss
        loss = torch.mean((preds / sum(preds) - target / sum(target)) ** 2)
        self.total_loss += loss
        self.num_batches += 1

    def compute(self) -> float:
        return self.total_loss / self.num_batches if self.num_batches > 0 else 0.0
