import abc
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


class Metric(abc.ABC):
    def __init__(self):
        self.values = []
        self.reset()

    @abc.abstractmethod
    def reset(self):
        self.values = []

    @abc.abstractmethod
    def update(self, preds: torch.Tensor, target: torch.Tensor): ...

    @abc.abstractmethod
    def compute(self) -> float:
        return np.sum(self.values) / len(self.values)


class MetricCollection(Metric):
    def __init__(self, metrics: Dict[str, Metric]):
        self.metrics = metrics
        self.reset()

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for metric in self.metrics.values():
            metric.update(preds, target)

    def compute(self) -> Dict[str, float]:
        return {name: metric.compute() for name, metric in self.metrics.items()}


class MetricTracker(Metric):
    def __init__(self, metrics: Dict[str, Metric], **kwargs):
        self.count = 0
        self.metrics = metrics
        self.reset()

    def increment(self) -> None:
        for metric_name in self.metrics.keys():
            current_value = self.metrics[metric_name].compute()
            self.values[metric_name].append(current_value)
            self.metrics[metric_name].reset()
        self.count += 1

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with new predictions and targets."""
        for metric in self.metrics.values():
            metric.update(preds, target)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute final metric values."""
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()
        return results

    def get_values(self) -> Dict[str, List[float]]:
        return self.values

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
        self.values = {metric_name: [] for metric_name in self.metrics.keys()}
        self.count = 0


class MetricLogger:
    def __init__(
        self,
        experiment_id: int,
        train_metric_collection: MetricCollection,
        val_metric_collection: MetricCollection,
        csv_path: Path,
    ):
        self.experiment_id = experiment_id
        # Use MetricCollection to group metrics, and MetricTracker to track them
        self.train_tracker = MetricTracker(train_metric_collection.metrics)
        self.val_tracker = MetricTracker(val_metric_collection.metrics)
        self.csv_path = csv_path

    def log_metrics(self):
        train_metrics = self.train_tracker.compute()
        val_metrics = self.val_tracker.compute()
        self.train_tracker.increment()
        self.val_tracker.increment()
        print(f"\nEpoch Summary:")
        for name in train_metrics:
            print(f"Train {name.upper()}: {train_metrics[name]:.4f}")
            print(f"Val {name.upper()}: {val_metrics[name]:.4f}")

    def save_metrics(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame()
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        f"train_{metric}": self.train_tracker.get_values()[metric]
                        for metric in self.train_tracker.get_values().keys()
                    }
                ),
            ],
            axis=1,
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        f"val_{metric}": self.val_tracker.get_values()[metric]
                        for metric in self.val_tracker.get_values().keys()
                    }
                ),
            ],
            axis=1,
        )
        df.to_csv(self.csv_path, index=False)


def get_metric_collection() -> Dict:
    """Create a collection of metrics for depth estimation evaluation."""
    return {"mae": MAE(), "perceptual": PerceptualLoss()}


class MAE(Metric):
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
class PerceptualLoss(Metric):
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


if __name__ == "__main__":
    mc = MetricCollection({"mae": MAE(), "perceptual": PerceptualLoss()})
    tracker = MetricTracker(mc.metrics)
    tracker.update(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))
    tracker.increment()
    tracker.update(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))
    tracker.increment()
    print(tracker.get_values())

    train_mc = MetricCollection({"mae": MAE(), "perceptual": PerceptualLoss()})
    val_mc = MetricCollection({"mae": MAE(), "perceptual": PerceptualLoss()})
    logger = MetricLogger(0, train_mc, val_mc, Path(".tmp/metrics.csv"))
    train_mc.update(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))

    train_mc.update(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))
    val_mc.update(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))
    logger.log_metrics()
    logger.save_metrics()
