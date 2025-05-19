import abc
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch

from .run_configs import ModelName, RunConfig


class Metric(abc.ABC):
    def __init__(self):
        self.values = []
        self.reset()

    def reset(self):
        self.values = []

    @abc.abstractmethod
    def update(self, preds: torch.Tensor, target: torch.Tensor): ...

    def compute(self) -> float:
        return np.sum(self.values) / len(self.values) if self.values else 0.0


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
    ):
        self.experiment_id = experiment_id
        # Use MetricCollection to group metrics, and MetricTracker to track them
        self.train_tracker = MetricTracker(train_metric_collection.metrics)
        self.val_tracker = MetricTracker(val_metric_collection.metrics)

    def log_metrics(self):
        train_metrics = self.train_tracker.compute()
        val_metrics = self.val_tracker.compute()
        self.train_tracker.increment()
        self.val_tracker.increment()
        print(f"\nEpoch Summary:")
        for name in train_metrics:
            print(f"Train {name.upper()}: {train_metrics[name]:.4f}")
            print(f"Val {name.upper()}: {val_metrics[name]:.4f}")

    def save_metrics(self, csv_dir_path: Path):
        csv_dir_path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.train_tracker.get_values())
        df.to_csv(csv_dir_path / f"train_{self.experiment_id}.csv", index=False)
        df = pd.DataFrame(self.val_tracker.get_values())
        df.to_csv(csv_dir_path / f"val_{self.experiment_id}.csv", index=False)


class MAE(Metric):
    def __init__(self):
        self.loss_fn = torch.nn.L1Loss(reduction="mean")
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target should be [B, 1, H, W] tensors
        self.values.append(self.loss_fn(preds, target).item())


class MSE(Metric):
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.values.append(self.loss_fn(preds, target).item())


class PerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        discrepancy_error: Literal["L1", "L2"] = "L2",
    ) -> None:
        super().__init__()
        if discrepancy_error not in ["L1", "L2"]:
            raise ValueError("discrepancy_error must be 'L1' or 'L2'")
        self.discrepancy_error = discrepancy_error

    @staticmethod
    def _compute_normal_map(depth_map: torch.Tensor) -> torch.Tensor:
        # depth_map shape: [B, 1, H, W]
        # Ensure kernel is on the same device and dtype as depth_map
        kernel_dtype = depth_map.dtype
        kernel_device = depth_map.device

        # Sobel kernels for X and Y gradients
        sobel_x_kernel = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=kernel_dtype,
                device=kernel_device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y_kernel = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=kernel_dtype,
                device=kernel_device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Compute gradients using 2D convolution
        gx = torch.nn.functional.conv2d(
            depth_map, sobel_x_kernel, padding=1
        )  #  [B, 1, H, W]
        gy = torch.nn.functional.conv2d(
            depth_map, sobel_y_kernel, padding=1
        )  #  [B, 1, H, W]

        # Create the nz component for the normal vector (gx, gy, nz)
        nz = torch.ones_like(gx)  #  [B, 1, H, W]

        # Concatenate gx, gy, nz along the channel dimension (dim=1) to form normal vectors
        normals = torch.cat((gx, gy, nz), dim=1)  #  [B, 3, H, W]

        # Normalize the normal vectors to unit length along the channel dimension (dim=1)
        normals = torch.nn.functional.normalize(normals, p=2, dim=1, eps=1e-6)

        return normals

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds and target shape: [B, 1, H, W]

        # Compute normal maps from the predicted and target depth images
        normal_preds = self._compute_normal_map(preds)
        normal_target = self._compute_normal_map(target)

        # Calculate the discrepancy error between the normal maps
        if self.discrepancy_error == "L1":
            perceptual_loss = torch.mean(torch.abs(normal_preds - normal_target))
        else:
            perceptual_loss = torch.mean((normal_preds - normal_target) ** 2)

        return perceptual_loss


class CombinedLoss(torch.nn.Module):
    def __init__(
        self, discrepancy_error: Literal["L1", "L2"] = "L2", weight: float = 0.5
    ):
        super().__init__()
        self.weight = weight
        self.mse_loss = torch.nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(discrepancy_error)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        perceptual_loss = self.perceptual_loss(preds, target)
        mse_loss = self.mse_loss(preds, target)
        return self.weight * perceptual_loss + (1 - self.weight) * mse_loss


class PerceptualLossMetric(Metric):
    def __init__(self, perceptual_loss: Literal["L1", "L2"] = "L2"):
        super().__init__()
        self.loss_fn = PerceptualLoss(perceptual_loss)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.values.append(self.loss_fn(preds, target).item())


class CombinedLossMetric(Metric):
    def __init__(
        self, perceptual_loss: Literal["L1", "L2"] = "L2", weight: float = 0.5
    ):
        super().__init__()
        self.loss_fn = CombinedLoss(perceptual_loss, weight)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.values.append(self.loss_fn(preds, target).item())


def get_metric_collection(run_config: RunConfig) -> MetricCollection:
    loss = (
        CombinedLossMetric(
            run_config.perceptual_loss, run_config.perceptual_loss_weight
        )
        if run_config.perceptual_loss_weight is not None
        else MSE()
    )
    return MetricCollection(
        {
            "mae": MAE(),
            "mse": MSE(),
            "perceptual_l2": PerceptualLossMetric("L2"),
            "perceptual_l1": PerceptualLossMetric("L1"),
            "loss": loss,
        }
    )


if __name__ == "__main__":
    run_config = RunConfig(
        id=0,
        model_name=ModelName.UNET2D,
        learning_rate=0.001,
        perceptual_loss="L2",
        perceptual_loss_weight=0.5,
    )

    train_mc = get_metric_collection(run_config)
    val_mc = get_metric_collection(run_config)
    logger = MetricLogger(0, train_mc, val_mc)
    train_mc.update(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))

    train_mc.update(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))
    val_mc.update(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))
    logger.log_metrics()
    logger.save_metrics(Path(".tmp/metrics.csv"))

    visualize_normal_map = False  # Set to True to visualize the normal maps
    if visualize_normal_map:
        import matplotlib.pyplot as plt

        # Visualize the normal maps
        depth = (
            torch.from_numpy(
                np.load("practicals/p3/demo/cloth3d/data/depth/03543_0.npy")
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        normal_map = PerceptualLoss._compute_normal_map(depth)

        # Move tensors to CPU and convert to numpy for visualization
        depth_np = depth[0, 0].detach().cpu().numpy()
        normal_map_np = normal_map[0].detach().cpu().numpy()

        # Transpose normal_map to (H, W, C) for visualization
        normal_map_img = np.transpose(normal_map_np, (1, 2, 0))

        # Normalize normal_map to [0, 1] for display
        normal_map_img = (normal_map_img - normal_map_img.min()) / (
            normal_map_img.max() - normal_map_img.min() + 1e-8
        )

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(depth_np, cmap="gray")
        axs[0].set_title("Depth Map")
        axs[0].axis("off")
        axs[1].imshow(normal_map_img)
        axs[1].set_title("Normal Map")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
