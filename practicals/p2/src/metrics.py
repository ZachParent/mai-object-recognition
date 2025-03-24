from typing import Callable, Literal, Optional, List, Dict
import torch
import pandas as pd
import torch.nn.functional as F
import torchmetrics
import torchmetrics.segmentation
from torch.utils.tensorboard import SummaryWriter
from config import RUNS_DIR, METRICS_DIR


# Define metrics to use
def get_metric_collection(num_classes: int) -> torchmetrics.MetricCollection:
    return torchmetrics.MetricCollection(
        [
            torchmetrics.segmentation.DiceScore(num_classes=num_classes),
        ]
    )


class MetricsLogger:
    def __init__(
        self,
        experiment_id: int,
        train_metrics: torchmetrics.MetricCollection,
        val_metrics: torchmetrics.MetricCollection,
    ) -> None:
        self._create_dirs()
        self.tb_writer = SummaryWriter(f"{RUNS_DIR}/experiment_{experiment_id:02d}")
        self.csv_path = f"{METRICS_DIR}/experiment_{experiment_id:02d}.csv"
        self.columns = ["epoch"]
        self.columns.extend([f"train_{name}" for name in train_metrics.keys()])
        self.columns.extend([f"val_{name}" for name in val_metrics.keys()])
        self.df = pd.DataFrame(columns=self.columns).astype({"epoch": int})
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.epoch = 0

    def _create_dirs(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def update_metrics(self) -> None:
        self.epoch += 1
        train_metric_values = self.train_metrics.compute()
        val_metric_values = self.val_metrics.compute()
        row = [self.epoch]
        for name, value in train_metric_values.items():
            row.append(value.item())
        for name, value in val_metric_values.items():
            row.append(value.item())
        self.df.loc[len(self.df)] = row

    def log_metrics(
        self,
    ) -> None:
        # Log to TensorBoard
        for col in self.columns[1:]:
            self.tb_writer.add_scalar(
                f"{col}".replace("train_", "train/").replace("val_", "val/"),
                self.df.iloc[-1][col],
                self.epoch,
            )

        # Log to CSV
        self.df["epoch"] = self.df["epoch"].astype(int)
        self.df.to_csv(self.csv_path, index=False)

        # Print summary after logging metrics
        self.print_epoch_summary()

    def print_epoch_summary(self) -> None:
        """
        Print a summary of the current epoch's metrics, with indicators showing
        improvement relative to the previous epoch.
        """
        width = 90
        print("\n" + "=" * width)
        print(f"EPOCH {self.epoch} SUMMARY".center(width))
        print("-" * width)

        # Headers
        print(f"{'Metric':<12} {'Train':<10} {'Val':<10} {'Train Δ':<15} {'Val Δ':<15}")
        print("-" * width)

        # Get previous epoch's metrics if available
        has_prev = len(self.df) > 1
        prev_epoch = None
        if has_prev:
            prev_epoch = self.df.iloc[-2]  # Previous epoch's data

        # Print each metric with trend indicators
        for name in self.train_metrics.keys():
            train_value = self.df.iloc[-1][f"train_{name}"]
            val_value = self.df.iloc[-1][f"val_{name}"]

            # Initialize trend indicators
            train_change = "="
            val_change = "="

            # Calculate changes from previous epoch
            train_diff = 0.0
            val_diff = 0.0

            if has_prev:
                prev_train = prev_epoch[f"train_{name}"]
                prev_val = prev_epoch[f"val_{name}"]

                train_diff = train_value - prev_train
                val_diff = val_value - prev_val

                # Determine if change is good or bad
                if name == "loss":  # For loss, lower is better
                    train_change = (
                        "↓ (good)"
                        if train_diff < 0
                        else "↑ (bad)" if train_diff > 0 else "="
                    )
                    val_change = (
                        "↓ (good)"
                        if val_diff < 0
                        else "↑ (bad)" if val_diff > 0 else "="
                    )
                else:  # For other metrics (like accuracy, IoU), higher is better
                    train_change = (
                        "↑ (good)"
                        if train_diff > 0
                        else "↓ (bad)" if train_diff < 0 else "="
                    )
                    val_change = (
                        "↑ (good)"
                        if val_diff > 0
                        else "↓ (bad)" if val_diff < 0 else "="
                    )

            # Format indicators
            train_indicator = f"{train_diff:+.6f} {train_change}"
            val_indicator = f"{val_diff:+.6f} {val_change}"

            print(
                f"{name:<12} {train_value:.6f}  {val_value:.6f}  {train_indicator:<15}  {val_indicator:<15}"
            )

        print("=" * width + "\n")

    def close(self) -> None:
        self.tb_writer.close()


if __name__ == "__main__":
    train_metrics = get_metric_collection(3)
    val_metrics = get_metric_collection(3)
    metrics_logger = MetricsLogger(69, train_metrics, val_metrics)

    train_metrics.update(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 10, 10))
    val_metrics.update(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 10, 10))

    metrics_logger.update_metrics()
    metrics_logger.log_metrics()
