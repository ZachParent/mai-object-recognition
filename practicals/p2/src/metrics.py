from typing import Callable, Literal, Optional, List, Dict
import torch
import pandas as pd
import torch.nn.functional as F
import torchmetrics
import torchmetrics.segmentation
import torchmetrics.classification
from torch.utils.tensorboard import SummaryWriter
from config import RUNS_DIR, METRICS_DIR


# Define metrics to use
def get_metric_collection(num_classes: int) -> torchmetrics.MetricCollection:
    return torchmetrics.MetricCollection(
        {
            "accuracy_w_bg": torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes
            ),
            "accuracy": torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, ignore_index=0
            ),
            "dice_w_bg": torchmetrics.segmentation.DiceScore(
                input_format="index", num_classes=num_classes, average="macro"
            ),
            "dice": torchmetrics.segmentation.DiceScore(
                input_format="index",
                num_classes=num_classes,
                include_background=False,
                average="macro",
            ),
            "f1_w_bg": torchmetrics.classification.MulticlassF1Score(
                num_classes=num_classes
            ),
            "f1": torchmetrics.classification.MulticlassF1Score(
                num_classes=num_classes, ignore_index=0
            ),
            "precision_w_bg": torchmetrics.classification.MulticlassPrecision(
                num_classes=num_classes
            ),
            "precision": torchmetrics.classification.MulticlassPrecision(
                num_classes=num_classes, ignore_index=0
            ),
            "recall_w_bg": torchmetrics.classification.MulticlassRecall(
                num_classes=num_classes
            ),
            "recall": torchmetrics.classification.MulticlassRecall(
                num_classes=num_classes, ignore_index=0
            ),
        }
    )


metrics_order = [
    # Primary metrics
    "loss",
    "dice",
    "f1",
    "accuracy",
    # Additional metrics
    "precision",
    "recall",
    # With background metrics
    "dice_w_bg",
    "f1_w_bg",
    "accuracy_w_bg",
    "precision_w_bg",
    "recall_w_bg",
]


class MetricLogger:
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
        self.columns.extend([f"train_{name}" for name in metrics_order])
        self.columns.extend([f"val_{name}" for name in metrics_order])
        self.df = pd.DataFrame(columns=self.columns).astype({"epoch": int})
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.epoch = 0

    def _create_dirs(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def update_metrics(self, train_loss: float, val_loss: float) -> None:
        self.epoch += 1
        train_metric_values = self.train_metrics.compute()
        val_metric_values = self.val_metrics.compute()
        row = [self.epoch]
        row.append(train_loss)
        for name in metrics_order[1:]:
            row.append(train_metric_values[name].item())
        row.append(val_loss)
        for name in metrics_order[1:]:
            row.append(val_metric_values[name].item())
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
        print(f"{'Metric':<15} {'Train':<10} {'Val':<10} {'Train Δ':<15} {'Val Δ':<15}")
        print("-" * width)

        # Get previous epoch's metrics if available
        has_prev = len(self.df) > 1
        prev_epoch = None
        if has_prev:
            prev_epoch = self.df.iloc[-2]  # Previous epoch's data

        # Print each metric in the defined order
        for name in metrics_order:
            if f"train_{name}" not in self.df.columns:
                continue

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
                f"{name:<15} {train_value:.6f}  {val_value:.6f}  {train_indicator:<15}  {val_indicator:<15}"
            )

        print("=" * width + "\n")

    def close(self) -> None:
        self.tb_writer.close()


if __name__ == "__main__":
    num_classes = 20
    batch_size = 16
    train_metrics = get_metric_collection(num_classes)
    val_metrics = get_metric_collection(num_classes)
    metrics_logger = MetricLogger(69, train_metrics, val_metrics)

    example_output = torch.randn(batch_size, num_classes, 10, 10)
    example_target = torch.randint(0, num_classes, (batch_size, 10, 10))
    argmax_output = example_output.argmax(dim=1)
    train_loss = torch.randn(1).item()
    val_loss = torch.randn(1).item()

    train_metrics.update(argmax_output, example_target)
    val_metrics.update(argmax_output, example_target)

    metrics_logger.update_metrics(train_loss, val_loss)
    metrics_logger.log_metrics()
