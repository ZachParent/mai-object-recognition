from typing import Callable, Literal, Optional, List, Dict
import os
import torch
import pandas as pd
import torch.nn.functional as F
import torchmetrics
import torchmetrics.segmentation
from torch.utils.tensorboard import SummaryWriter
from config import RUNS_DIR, METRICS_DIR


def compile_best_runs_csv(experiment_set, metric="dice"):
    best_runs_path = f"{METRICS_DIR}/best_runs.csv"
    column_names = [
        "experiment_set",
        "experiment_id",
        "model_name",
        "learning_rate",
        "batch_size",
        "img_size",
        metric,
    ]

    # Load existing CSV if it exists, otherwise create a new DataFrame
    if os.path.exists(best_runs_path):
        best_runs_df = pd.read_csv(best_runs_path)
    else:
        best_runs_df = pd.DataFrame(columns=column_names)

    for experiment in experiment_set.configs:
        csv_path = f"{METRICS_DIR}/experiment_{experiment.id:02d}.csv"
        try:
            df = pd.read_csv(csv_path)
            # Find row with best metric value
            best_row = df[df[f"val_{metric}"] == df[f"val_{metric}"].max()].iloc[0]

            # Prepare new row data
            new_row = {
                "experiment_set": experiment_set.name,
                "experiment_id": experiment.id,
                "model_name": experiment.model_name,
                "learning_rate": experiment.learning_rate,
                "batch_size": experiment.batch_size,
                "img_size": experiment.img_size,
                metric: best_row[f"val_{metric}"],
            }

            # Check if this experiment_set and model_name combination already exists
            mask = (best_runs_df["experiment_set"] == experiment_set.name) & (
                best_runs_df["model_name"] == experiment.model_name
            )

            if mask.any():
                # If exists, check if new value is better
                if new_row[metric] > best_runs_df.loc[mask, metric].values[0]:
                    best_runs_df.loc[mask] = new_row
            else:
                # If doesn't exist, append new row
                best_runs_df = pd.concat(
                    [best_runs_df, pd.DataFrame([new_row])], ignore_index=True
                )
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue

    # Save updated DataFrame
    best_runs_df.to_csv(best_runs_path, index=False)
    return best_runs_df


# Define metrics to use
def get_metric_collection(num_classes: int) -> torchmetrics.MetricCollection:
    return torchmetrics.MetricCollection(
        {
            "dice": torchmetrics.segmentation.DiceScore(
                input_format="index", num_classes=num_classes, include_background=False
            ),
            "accuracy": torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes
            ),
            "precision": torchmetrics.classification.MulticlassPrecision(
                num_classes=num_classes
            ),
            "recall": torchmetrics.classification.MulticlassRecall(
                num_classes=num_classes
            ),
            "f1": torchmetrics.classification.MulticlassF1Score(
                num_classes=num_classes
            ),
        }
    )


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
    metrics_logger = MetricLogger(69, train_metrics, val_metrics)

    example_output = torch.randn(3, 3, 10, 10)
    example_target = torch.randint(0, 3, (3, 10, 10))
    argmax_output = example_output.argmax(dim=1)

    train_metrics.update(argmax_output, example_target)
    val_metrics.update(argmax_output, example_target)

    metrics_logger.update_metrics()
    metrics_logger.log_metrics()
