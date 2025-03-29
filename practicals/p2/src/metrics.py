from typing import Callable, Literal, Optional, List, Dict
import os
import torch
import pandas as pd
import torch.nn.functional as F
import torchmetrics
import torchmetrics.segmentation
import torchmetrics.classification
from torch.utils.tensorboard import SummaryWriter
from config import RUNS_DIR, METRICS_DIR, CONFUSION_MATRICES_DIR

# Metric name constants
METRIC_LOSS = "loss"
METRIC_ACCURACY = "accuracy"
METRIC_ACCURACY_W_BG = "accuracy_w_bg"
METRIC_DICE = "dice"
METRIC_DICE_W_BG = "dice_w_bg"
METRIC_F1 = "f1"
METRIC_F1_W_BG = "f1_w_bg"
METRIC_PRECISION = "precision"
METRIC_PRECISION_W_BG = "precision_w_bg"
METRIC_RECALL = "recall"
METRIC_RECALL_W_BG = "recall_w_bg"
METRIC_CONFUSION_MATRIX = "confusion_matrix"

# Prefixes
PREFIX_TRAIN = "train_"
PREFIX_VAL = "val_"


def compile_best_runs_csv(experiment_set, metric=METRIC_DICE):
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

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(best_runs_path), exist_ok=True)

    # Load existing CSV if it exists, otherwise create a new DataFrame
    if os.path.exists(best_runs_path):
        best_runs_df = pd.read_csv(best_runs_path)
        # Ensure all required columns exist
        for col in column_names:
            if col not in best_runs_df.columns:
                best_runs_df[col] = None
    else:
        best_runs_df = pd.DataFrame(columns=column_names)
        # Create the file immediately
        best_runs_df.to_csv(best_runs_path, index=False)

    for experiment in experiment_set.configs:
        csv_path = f"{METRICS_DIR}/experiment_{experiment.id:02d}.csv"
        try:
            df = pd.read_csv(csv_path)
            # Find row with best metric value
            val_metric_col = f"{PREFIX_VAL}{metric}"
            best_row = df[df[val_metric_col] == df[val_metric_col].max()].iloc[0]

            # Prepare new row data
            new_row = {
                "experiment_set": experiment_set.title,
                "experiment_id": experiment.id,
                "model_name": experiment.model_name,
                "learning_rate": experiment.learning_rate,
                "batch_size": experiment.batch_size,
                "img_size": experiment.img_size,
                metric: best_row[f"val_{metric}"],
            }

            # Check if this experiment_set and model_name combination already exists
            mask = (best_runs_df["experiment_set"] == experiment_set.title) & (
                best_runs_df["model_name"] == experiment.model_name
            )

            if mask.any():
                # If exists, check if new value is better
                if new_row[metric] > best_runs_df.loc[mask, metric].values[0]:
                    # Update each column individually to avoid dimension mismatch
                    for col, val in new_row.items():
                        best_runs_df.loc[mask, col] = val
            else:
                # If doesn't exist, append new row using pandas loc with length as index
                best_runs_df.loc[len(best_runs_df)] = new_row
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue

    # Save updated DataFrame
    best_runs_df.to_csv(best_runs_path, index=False)
    return best_runs_df


# Define metrics to use
def get_metric_collection(num_classes: int) -> torchmetrics.MetricCollection:
    return torchmetrics.MetricCollection(
        {
            METRIC_ACCURACY_W_BG: torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes
            ),
            METRIC_ACCURACY: torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, ignore_index=0
            ),
            METRIC_DICE_W_BG: torchmetrics.segmentation.DiceScore(
                input_format="index", num_classes=num_classes, average="macro"
            ),
            METRIC_DICE: torchmetrics.segmentation.DiceScore(
                input_format="index",
                num_classes=num_classes,
                include_background=False,
                average="macro",
            ),
            METRIC_F1_W_BG: torchmetrics.classification.MulticlassF1Score(
                num_classes=num_classes
            ),
            METRIC_F1: torchmetrics.classification.MulticlassF1Score(
                num_classes=num_classes, ignore_index=0
            ),
            METRIC_PRECISION_W_BG: torchmetrics.classification.MulticlassPrecision(
                num_classes=num_classes
            ),
            METRIC_PRECISION: torchmetrics.classification.MulticlassPrecision(
                num_classes=num_classes, ignore_index=0
            ),
            METRIC_RECALL_W_BG: torchmetrics.classification.MulticlassRecall(
                num_classes=num_classes
            ),
            METRIC_RECALL: torchmetrics.classification.MulticlassRecall(
                num_classes=num_classes, ignore_index=0
            ),
            METRIC_CONFUSION_MATRIX: torchmetrics.classification.MulticlassConfusionMatrix(
                num_classes=num_classes
            ),
        }
    )


metrics_order = [
    # Primary metrics
    METRIC_LOSS,
    METRIC_DICE,
    METRIC_F1,
    METRIC_ACCURACY,
    # Additional metrics
    METRIC_PRECISION,
    METRIC_RECALL,
    # With background metrics
    METRIC_DICE_W_BG,
    METRIC_F1_W_BG,
    METRIC_ACCURACY_W_BG,
    METRIC_PRECISION_W_BG,
    METRIC_RECALL_W_BG,
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
        self.metrics_path = f"{METRICS_DIR}/experiment_{experiment_id:02d}.csv"
        self.confusion_matrix_path = (
            f"{CONFUSION_MATRICES_DIR}/experiment_{experiment_id:02d}.csv"
        )
        self.columns = ["epoch"]
        self.columns.extend([f"{PREFIX_TRAIN}{name}" for name in metrics_order])
        self.columns.extend([f"{PREFIX_VAL}{name}" for name in metrics_order])
        self.df = pd.DataFrame(columns=self.columns).astype({"epoch": int})
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.val_confusion_matrix: pd.DataFrame
        self.epoch = 0

    def _create_dirs(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)

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
        self.val_confusion_matrix = pd.DataFrame(
            val_metric_values[METRIC_CONFUSION_MATRIX].cpu()
        )

    def log_metrics(
        self,
    ) -> None:
        # Log to TensorBoard
        for col in self.columns[1:]:
            self.tb_writer.add_scalar(
                f"{col}".replace(PREFIX_TRAIN, "train/").replace(PREFIX_VAL, "val/"),
                self.df.iloc[-1][col],
                self.epoch,
            )

        # Log to CSV
        self.df["epoch"] = self.df["epoch"].astype(int)
        self.df.to_csv(self.metrics_path, index=False)

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
            train_col = f"{PREFIX_TRAIN}{name}"
            val_col = f"{PREFIX_VAL}{name}"
            
            if train_col not in self.df.columns:
                continue

            train_value = self.df.iloc[-1][train_col]
            val_value = self.df.iloc[-1][val_col]

            # Initialize trend indicators
            train_change = "="
            val_change = "="

            # Calculate changes from previous epoch
            train_diff = 0.0
            val_diff = 0.0

            if has_prev:
                prev_train = prev_epoch[train_col]
                prev_val = prev_epoch[val_col]

                train_diff = train_value - prev_train
                val_diff = val_value - prev_val

                # Determine if change is good or bad
                if name == METRIC_LOSS:  # For loss, lower is better
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

    def save_val_confusion_matrix(self) -> None:
        self.val_confusion_matrix.to_csv(self.confusion_matrix_path)

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
    metrics_logger.save_val_confusion_matrix()
