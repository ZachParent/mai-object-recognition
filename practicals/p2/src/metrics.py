from typing import Callable, Literal, Optional, List, Dict
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from config import RUNS_DIR, METRICS_DIR


class Metric(Callable):
    def __init__(self):
        self.name: str | None = None
        self.display_name: str | None = None

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        pass


class MDice(Metric):
    def __init__(self):
        super().__init__()
        self.name = "m_dice"
        self.display_name = "mDice"
        self.smooth = 1e-6

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        preds = outputs.argmax(dim=1)
        num_classes = outputs.size(1)

        # One-hot encode predictions and targets
        preds_one_hot = F.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Calculate dice scores for each class (skip background)
        dice_scores = []
        for c in range(1, num_classes):
            pred_c = preds_one_hot[:, c].contiguous().view(-1)
            target_c = targets_one_hot[:, c].contiguous().view(-1)

            intersection = (pred_c * target_c).sum()
            dice = (2.0 * intersection + self.smooth) / (
                pred_c.sum() + target_c.sum() + self.smooth
            )
            dice_scores.append(dice.item())

        # Average dice score across classes
        return sum(dice_scores) / max(len(dice_scores), 1)


class IoU(Metric):
    def __init__(self):
        super().__init__()
        self.name = "iou"
        self.display_name = "IoU"
        self.smooth = 1e-6

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        preds = outputs.argmax(dim=1)
        num_classes = outputs.size(1)

        # One-hot encode predictions and targets
        preds_one_hot = F.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Calculate IoU for each class (skip background)
        iou_scores = []
        for c in range(1, num_classes):
            pred_c = preds_one_hot[:, c].contiguous().view(-1)
            target_c = targets_one_hot[:, c].contiguous().view(-1)

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_scores.append(iou.item())

        # Average IoU across classes
        return sum(iou_scores) / max(len(iou_scores), 1)


# Define metrics to use
ALL_METRICS = [MDice(), IoU()]


class MetricsLogger:
    def __init__(self, experiment_id: int, metrics: Optional[List[str]] = None) -> None:
        self._create_dirs()
        self.tb_writer = SummaryWriter(f"{RUNS_DIR}/experiment_{experiment_id:02d}")
        self.csv_path = f"{METRICS_DIR}/experiment_{experiment_id:02d}.csv"
        self.metrics = metrics or ["loss"] + [metric.name for metric in ALL_METRICS]
        self.df = pd.DataFrame(
            columns=["epoch"]
            + [f"train_{name}" for name in self.metrics]
            + [f"val_{name}" for name in self.metrics]
        ).astype({"epoch": int})

    def _create_dirs(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def log_metrics(
        self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int
    ) -> None:
        # Log to TensorBoard
        for name, value in train_metrics.items():
            self.tb_writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in val_metrics.items():
            self.tb_writer.add_scalar(f"val/{name}", value, epoch)

        # Log to CSV
        self.df.loc[len(self.df)] = (
            [epoch]
            + [train_metrics.get(name, 0.0) for name in self.metrics]
            + [val_metrics.get(name, 0.0) for name in self.metrics]
        )
        self.df["epoch"] = self.df["epoch"].astype(int)
        self.df.to_csv(self.csv_path, index=False)

        # Print summary after logging metrics
        self.print_epoch_summary(train_metrics, val_metrics, epoch)

    def print_epoch_summary(
        self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int
    ) -> None:
        """
        Print a summary of the current epoch's metrics, with indicators showing
        improvement relative to the previous epoch.
        """
        width = 90
        print("\n" + "=" * width)
        print(f"EPOCH {epoch+1} SUMMARY".center(width))
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
        for name in self.metrics:
            train_value = train_metrics.get(name, 0.0)
            val_value = val_metrics.get(name, 0.0)

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
