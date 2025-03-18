from typing import Callable, Literal, Optional, List, Dict
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from config import RUNS_DIR, METRICS_DIR


class Metric(Callable):
    def __init__(self):
        self.name: str | None = None
        self.display_name: str | None = None

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        pass


# TODO: implement MDice (or use existing library)
class MDice(Metric):
    def __init__(self):
        super().__init__()
        self.name: Literal["m_dice"] = "m_dice"
        self.display_name: Literal["mDice"] = "mDice"

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return (outputs.argmax(dim=1) == targets).float().mean().item()


# TODO: add other metrics
class F1Score(Metric):
    def __init__(self):
        super().__init__()
        self.name = "f1_score"
        self.display_name = "F1 Score"

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return (outputs.argmax(dim=1) == targets).float().mean().item()


ALL_METRICS = [MDice(), F1Score()]


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
            + [train_metrics[name] for name in self.metrics]
            + [val_metrics[name] for name in self.metrics]
        )
        self.df["epoch"] = self.df["epoch"].astype(int)
        self.df.to_csv(self.csv_path, index=False)

    def close(self) -> None:
        self.tb_writer.close()
