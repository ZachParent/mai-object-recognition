from experiment_config import ExperimentConfig
from models import get_model
from dataset import get_dataloaders
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.tensorboard import SummaryWriter
import csv
from config import RUNS_DIR, METRICS_DIR
import numpy as np
import pandas as pd
from metrics import ALL_METRICS, MetricsLogger


class TrainingProgress:
    def __init__(
        self,
        dataloader: DataLoader,
        desc: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        self.pbar = tqdm(total=len(dataloader), desc=desc)
        self.metrics = metrics or {}  # Dict to track moving averages
        self.desc = desc.ljust(10)

        # Format description
        if self.metrics:
            self._set_description(self.metrics)
        self.pbar.update(0)  # Don't increment on init

    def _set_description(self, metrics: Dict[str, float]) -> None:
        metrics_str = " | ".join(
            f"{name}: {value:.4f}" for name, value in metrics.items()
        )
        self.pbar.set_description(f"{self.desc} | {metrics_str}")

    def update(self, metrics: Dict[str, float]) -> None:
        # Replace metrics instead of updating
        self.metrics = metrics

        # Update progress bar description
        self._set_description(self.metrics)
        self.pbar.update(1)

    def close(self) -> None:
        self.pbar.close()


class Trainer:
    def __init__(self, experiment: ExperimentConfig) -> None:
        self.experiment = experiment
        self.model = get_model(experiment.model_name)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=experiment.learning_rate
        )
        self.criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        progress = TrainingProgress(dataloader, desc="Training")
        metrics: Dict[str, float] = {}

        for image, target in dataloader:
            # Your training loop here
            # TODO: fix this
            loss = torch.tensor(np.random.uniform(0, 1))
            # outputs = self.model(image)
            # loss = self.criterion(outputs, target['masks'])

            metrics = {
                "loss": loss.item(),
            }
            # TODO: fix this
            for metric in ALL_METRICS:
                metrics[metric.name] = torch.tensor(np.random.uniform(0, 1)).item()
            progress.update(metrics)

            # self.model.backward(loss)
            # self.optimizer.step()

        progress.close()
        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        progress = TrainingProgress(dataloader, desc="Evaluating")
        metrics: Dict[str, float] = {}

        with torch.no_grad():
            for image, target in dataloader:
                # TODO: fix this
                loss = torch.tensor(np.random.uniform(0, 1))
                # outputs = self.model(image)
                # loss = self.criterion(outputs, target['masks'])

                metrics = {
                    "loss": loss.item(),
                }
                # TODO: fix this
                for metric in ALL_METRICS:
                    metrics[metric.name] = torch.tensor(np.random.uniform(0, 1)).item()
                progress.update(metrics)

        progress.close()
        return metrics


def run_experiment(experiment: ExperimentConfig) -> None:
    train_dataloader, val_dataloader = get_dataloaders(experiment)

    trainer = Trainer(experiment)
    metrics_logger = MetricsLogger(experiment.id)
    for epoch in range(experiment.epochs):
        print(f"Epoch {epoch}/{experiment.epochs}")
        train_metrics = trainer.train_epoch(train_dataloader)
        val_metrics = trainer.evaluate(val_dataloader)
        metrics_logger.log_metrics(train_metrics, val_metrics, epoch)
    metrics_logger.close()


if __name__ == "__main__":
    experiment = ExperimentConfig(
        id=0,
        model_name="resnet18",
        learning_rate=0.001,
        batch_size=16,
        epochs=1,
    )
    run_experiment(experiment)
