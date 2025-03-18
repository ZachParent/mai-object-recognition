from experiment_config import ExperimentConfig
from models import get_model
from dataset import get_dataloaders
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import csv
from config import RUNS_DIR, METRICS_DIR
import numpy as np
import pandas as pd


class MetricsLogger:
    def __init__(self, experiment_id):
        self._create_dirs()
        self.tb_writer = SummaryWriter(f"{RUNS_DIR}/experiment_{experiment_id:02d}")
        self.csv_path = f"{METRICS_DIR}/experiment_{experiment_id:02d}.csv"
        self.df = pd.DataFrame(
            columns=["epoch"]
            + [f"train_{name}" for name in self.get_metric_names()]
            + [f"val_{name}" for name in self.get_metric_names()]
        )
        self.experiment_id = experiment_id

    def _create_dirs(self):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def get_metric_names(self):
        return ["loss"]

    def log_metrics(self, train_metrics: dict, val_metrics: dict, epoch: int):
        # Log to TensorBoard
        for name, value in train_metrics.items():
            self.tb_writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in val_metrics.items():
            self.tb_writer.add_scalar(f"val/{name}", value, epoch)

        # Log to CSV
        self.df.loc[len(self.df)] = (
            [epoch]
            + [train_metrics[name] for name in self.get_metric_names()]
            + [val_metrics[name] for name in self.get_metric_names()]
        )
        self.df.to_csv(self.csv_path, index=False)

    def close(self):
        self.tb_writer.close()


class TrainingProgress:
    def __init__(self, dataloader: DataLoader, desc="", metrics=None):
        self.pbar = tqdm(total=len(dataloader), desc=desc)
        self.metrics = metrics or {}  # Dict to track moving averages
        self.desc = desc

        # Format description
        if self.metrics:
            self._set_description(self.metrics)
        self.pbar.update(0)  # Don't increment on init

    def _set_description(self, metrics):
        metrics_str = " | ".join(
            f"{name}: {value:.4f}" for name, value in metrics.items()
        )
        self.pbar.set_description(f"{self.desc} | {metrics_str}")

    def update(self, metrics):
        # Replace metrics instead of updating
        self.metrics = metrics

        # Update progress bar description
        self._set_description(self.metrics)
        self.pbar.update(1)

    def close(self):
        self.pbar.close()


class Trainer:
    def __init__(self, experiment: ExperimentConfig):
        self.experiment = experiment
        self.model = get_model(experiment.model_name)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=experiment.learning_rate
        )
        self.criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        progress = TrainingProgress(dataloader, desc="Training")
        metrics: Dict[str, float]

        for image, target in dataloader:
            # Your training loop here
            # TODO: fix this
            loss = torch.tensor(np.random.uniform(0, 1))
            # outputs = self.model(image)
            # loss = self.criterion(outputs, target['masks'])

            metrics = {
                "loss": loss.item(),
            }
            progress.update(metrics)

            # self.model.backward(loss)
            # self.optimizer.step()

        progress.close()
        return metrics

    def evaluate(self, dataloader):
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
                progress.update(metrics)

        progress.close()
        return metrics


def run_experiment(experiment: ExperimentConfig):
    train_dataloader, val_dataloader = get_dataloaders(experiment)

    trainer = Trainer(experiment)
    metrics_logger = MetricsLogger(experiment.id)
    for epoch in range(experiment.epochs):
        print(f"\tEpoch {epoch}/{experiment.epochs}")
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
