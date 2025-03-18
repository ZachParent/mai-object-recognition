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


class MetricsLogger:
    def __init__(self, experiment_id):
        self._create_dirs()
        self.tb_writer = SummaryWriter(f"{RUNS_DIR}/{experiment_id:02d}")
        self.csv_path = f"{METRICS_DIR}/{experiment_id:02d}.csv"
        self.experiment_id = experiment_id
        self._init_csv()

    def _create_dirs(self):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def _init_csv(self):
        with open(self.csv_path, "w") as f:
            headers = ["step", "timestamp"] + self.get_metric_names()
            csv.writer(f).writerow(headers)

    def get_metric_names(self):
        return ["loss"]

    def log_metrics(self, train_metrics: dict, val_metrics: dict, step: int):
        # Log to TensorBoard
        for name, value in train_metrics.items():
            self.tb_writer.add_scalar(f"train/{name}", value, step)
        for name, value in val_metrics.items():
            self.tb_writer.add_scalar(f"val/{name}", value, step)

        # Log to CSV
        self._log_to_csv(train_metrics, val_metrics, step)

    def _log_to_csv(self, train_metrics: dict, val_metrics: dict, step: int):
        with open(self.csv_path, "a") as f:
            row = [step]
            row += [
                f"train_{name}: {train_metrics[name]}"
                for name in self.get_metric_names()
            ]
            row += [
                f"val_{name}: {val_metrics[name]}" for name in self.get_metric_names()
            ]
            csv.writer(f).writerow(row)

    def close(self):
        self.tb_writer.close()


class TrainingProgress:
    def __init__(self, dataloader: DataLoader, desc="", metrics=None):
        self.pbar = tqdm(total=len(dataloader), desc=desc)
        self.metrics = metrics or {}  # Dict to track moving averages

        # Format description
        metrics_str = " | ".join(
            f"{name}: {value:.4f}" for name, value in self.metrics.items()
        )
        self.pbar.set_description(f"{self.pbar.desc} | {metrics_str}")
        self.pbar.update(1)

    def update(self, metrics):
        # Update metrics
        self.metrics.update(metrics)

        # Update progress bar description
        metrics_str = " | ".join(
            f"{name}: {value:.4f}" for name, value in self.metrics.items()
        )
        self.pbar.set_description(f"{self.pbar.desc} | {metrics_str}")
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
            outputs = self.model(image)
            loss = self.criterion(outputs, target['masks'].squeeze(1))
            metrics = {
                "loss": loss.item(),
            }
            progress.update(metrics)

            self.model.backward(loss)
            self.optimizer.step()

        progress.close()
        return metrics

    def evaluate(self, dataloader):
        self.model.eval()
        progress = TrainingProgress(dataloader, desc="Evaluating")

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(batch["pixel_values"])
                loss = self.criterion(outputs, batch["label"])
                # Your evaluation loop here
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
