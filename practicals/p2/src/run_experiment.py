from experiment_config import ExperimentConfig
from models import get_model
from dataset import get_dataloaders
from metrics import calculate_accuracy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

class TrainingProgress:
    def __init__(self, dataloader: DataLoader, desc="", metrics=Dict[str, float]):
        self.pbar = tqdm(total=len(dataloader), desc=desc)
        self.metrics = metrics or {}  # Dict to track moving averages
        
        # Format description
        metrics_str = " | ".join(
            f"{name}: {value:.4f}" 
            for name, value in self.metrics.items()
        )
        self.pbar.set_description(f"{self.pbar.desc} | {metrics_str}")
        self.pbar.update(1)
    
    def close(self):
        self.pbar.close()

class Trainer:
    def __init__(self, experiment: ExperimentConfig):
        self.experiment = experiment
        self.model = get_model(experiment.model_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        progress = TrainingProgress(dataloader, desc="Training")
        
        for batch in dataloader:
            # Your training loop here
            outputs = self.model(batch["pixel_values"])
            loss = self.criterion(outputs, batch["label"])
            for metric in self.experiment.metrics:
            
            # Update progress
            progress.update({
                "loss": loss.item(),
                "acc": acc
            })
        
        progress.close()
        return progress.metrics

    def evaluate(model, dataloader):
        model.eval()
        progress = TrainingProgress(dataloader, desc="Evaluating")
        
        with torch.no_grad():
            for batch in dataloader:
                # Your evaluation loop here
                progress.update({
                    "loss": loss.item(),
                    "acc": acc
                })
        
        progress.close()
        return progress.metrics

def run_experiment(experiment: ExperimentConfig):
    model = get_model(experiment.model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.learning_rate)

    train_dataloader, val_dataloader= get_dataloaders(experiment)

    trainer = Trainer(experiment)
    for epoch in range(experiment.epochs):
        train_metrics = trainer.train_epoch(train_dataloader)
        val_metrics = trainer.evaluate(model, val_dataloader)
