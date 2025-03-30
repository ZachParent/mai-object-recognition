from dataset import get_dataloaders
from experiment_config import best_model_experiment
from metrics import get_metric_collection
from run_experiment import Trainer, MetricLogger
from config import DEVICE
import torch

# Set spawn method at module level before any other multiprocessing operations
torch.multiprocessing.set_start_method("spawn", force=True)

item_names = [
    "bag, wallet",
    "scarf",
    "umbrella",
]

def main():
    train_dataloader, val_dataloader = get_dataloaders(best_model_experiment, item_names)
    num_classes = len(item_names) + 1  # +1 for background class
    train_metrics_collection = get_metric_collection(num_classes)
    val_metrics_collection = get_metric_collection(num_classes)

    trainer = Trainer(best_model_experiment, train_metrics_collection, val_metrics_collection)
    model = trainer.load_previous_model()

    model.classifier[-1] = torch.nn.Conv2d(256, num_classes, 1).to(DEVICE)
    print(f"Classifier head adjusted to {num_classes} classes")

    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(), lr=best_model_experiment.learning_rate
    )

    metrics_logger = MetricLogger(
        best_model_experiment.id, trainer.train_metrics_collection, trainer.val_metrics_collection
    )

    for epoch in range(best_model_experiment.epochs):
        width = 90
        print("\n" + "=" * width)
        print(f"EPOCH {epoch+1} / {best_model_experiment.epochs}".center(width))
        print("-" * width)
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss = trainer.evaluate(val_dataloader)

        # Log metrics to TensorBoard and CSV (will also print epoch summary)
        metrics_logger.update_metrics(train_loss, val_loss)
        metrics_logger.log_metrics()

    metrics_logger.save_val_confusion_matrix()
    metrics_logger.close()

    if best_model_experiment.save_weights:
        trainer.save_model()

if __name__ == "__main__":
    main()