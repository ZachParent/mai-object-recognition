from dataset import get_dataloaders
from experiment_config import best_model_experiment, balancing_experiment
from metrics import get_metric_collection
from run_experiment import Trainer, MetricLogger
from config import DEVICE
import torch
from dataset import get_aux_dataloader
from visualize import get_best_and_worst_images, visualize_predictions

# Set spawn method at module level before any other multiprocessing operations
torch.multiprocessing.set_start_method("spawn", force=True)

item_names = [
    "headband, head covering, hair accessory",
    "jumpsuit",
    "belt",
    "sock",
    "glove",
    "tie",
    "cape",
    "scarf",
    "cardigan",
    "vest",
    "watch",
    "leg warmer",
]


def run_imbalance_finetuning():
    train_dataloader, val_dataloader = get_dataloaders(balancing_experiment, item_names)
    num_classes = len(item_names) + 1  # +1 for background class
    train_metrics_collection = get_metric_collection(num_classes)
    val_metrics_collection = get_metric_collection(num_classes)

    trainer = Trainer(
        balancing_experiment, train_metrics_collection, val_metrics_collection
    )
    model = trainer.load_previous_model(previous_experiment=best_model_experiment)
    model.classifier[-1] = torch.nn.Conv2d(256, num_classes, 1).to(DEVICE)
    model = trainer.load_previous_model(previous_experiment=balancing_experiment)


    print(f"Classifier head adjusted to {num_classes} classes")

    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(), lr=balancing_experiment.learning_rate
    )

    metrics_logger = MetricLogger(
        balancing_experiment.id,
        trainer.train_metrics_collection,
        trainer.val_metrics_collection,
    )

    for epoch in range(balancing_experiment.epochs):
        width = 90
        print("\n" + "=" * width)
        print(f"EPOCH {epoch+1} / {balancing_experiment.epochs}".center(width))
        print("-" * width)
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss = trainer.evaluate(val_dataloader)

        # Log metrics to TensorBoard and CSV (will also print epoch summary)
        metrics_logger.update_metrics(train_loss, val_loss)
        metrics_logger.log_metrics()

    metrics_logger.save_val_confusion_matrix()
    metrics_logger.close()

    if balancing_experiment.visualize:
        aux_dataloader = get_aux_dataloader(balancing_experiment, item_names)
        visualize_predictions(
            model=trainer.model,
            dataloader=aux_dataloader,
            num_classes=num_classes,
        )

    if balancing_experiment.save_weights:
        trainer.save_model()


if __name__ == "__main__":
    run_imbalance_finetuning()
