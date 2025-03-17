from config import RAW_DATA_DIR
from experiment_config import ExperimentConfig
from dataset import get_dataloaders
from pathlib import Path


def main():
    # Create data loaders
    experiment = ExperimentConfig(
        model_name="resnet18",
        learning_rate=0.001,
        epochs=4,
        batch_size=4,
    )
    train_loader, val_loader = get_dataloaders(experiment)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # # Visualize a sample
    # for images, masks in train_loader:
    #     # Display the first image and mask in batch
    #     visualize_segmentation(images[0], masks[0], category_mappings['id_to_name'])
    #     break

if __name__ == "__main__":
    main()
