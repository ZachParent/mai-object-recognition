# %%
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from experiment_config import ExperimentConfig
from config import DEVICE, USING_CUDA
from torchvision.transforms import Compose, ColorJitter, ToTensor, ToPILImage, Resize
import torch
from PIL import Image
import numpy as np


def transforms(examples):
    """Transform images to tensors and resize them."""
    transformed = []
    for image in examples["image"]:
        # Convert to PIL Image first (handles different input formats)
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(image, np.ndarray):
            # Ensure image is in uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Apply transforms
        transformed_image = jitter(image)
        transformed.append(transformed_image)

    # Stack the tensors to ensure consistent shape
    examples["pixel_values"] = torch.stack(transformed)
    return examples


# Define transforms
jitter = Compose(
    [
        Resize((224, 224)),  # Specify both dimensions
        ToTensor(),
    ]
)


def custom_collate(batch):
    """Custom collate function to handle the batch creation."""
    pixel_values = torch.stack(
        [torch.tensor(item["pixel_values"]) for item in batch]
    )

    # Collect all other keys that aren't pixel_values
    other_keys = {key: [] for key in batch[0].keys() if key != "pixel_values"}
    for item in batch:
        for key in other_keys:
            other_keys[key].append(item[key])

    # Create the final batch
    final_batch = {"pixel_values": pixel_values}
    final_batch.update(other_keys)

    return final_batch


def get_dataloaders(experiment: ExperimentConfig):
    # Load dataset without torch format initially
    dataset_dict = load_dataset("detection-datasets/fashionpedia")

    # Limit the dataset size to 100 for testing
    # TODO: Remove this
    for split in dataset_dict.keys():
        dataset_dict[split] = dataset_dict[split].select(range(100))
        print(f"{split} size: {len(dataset_dict[split])}")

    # Apply transforms
    dataset_dict = dataset_dict.map(
        transforms,
        remove_columns=["image"],
        batched=True,
        batch_size=32,
        desc="Processing images",
    )

    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["val"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=experiment.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    experiment = ExperimentConfig(
        batch_size=32, model_name="resnet18", learning_rate=0.001, epochs=4
    )
    train_dataloader, val_dataloader = get_dataloaders(experiment)

    # Test the first batch
    batch = next(iter(train_dataloader))
    print("Image shape:", batch["pixel_values"].shape)
    print("Available keys:", batch.keys())

# %%
