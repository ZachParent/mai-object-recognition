from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from experiment_config import ExperimentConfig
from typing import Literal

def get_dataloader( train_val_test: Literal["train", "val", "test"], batch_size: int, shuffle: bool,):
    fashionpedia_dataset = load_dataset("detection-datasets/fashionpedia")
    return DataLoader(fashionpedia_dataset[train_val_test], batch_size=batch_size, shuffle=shuffle)

def augment_dataloader(dataloader: DataLoader):
    # Add random jitter to the images
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=10),
        T.ToTensor(),
    ])
    for batch in dataloader:
        batch["image"] = transform(batch["image"])
    return dataloader

def preprocess_dataloader(dataloader: DataLoader):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    for batch in dataloader:
        batch["image"] = transform(batch["image"])
    return dataloader

def get_dataloaders(experiment: ExperimentConfig):
    train_dataloader = preprocess_dataloader(get_dataloader(train_val_test="train", batch_size=experiment.batch_size, shuffle=True))
    val_dataloader = preprocess_dataloader(get_dataloader(train_val_test="val", batch_size=experiment.batch_size, shuffle=False))
    test_dataloader = preprocess_dataloader(get_dataloader(train_val_test="test", batch_size=experiment.batch_size, shuffle=False))
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    dataloaders = get_dataloaders(ExperimentConfig(batch_size=32, model_name="resnet18", learning_rate=0.001, epochs=4))
    print(dataloaders)