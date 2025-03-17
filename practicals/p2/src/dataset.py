# %%
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import tv_tensors
from experiment_config import ExperimentConfig
from config import DEVICE, USING_CUDA, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_ANNOTATIONS_JSON, VAL_ANNOTATIONS_JSON
from torchvision.transforms import Compose, ColorJitter, ToTensor, ToPILImage, Resize
import torch
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import json


class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # Determine the width and height of the image
        width, height = image.size

        # Determine the scaling factor
        x_scale = self.size / width
        y_scale = self.size / height

        # Resize the image
        image = image.resize((self.size, self.size), Image.BILINEAR)
        
        # Scale the segmentation values in the target
        if 'segmentations' in target:
            target['segmentations'] = [
                [(x * x_scale, y * y_scale) for x, y in zip(seg[::2], seg[1::2])]
                for seg in target['segmentations']
            ]
        if 'bbox' in target:
            target['bbox'] = [
                [x * x_scale]
                for x in target['bbox']
            ]
        return image, target

class FashionpediaDataset(Dataset):
    def __init__(self, img_dir, instances_file, img_size, transform=None, max_samples=None):
        with open(instances_file, "r") as f:
            self.instances_file_data = json.load(f)
        self.annotations = pd.DataFrame(self.instances_file_data["annotations"]).set_index("image_id")
        self.images = pd.DataFrame(self.instances_file_data["images"]).set_index("id")
        self.img_dir = img_dir
        self.img_size = img_size
        self.resize_transform = ResizeTransform(img_size)
        self.transform = transform
        self.max_samples = max_samples
        if max_samples:
            self.annotations = self.annotations.sample(n=max_samples, random_state=42)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]

        img_id = self.annotations.index[idx]
        img_path = Path(self.img_dir) / self.images.loc[img_id]["file_name"]
        image = Image.open(img_path).convert('RGB')
        
        target = annotation
        
        image, target = self.resize_transform(image, target)

        if self.transform:
            image = self.transform(image)
        
        # convert each part of the target to a tensor
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                data=target["bbox"],
                format=tv_tensors.BoundingBoxFormat.XYWH,
                canvas_size=(self.img_size, self.img_size),
                dtype=torch.float32,
                device=torch.device(DEVICE),
            ),
            "area": torch.tensor(target["area"]),
            "labels": torch.tensor(target["category_id"]),
            "masks": [
                tv_tensors.Mask(
                    data=np.array(seg).reshape(-1, 2),
                    dtype=torch.uint16,
                    device=torch.device(DEVICE),
                )
                for seg in target["segmentation"]
            ],
        }
            
        return image, target

def get_dataloaders(experiment: ExperimentConfig):
    # Load dataset without torch format initially
    train_dataset = FashionpediaDataset(
        img_dir=Path(TRAIN_IMAGES_DIR),
        instances_file=Path(TRAIN_ANNOTATIONS_JSON),
        img_size=224,
        transform=T.ToTensor(),
        max_samples=100,
        # target_transform=T.ToTensor(),
    )
    
    # val_dataset = FashionpediaDataset(
    #     img_dir=Path(VAL_IMAGES_DIR),
    #     instances_file=Path(VAL_ANNOTATIONS_JSON),
    #     transform=T.ToTensor(),
    #     target_transform=T.ToTensor(),
    # )

    # Limit the dataset size to 100 for testing
    # TODO: Remove this
    # train_dataset = train_dataset[:100]
    # val_dataset = val_dataset[:100]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        # collate_fn=custom_collate,
    )
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=experiment.batch_size,
    #     shuffle=False,
    #     # collate_fn=custom_collate,
    # )
    return train_dataloader


if __name__ == "__main__":
    experiment = ExperimentConfig(
        batch_size=32, model_name="resnet18", learning_rate=0.001, epochs=4
    )
    train_dataloader = get_dataloaders(experiment)

    # Test the first batch
    batch = next(iter(train_dataloader))
    print("Image shape:", batch["pixel_values"].shape)
    print("Available keys:", batch.keys())
    print("Objects:", batch["objects"])

# %%
