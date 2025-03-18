# %%
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import tv_tensors
from experiment_config import ExperimentConfig
from config import (
    DEVICE,
    USING_CUDA,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_ANNOTATIONS_JSON,
    VAL_ANNOTATIONS_JSON,
)
from torchvision.transforms import Compose, ColorJitter, ToTensor, ToPILImage, Resize
import torch
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import json
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from visualize import visualize_segmentation

# Define main garment categories to focus on
main_item_names = [
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit",
    "cape",
    "glasses",
    "hat",
    "headband, head covering, hair accessory",
    "tie",
    "glove",
    "watch",
    "belt",
    "leg warmer",
    "tights, stockings",
    "sock",
    "shoe",
    "bag, wallet",
    "scarf",
    "umbrella",
]


def load_category_mappings(ann_file):
    """
    Load Fashionpedia categories and create id mappings for main garments
    """
    with open(ann_file, "r") as f:
        dataset = json.load(f)

    categories = dataset["categories"]

    # Create mappings
    orig_id_to_name = {}
    name_to_orig_id = {}

    for cat in categories:
        cat_id = cat["id"]
        cat_name = cat["name"]
        orig_id_to_name[cat_id] = cat_name
        name_to_orig_id[cat_name] = cat_id

    # Create our selected mapping (main garments only)
    main_category_ids = []
    for name in main_item_names:
        if name in name_to_orig_id:
            main_category_ids.append(name_to_orig_id[name])

    # Create our own consecutive ids for the categories
    id_to_name = {0: "background"}
    name_to_id = {"background": 0}
    orig_id_to_new_id = {}

    for i, name in enumerate(main_item_names):
        new_id = i + 1  # +1 for background
        id_to_name[new_id] = name
        name_to_id[name] = new_id
        if name in name_to_orig_id:
            orig_id_to_new_id[name_to_orig_id[name]] = new_id

    num_classes = len(main_item_names) + 1  # +1 for background

    return {
        "orig_id_to_name": orig_id_to_name,
        "name_to_orig_id": name_to_orig_id,
        "id_to_name": id_to_name,
        "name_to_id": name_to_id,
        "orig_id_to_new_id": orig_id_to_new_id,
        "main_category_ids": main_category_ids,
        "num_classes": num_classes,
    }


def decode_rle_mask(rle, height, width):
    """Decode RLE encoded mask to binary mask"""
    mask = coco_mask.decode(rle)
    return mask


def create_segmentation_mask(coco_obj, img_id, height, width, mappings):
    """
    Create segmentation mask from COCO annotations
    """
    # Initialize empty mask with zeros (background)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Get annotations for this image
    ann_ids = coco_obj.getAnnIds(imgIds=img_id)
    anns = coco_obj.loadAnns(ann_ids)

    # Create a list to store masks and category IDs for ordering
    masks_with_cats = []

    for ann in anns:
        cat_id = ann["category_id"]

        # Check if this category is in our main categories
        if cat_id not in mappings["orig_id_to_new_id"]:
            continue

        # Get our new category ID
        new_cat_id = mappings["orig_id_to_new_id"][cat_id]

        # Get segmentation
        if "segmentation" in ann:
            seg = ann["segmentation"]
            if isinstance(seg, dict):  # RLE format
                binary_mask = decode_rle_mask(seg, height, width)
            elif isinstance(seg, list):  # Polygon format
                # Convert polygon to mask using COCO API
                binary_mask = coco_obj.annToMask(ann)
            else:
                continue

            # Store mask with category ID and area (for ordering)
            area = ann.get("area", np.sum(binary_mask))
            masks_with_cats.append((binary_mask, new_cat_id, area))

    # Sort by area (ascending) so smaller objects appear on top
    masks_with_cats.sort(key=lambda x: x[2])

    # Apply masks in sorted order
    for binary_mask, category_id, _ in masks_with_cats:
        mask = np.where(binary_mask == 1, category_id, mask)

    return mask


class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        """Resize both image and mask preserving aspect ratio"""
        # Calculate resize factors
        width, height = image.size
        x_scale = self.size / width
        y_scale = self.size / height

        # Resize image using BILINEAR interpolation
        image = image.resize((self.size, self.size), Image.BILINEAR)

        # Resize mask using NEAREST interpolation to preserve class IDs
        if mask is not None:
            mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)
            mask = np.array(mask)

        return image, mask


class FashionpediaSegmentationDataset(Dataset):
    def __init__(
        self, img_dir, ann_file, img_size=224, transform=None, max_samples=None
    ):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.img_size = img_size
        self.transform = transform
        self.max_samples = max_samples

        # Load COCO API and category mappings
        self.coco = COCO(ann_file)
        self.mappings = load_category_mappings(ann_file)

        # Get image IDs containing our main categories
        self.img_ids = []
        for cat_id in self.mappings["main_category_ids"]:
            cat_img_ids = self.coco.getImgIds(catIds=[cat_id])
            self.img_ids.extend(cat_img_ids)

        # Remove duplicates
        self.img_ids = list(set(self.img_ids))

        # Limit dataset size if specified
        if self.max_samples and self.max_samples < len(self.img_ids):
            self.img_ids = self.img_ids[: self.max_samples]

        # Setup resize transform
        self.resize_transform = ResizeTransform(img_size)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = Path(self.img_dir) / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        # Get image dimensions
        height, width = image.height, image.width

        # Create segmentation mask
        mask = create_segmentation_mask(self.coco, img_id, height, width, self.mappings)

        # Apply resize transform
        image, mask = self.resize_transform(image, mask)

        # Apply additional transforms
        if self.transform:
            image = self.transform(image)

        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).long()

        # For compatibility with detection models, we'll create target dict
        target = {
            "masks": tv_tensors.Mask(
                mask_tensor.unsqueeze(0),  # Add channel dimension
                dtype=torch.long,
                device=torch.device(DEVICE),
            ),
            "labels": mask_tensor,  # Keep original format for segmentation models
            "num_classes": self.mappings["num_classes"],
            "class_names": self.mappings["id_to_name"],
        }

        return image, target


def get_dataloaders(experiment: ExperimentConfig):
    # Define transforms
    transform = T.Compose(
        [
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = FashionpediaSegmentationDataset(
        img_dir=TRAIN_IMAGES_DIR,
        ann_file=TRAIN_ANNOTATIONS_JSON,
        img_size=512,
        transform=transform,
        max_samples=100,
    )

    val_dataset = FashionpediaSegmentationDataset(
        img_dir=VAL_IMAGES_DIR,
        ann_file=VAL_ANNOTATIONS_JSON,
        img_size=512,
        transform=transform,
        max_samples=100,
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=experiment.batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    experiment = ExperimentConfig(
        id=0,
        batch_size=2,
        model_name="segmentation_model",
        learning_rate=0.001,
        epochs=4,
        img_size=512,
        max_samples=10,
        num_workers=0,  # Set to 0 for debugging
    )

    train_dataloader, val_dataloader = get_dataloaders(experiment)
    print(
        f"Created dataloaders with {len(train_dataloader.dataset)} training and {len(val_dataloader.dataset)} validation examples"
    )

    # Test visualization with one sample
    image, target = next(iter(train_dataloader))

    # Print information about the batch
    print("Image batch shape:", image.shape)
    print("Available target keys:", target.keys())
    print("Mask shape:", target["masks"].shape)
    print("Number of classes:", target["num_classes"])

    # Visualize the first image and its mask
    visualize_segmentation(image[0], target["labels"][0], target["class_names"])
# %%
