from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import tv_tensors
from experiment_config import ExperimentConfig
from config import (
    DEVICE,
    MINI_RUN,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_ANNOTATIONS_JSON,
    VAL_ANNOTATIONS_JSON,
)
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from visualize import visualize_segmentation


def load_category_mappings(ann_file):
    with open(ann_file, "r") as f:
        dataset = json.load(f)
    categories = dataset["categories"]

    orig_id_to_name = {}
    name_to_orig_id = {}
    for cat in categories:
        cat_id = cat["id"]
        cat_name = cat["name"]
        orig_id_to_name[cat_id] = cat_name
        name_to_orig_id[cat_name] = cat_id

    # Keep original category IDs and return total number of categories
    max_id = max(cat["id"] for cat in categories)
    num_classes = max_id + 1

    return {
        "orig_id_to_name": orig_id_to_name,
        "name_to_orig_id": name_to_orig_id,
        "num_classes": num_classes,
    }


def decode_rle_mask(rle, height, width):
    """Decode RLE encoded mask to binary mask"""
    mask = coco_mask.decode(rle)
    return mask


def create_segmentation_mask(coco_obj, img_id, height, width, mappings):
    mask = np.zeros((height, width), dtype=np.uint8)
    ann_ids = coco_obj.getAnnIds(imgIds=img_id)
    anns = coco_obj.loadAnns(ann_ids)
    masks_with_cats = []

    for ann in anns:
        cat_id = ann["category_id"]
        if "segmentation" in ann:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                binary_mask = decode_rle_mask(seg, height, width)
            elif isinstance(seg, list):
                binary_mask = coco_obj.annToMask(ann)
            else:
                continue
            area = ann.get("area", np.sum(binary_mask))
            masks_with_cats.append((binary_mask, cat_id, area))

    masks_with_cats.sort(key=lambda x: x[2])
    for binary_mask, cat_id, _ in masks_with_cats:
        mask = np.where(binary_mask == 1, cat_id, mask)
    return mask


class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        """Resize both image and mask preserving aspect ratio"""
        # Resize image using BILINEAR interpolation
        image = image.resize((self.size, self.size), Image.BILINEAR)

        # Resize mask using NEAREST interpolation to preserve class IDs
        if mask is not None:
            mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)
            mask = np.array(mask)

        return image, mask


class FashionpediaSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size, transform=None, max_samples=None):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.img_size = img_size
        self.transform = transform
        self.max_samples = max_samples

        self.coco = COCO(ann_file)
        self.mappings = load_category_mappings(ann_file)

        # Get all image IDs (no filtering by main categories)
        self.img_ids = self.coco.getImgIds()
        self.img_ids = list(set(self.img_ids))

        if self.max_samples and self.max_samples < len(self.img_ids):
            self.img_ids = self.img_ids[: self.max_samples]

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

        # TODO: check that the mask is being created correctly
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
            "masks": tv_tensors.Mask(mask_tensor.unsqueeze(0), device=torch.device(DEVICE)),
            "labels": mask_tensor,
            "num_classes": int(self.mappings["num_classes"]),
            "class_names": self.mappings["orig_id_to_name"],  # Keep original names
        }
        return image, target


def get_dataloaders(experiment: ExperimentConfig):
    # TODO: add more preprocessing steps for trying data augmentation
    # Define transforms
    transform = T.Compose(
        [
            T.Resize((experiment.img_size, experiment.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = FashionpediaSegmentationDataset(
        img_dir=TRAIN_IMAGES_DIR,
        ann_file=TRAIN_ANNOTATIONS_JSON,
        img_size=experiment.img_size,
        transform=transform,
        max_samples=100 if MINI_RUN else None,
    )

    val_dataset = FashionpediaSegmentationDataset(
        img_dir=VAL_IMAGES_DIR,
        ann_file=VAL_ANNOTATIONS_JSON,
        img_size=experiment.img_size,
        transform=transform,
        max_samples=100 if MINI_RUN else None,
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


# Use this to run a quick test
if __name__ == "__main__":
    experiment = ExperimentConfig(
        id=0,
        batch_size=2,
        model_name="resnet18",
        learning_rate=0.001,
        img_size=512,
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
