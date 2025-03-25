import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torchvision import tv_tensors
from config import DEVICE, MINI_RUN, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_ANNOTATIONS_JSON, VAL_ANNOTATIONS_JSON
from experiment_config import ExperimentConfig
from visualize import visualize_segmentation
import json

MAIN_ITEM_NAMES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "pants", "shorts",
    "skirt", "coat", "dress", "jumpsuit", "cape", "glasses", "hat", "headband, head covering, hair accessory",
    "tie", "glove", "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe", "bag, wallet", "scarf",
    "umbrella"
]
NUM_CLASSES = len(MAIN_ITEM_NAMES) + 1

def load_category_mappings(ann_file):
    with open(ann_file, "r") as f:
        categories = {c["id"]: c["name"] for c in json.load(f)["categories"]}
    main_category_ids = {id: i+1 for i, (id, name) in enumerate(categories.items()) if name in MAIN_ITEM_NAMES}
    return {"id_to_name": {0: "background", **{i: name for i, name in enumerate(MAIN_ITEM_NAMES, 1)}},
            "orig_to_new_id": main_category_ids, "num_classes": NUM_CLASSES}

def decode_mask(ann, coco_obj):
    return coco_mask.decode(ann["segmentation"]) if isinstance(ann["segmentation"], dict) else coco_obj.annToMask(ann)

def create_segmentation_mask(coco_obj, img_id, height, width, mappings):
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_id)):
        if (cat_id := ann["category_id"]) in mappings["orig_to_new_id"]:
            mask = np.where(decode_mask(ann, coco_obj) == 1, mappings["orig_to_new_id"][cat_id], mask)
    return mask

class FashionpediaDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size, transform=None, max_samples=None):
        self.coco, self.mappings = COCO(ann_file), load_category_mappings(ann_file) # loading annotations into memory...
        self.img_ids = [i for c in self.mappings["orig_to_new_id"] for i in self.coco.getImgIds(catIds=c)][:max_samples]
        self.img_dir, self.img_size, self.transform = img_dir, img_size, transform

    def __len__(self): return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(Path(self.img_dir) / img_info["file_name"]).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = create_segmentation_mask(self.coco, img_id, img_info["height"], img_info["width"], self.mappings)
        mask = np.array(Image.fromarray(mask).resize((self.img_size, self.img_size), Image.NEAREST))
        return self.transform(image) if self.transform else image, {"masks": tv_tensors.Mask(torch.tensor(mask).unsqueeze(0)), "labels": torch.tensor(mask), "num_classes": NUM_CLASSES, "class_names": self.mappings["id_to_name"]}

def get_dataloaders(experiment: ExperimentConfig):
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return (
        DataLoader(FashionpediaDataset(TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_JSON, experiment.img_size, transform, max_samples=100 if MINI_RUN else None), batch_size=experiment.batch_size, shuffle=True, num_workers=2, drop_last=True),
        DataLoader(FashionpediaDataset(VAL_IMAGES_DIR, VAL_ANNOTATIONS_JSON, experiment.img_size, transform, max_samples=100 if MINI_RUN else None), batch_size=experiment.batch_size, shuffle=False, num_workers=2)
    )

if __name__ == "__main__":
    experiment = ExperimentConfig(id=0, batch_size=2, model_name="deeplab", learning_rate=0.001, img_size=512)
    train_dataloader, val_dataloader = get_dataloaders(experiment)
    print(f"Dataloaders: {len(train_dataloader.dataset)} training, {len(val_dataloader.dataset)} validation")
    image, target = next(iter(train_dataloader))
    print("Image shape:", image.shape, "Mask shape:", target["masks"].shape, "Classes:", target["num_classes"])
    visualize_segmentation(image[0], target["labels"][0], target["class_names"])
