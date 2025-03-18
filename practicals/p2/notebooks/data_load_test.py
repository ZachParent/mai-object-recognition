import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import cv2
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from tqdm.auto import tqdm

print("Libraries imported successfully!")

# Define main garment categories to focus on
main_item_names = [
    'shirt, blouse', 
    'top, t-shirt, sweatshirt', 
    'sweater', 
    'cardigan', 
    'jacket', 
    'vest', 
    'pants', 
    'shorts', 
    'skirt', 
    'coat', 
    'dress', 
    'jumpsuit', 
    'cape', 
    'glasses', 
    'hat', 
    'headband, head covering, hair accessory', 
    'tie', 
    'glove', 
    'watch', 
    'belt', 
    'leg warmer', 
    'tights, stockings', 
    'sock', 
    'shoe', 
    'bag, wallet', 
    'scarf', 
    'umbrella'
]

def setup_fashionpedia(data_dir):
    """
    Setup paths for Fashionpedia dataset.
    """
    train_img_dir = os.path.join(data_dir, 'images', 'train')
    val_img_dir = os.path.join(data_dir, 'images', 'val')
    
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_attributes_train2020.json')
    val_ann_file = os.path.join(data_dir, 'annotations', 'instances_attributes_val2020.json')
    
    # Verify paths exist
    for path in [train_img_dir, val_img_dir, train_ann_file, val_ann_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
    
    return {
        'train_img_dir': train_img_dir,
        'val_img_dir': val_img_dir,
        'train_ann_file': train_ann_file,
        'val_ann_file': val_ann_file
    }

def load_fashionpedia_categories(ann_file):
    """
    Load Fashionpedia categories and create id mappings for our main garments.
    """
    with open(ann_file, 'r') as f:
        dataset = json.load(f)
    
    categories = dataset['categories']
    
    # Use dictionary comprehensions for more concise mapping creation
    orig_id_to_name = {cat['id']: cat['name'] for cat in categories}
    name_to_orig_id = {cat['name']: cat['id'] for cat in categories}
    
    # Get main category ids from our target list
    main_category_ids = [name_to_orig_id[name] for name in main_item_names if name in name_to_orig_id]
    
    # Create our own consecutive ids for the categories, with 0 as background
    id_to_name = {0: 'background'}
    name_to_id = {'background': 0}
    orig_id_to_new_id = {}
    
    for i, name in enumerate(main_item_names):
        new_id = i + 1  # Reserve 0 for background
        id_to_name[new_id] = name
        name_to_id[name] = new_id
        if name in name_to_orig_id:
            orig_id_to_new_id[name_to_orig_id[name]] = new_id
    
    num_classes = len(main_item_names) + 1  # Including background
    
    return {
        'orig_id_to_name': orig_id_to_name,
        'name_to_orig_id': name_to_orig_id,
        'id_to_name': id_to_name,
        'name_to_id': name_to_id,
        'orig_id_to_new_id': orig_id_to_new_id,
        'main_category_ids': main_category_ids,
        'num_classes': num_classes
    }

def decode_rle_mask(rle, height, width):
    """
    Decode RLE encoded mask to binary mask.
    """
    return coco_mask.decode(rle)

def create_segmentation_mask(coco, img_id, height, width, mappings):
    """
    Create segmentation mask from COCO annotations.
    """
    # Initialize empty mask with zeros (background)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    masks_with_cats = []
    
    for ann in anns:
        cat_id = ann['category_id']
        # Skip if not in main categories
        if cat_id not in mappings['orig_id_to_new_id']:
            continue
        
        new_cat_id = mappings['orig_id_to_new_id'][cat_id]
        seg = ann.get('segmentation')
        if seg is None:
            continue
        
        # Decode segmentation mask based on its type
        if isinstance(seg, dict):  # RLE format
            binary_mask = decode_rle_mask(seg, height, width)
        elif isinstance(seg, list):  # Polygon format
            binary_mask = coco.annToMask(ann)
        else:
            continue
        
        area = ann.get('area', np.sum(binary_mask))
        masks_with_cats.append((binary_mask, new_cat_id, area))
    
    # Sort by area (ascending) so smaller objects overlay larger ones
    masks_with_cats.sort(key=lambda x: x[2])
    
    # Update mask in place to avoid extra memory allocation
    for binary_mask, category_id, _ in masks_with_cats:
        mask[binary_mask == 1] = category_id  # in-place assignment
    
    return mask

class FashionpediaSegmentationDataset(data.Dataset):
    def __init__(self, img_dir, ann_file, category_mappings, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.mappings = category_mappings
        self.transform = transform
        self.target_transform = target_transform
        
        # Collect image IDs from main categories (using set comprehension to remove duplicates)
        self.img_ids = list({
            img_id
            for cat_id in self.mappings['main_category_ids']
            for img_id in self.coco.getImgIds(catIds=[cat_id])
        })
        
        print(f"Found {len(self.img_ids)} images containing main garment categories")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # Load image info (assume one image per id)
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Use a context manager to ensure the file is closed after loading
        with PILImage.open(img_path) as img:
            image = img.convert('RGB')
        
        height, width = image.height, image.width
        
        # Create segmentation mask
        mask = create_segmentation_mask(self.coco, img_id, height, width, self.mappings)
        
        if self.transform is not None:
            image = self.transform(image)
        
        mask = torch.from_numpy(mask).long()
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        
        return image, mask

def get_transforms(split):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def target_transform(mask):
    # Resize mask to match image size (512x512) using nearest-neighbor interpolation
    mask = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze(0)
    return mask

def create_data_loaders(data_paths, category_mappings, batch_size=8, num_workers=4):
    # Create train and validation datasets
    train_dataset = FashionpediaSegmentationDataset(
        data_paths['train_img_dir'],
        data_paths['train_ann_file'],
        category_mappings,
        transform=get_transforms('train'),
        target_transform=target_transform
    )
    
    val_dataset = FashionpediaSegmentationDataset(
        data_paths['val_img_dir'],
        data_paths['val_ann_file'],
        category_mappings,
        transform=get_transforms('validation'),
        target_transform=target_transform
    )
    
    # Use pin_memory and drop_last for efficient loading
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def visualize_segmentation(image, mask, id_to_name):
    # Create a colormap for visualization
    cmap = plt.cm.get_cmap('tab20', len(id_to_name))
    
    plt.figure(figsize=(16, 8))
    
    # Display original image
    plt.subplot(1, 2, 1)
    if isinstance(image, torch.Tensor):
        # Denormalize and convert to numpy for visualization
        img = image.permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
    else:
        img = image
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display segmentation mask
    plt.subplot(1, 2, 2)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Create a colored mask without creating unnecessary large arrays
    colored_mask = np.zeros((*mask.shape, 3))
    for class_id in np.unique(mask):
        if class_id == 0:  # Skip background
            continue
        class_mask = mask == class_id
        color = cmap(class_id)[:3]  # Get RGB from colormap
        colored_mask[class_mask] = color  # in-place assignment
    
    plt.imshow(colored_mask)
    
    # Create a legend
    handles = []
    for class_id in sorted(np.unique(mask)):
        if class_id in id_to_name:
            color = cmap(class_id)[:3]
            patch = plt.Rectangle((0, 0), 1, 1, fc=color)
            handles.append((patch, id_to_name[class_id]))
    
    if handles:
        patches, labels = zip(*handles)
        plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_dataset_distribution(ann_file, category_mappings, output_csv="distribution_summary.csv", output_plot_prefix="distribution"):
    import pandas as pd
    """
    Analyzes the dataset for label distribution by:
      a. Counting objects per label.
      b. Counting images per label (each image counts once per label).
      c. Computing the mean and variance of the area ratio (object bbox area / image area).
    
    Results are saved in a CSV file and several plots.
    
    Args:
        ann_file (str): Path to the annotation JSON file.
        category_mappings (dict): The mappings returned by load_fashionpedia_categories.
        output_csv (str): Path for the CSV output.
        output_plot_prefix (str): Prefix for the saved plot filenames.
    """
    # Load the annotation data
    with open(ann_file, 'r') as f:
        dataset = json.load(f)
    
    # Build a mapping from image id to its (width, height)
    image_info = {}
    for img in dataset.get("images", []):
        image_info[img["id"]] = (img["width"], img["height"])
    
    # Initialize statistics dictionary for each main label (skip background with id 0)
    distribution = {}
    for new_id, label in category_mappings["id_to_name"].items():
        if new_id == 0:
            continue
        distribution[label] = {
            "object_count": 0,
            "image_ids": set(),   # to count images uniquely per label
            "area_ratios": []     # list to store area ratios for each object
        }
    
    # Iterate over each annotation in the dataset
    for ann in dataset.get("annotations", []):
        orig_cat_id = ann["category_id"]
        # Skip annotations not in our selected main categories
        if orig_cat_id not in category_mappings["orig_id_to_new_id"]:
            continue
        new_id = category_mappings["orig_id_to_new_id"][orig_cat_id]
        label = category_mappings["id_to_name"].get(new_id, "Unknown")
        
        # Update object count and unique image ids
        distribution[label]["object_count"] += 1
        image_id = ann["image_id"]
        distribution[label]["image_ids"].add(image_id)
        
        # Compute area ratio using the bounding box (COCO bbox: [x, y, width, height])
        bbox = ann.get("bbox")
        if bbox is not None and image_id in image_info:
            bbox_area = bbox[2] * bbox[3]
            img_width, img_height = image_info[image_id]
            img_area = img_width * img_height
            area_ratio = bbox_area / img_area
            distribution[label]["area_ratios"].append(area_ratio)
    
    # Prepare summary data for CSV output
    summary_data = []
    for label, stats in distribution.items():
        object_count = stats["object_count"]
        image_count = len(stats["image_ids"])
        if stats["area_ratios"]:
            mean_area_ratio = np.mean(stats["area_ratios"])
            var_area_ratio = np.var(stats["area_ratios"])
        else:
            mean_area_ratio = 0
            var_area_ratio = 0
        summary_data.append({
            "label": label,
            "object_count": object_count,
            "image_count": image_count,
            "mean_area_ratio": mean_area_ratio,
            "var_area_ratio": var_area_ratio
        })
    
    # Save summary as CSV using pandas
    df = pd.DataFrame(summary_data)
    df.to_csv(output_csv, index=False)
    print(f"CSV summary saved to {output_csv}")
    
    # --- Plotting ---
    import matplotlib.pyplot as plt
    labels = [entry["label"] for entry in summary_data]
    object_counts = [entry["object_count"] for entry in summary_data]
    image_counts = [entry["image_count"] for entry in summary_data]
    mean_area_ratios = [entry["mean_area_ratio"] for entry in summary_data]
    
    # Plot 1: Bar plot for object counts per label
    plt.figure(figsize=(10, 6))
    plt.bar(labels, object_counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Label")
    plt.ylabel("Object Count")
    plt.title("Object Count per Label")
    plt.tight_layout()
    plt.savefig(f"{output_plot_prefix}_object_count.png")
    plt.close()
    
    # Plot 2: Bar plot for image counts per label
    plt.figure(figsize=(10, 6))
    plt.bar(labels, image_counts, color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Label")
    plt.ylabel("Image Count")
    plt.title("Image Count per Label")
    plt.tight_layout()
    plt.savefig(f"{output_plot_prefix}_image_count.png")
    plt.close()
    
    # Plot 3: Bar plot for mean area ratio per label
    plt.figure(figsize=(10, 6))
    plt.bar(labels, mean_area_ratios, color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Label")
    plt.ylabel("Mean Area Ratio")
    plt.title("Mean Area Ratio per Label")
    plt.tight_layout()
    plt.savefig(f"{output_plot_prefix}_mean_area_ratio.png")
    plt.close()
    
    # Plot 4: Box plot showing area ratio distributions per label
    plt.figure(figsize=(12, 8))
    area_ratio_data = [distribution[label]["area_ratios"] for label in labels]
    plt.boxplot(area_ratio_data, labels=labels)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Label")
    plt.ylabel("Area Ratio")
    plt.title("Area Ratio Distribution per Label")
    plt.tight_layout()
    plt.savefig(f"{output_plot_prefix}_area_ratio_boxplot.png")
    plt.close()
    
    print("Plots saved successfully.")