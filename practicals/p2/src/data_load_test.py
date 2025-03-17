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

# Prepare Fashionpedia data paths
def setup_fashionpedia(data_dir):
    """
    Setup paths for Fashionpedia dataset
    
    Args:
        data_dir: Root directory for Fashionpedia dataset
    
    Returns:
        Dictionary containing paths and data loaders
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

# Load and parse Fashionpedia annotations
def load_fashionpedia_categories(ann_file):
    """
    Load Fashionpedia categories and create id mappings for our main garments
    
    Args:
        ann_file: Path to annotation file
    
    Returns:
        Dictionary with category mappings
    """
    with open(ann_file, 'r') as f:
        dataset = json.load(f)
    
    categories = dataset['categories']
    
    # Create mappings
    orig_id_to_name = {}
    name_to_orig_id = {}
    
    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']
        orig_id_to_name[cat_id] = cat_name
        name_to_orig_id[cat_name] = cat_id
    
    # Create our selected mapping (main garments only)
    main_category_ids = []
    for name in main_item_names:
        if name in name_to_orig_id:
            main_category_ids.append(name_to_orig_id[name])
    
    # Create our own consecutive ids for the categories
    id_to_name = {0: 'background'}
    name_to_id = {'background': 0}
    orig_id_to_new_id = {}
    
    for i, name in enumerate(main_item_names):
        new_id = i + 1  # +1 for background
        id_to_name[new_id] = name
        name_to_id[name] = new_id
        if name in name_to_orig_id:
            orig_id_to_new_id[name_to_orig_id[name]] = new_id
    
    num_classes = len(main_item_names) + 1  # +1 for background
    
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
    Decode RLE encoded mask to binary mask
    """
    mask = coco_mask.decode(rle)
    return mask

def create_segmentation_mask(coco, img_id, height, width, mappings):
    """
    Create segmentation mask from COCO annotations
    
    Args:
        coco: COCO API object
        img_id: Image ID
        height: Image height
        width: Image width
        mappings: Category mappings
    
    Returns:
        Segmentation mask as numpy array
    """
    # Initialize empty mask with zeros (background)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    # Create a list to store masks and category IDs for ordering
    masks_with_cats = []
    
    for ann in anns:
        cat_id = ann['category_id']
        
        # Check if this category is in our main categories
        if cat_id not in mappings['orig_id_to_new_id']:
            continue
        
        # Get our new category ID
        new_cat_id = mappings['orig_id_to_new_id'][cat_id]
        
        # Get segmentation
        if 'segmentation' in ann:
            seg = ann['segmentation']
            if isinstance(seg, dict):  # RLE format
                binary_mask = decode_rle_mask(seg, height, width)
            elif isinstance(seg, list):  # Polygon format
                # Convert polygon to mask
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                # COCO API can convert polygons to masks
                binary_mask = coco.annToMask(ann)
            else:
                continue
                
            # Store mask with category ID and area (for ordering)
            area = ann.get('area', np.sum(binary_mask))
            masks_with_cats.append((binary_mask, new_cat_id, area))
    
    # Sort by area (ascending) so smaller objects appear on top
    masks_with_cats.sort(key=lambda x: x[2])
    
    # Apply masks in sorted order
    for binary_mask, category_id, _ in masks_with_cats:
        mask = np.where(binary_mask == 1, category_id, mask)
    
    return mask

class FashionpediaSegmentationDataset(data.Dataset):
    def __init__(self, img_dir, ann_file, category_mappings, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.mappings = category_mappings
        self.transform = transform
        self.target_transform = target_transform
        
        # Get all image IDs containing our main categories
        self.img_ids = []
        for cat_id in self.mappings['main_category_ids']:
            cat_img_ids = self.coco.getImgIds(catIds=[cat_id])
            self.img_ids.extend(cat_img_ids)
        
        # Remove duplicates
        self.img_ids = list(set(self.img_ids))
        
        print(f"Found {len(self.img_ids)} images containing main garment categories")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = PILImage.open(img_path).convert('RGB')
        
        # Get image size
        height, width = image.height, image.width
        
        # Create segmentation mask
        mask = create_segmentation_mask(
            self.coco, 
            img_id, 
            height, 
            width, 
            self.mappings
        )
        
        # Apply transformations
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
    # Resize mask to match image size (512x512)
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
    
    # Create data loaders
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
    
    # Create a colored mask
    colored_mask = np.zeros((*mask.shape, 3))
    for class_id in np.unique(mask):
        if class_id == 0:  # Background
            continue
        
        # Create color mask
        class_mask = mask == class_id
        color = cmap(class_id)[:3]  # RGB
        for i in range(3):
            colored_mask[class_mask, i] = color[i]
    
    plt.imshow(colored_mask)
    
    # Create a legend
    handles = []
    for class_id in sorted(np.unique(mask)):
        if class_id in id_to_name:
            color = cmap(class_id)[:3]
            patch = plt.Rectangle((0, 0), 1, 1, fc=color)
            handles.append((patch, id_to_name[class_id]))
    
    # Only show legend if there are classes to display
    if handles:
        patches, labels = zip(*handles)
        plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()