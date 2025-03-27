import sys
import os
import json
import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm  # for progress bar
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import (
    DATA_DIR,
    ANNOTATIONS_DIR
)

def convert_coco_to_yolo(json_path, images_src_dir, images_dst_dir, labels_dst_dir, use_segments=True, max_class_id=25):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        json_path (Path): Path to COCO JSON annotation file
        images_src_dir (Path): Source directory containing original images
        images_dst_dir (Path): Destination directory for image symlinks
        labels_dst_dir (Path): Destination directory for YOLO labels
        use_segments (bool): Whether to include segmentation masks
        max_class_id (int): Maximum class ID to include (0-indexed)
    
    Returns:
        int: Number of processed images
    """
    # Ensure directories exist
    images_dst_dir.mkdir(exist_ok=True, parents=True)
    labels_dst_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Build image id to filename mapping
    image_id_to_info = {}
    for img in data['images']:
        image_id_to_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # Group annotations by image and filter by category_id
    annotations = defaultdict(list)
    filtered_count = 0
    for ann in data["annotations"]:
        # Only include annotations with category_id <= max_class_id + 1 (COCO is 1-indexed)
        if ann["category_id"] <= max_class_id:
            annotations[ann["image_id"]].append(ann)
        else:
            filtered_count += 1
    
    print(f"Converting {len(annotations)} images, filtered out {filtered_count} annotations with class ID > {max_class_id}")
    
    # Process each image
    processed_count = 0
    images_with_annotations = 0
    
    for img_id, anns in tqdm(annotations.items(), desc=f"Converting {json_path.stem}"):
        if img_id not in image_id_to_info:
            continue
        
        img = image_id_to_info[img_id]
        h, w = img["height"], img["width"]
        f = img["file_name"]
        
        # Create label file
        label_file = (labels_dst_dir / f).with_suffix(".txt")
        
        bboxes = []
        segments = []
        
        # Process annotations for this image
        for ann in anns:
            if ann.get("iscrowd", False):
                continue
            
            # COCO format: [top left x, top left y, width, height]
            box = np.array(ann["bbox"], dtype=np.float64)
            
            # Convert to YOLO format: [center_x, center_y, width, height]
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            
            if box[2] <= 0 or box[3] <= 0:  # Skip invalid boxes
                continue
            
            # Class ID (0-indexed in YOLO)
            cls = ann["category_id"] - 1
            
            # Format for YOLO: [class, x_center, y_center, width, height]
            box = [cls] + box.tolist()
            
            if box not in bboxes:
                bboxes.append(box)
                
                # Handle segmentation if present and requested
                if use_segments and ann.get("segmentation") is not None:
                    seg = ann["segmentation"]
                    if seg and isinstance(seg, list):
                        # Handle different segmentation formats
                        if isinstance(seg[0], list):
                            # Format: [[x1,y1,x2,y2,...]]
                            s = seg[0]
                        elif isinstance(seg[0], dict):
                            # Some other format, skip
                            continue
                        else:
                            # Format: [x1,y1,x2,y2,...]
                            s = seg
                            
                        # Convert to normalized coordinates
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        segments.append(s)
        
        # Skip images with no annotations after filtering
        if not bboxes:
            continue
            
        images_with_annotations += 1
        
        # Write label file
        with open(label_file, "w", encoding="utf-8") as file:
            for i in range(len(bboxes)):
                if use_segments and i < len(segments):
                    # Write segments
                    values = segments[i]
                    line = " ".join([f"{x}" for x in values])
                else:
                    # Write bounding box
                    values = bboxes[i]
                    line = " ".join([f"{x}" for x in values])
                
                file.write(line + "\n")
        
        # Create symlink for image
        src_img = images_src_dir / f
        dst_img = images_dst_dir / f
        
        if src_img.exists() and not dst_img.exists():
            try:
                os.symlink(src_img, dst_img)
            except OSError as e:
                # If symlink fails, copy the file
                shutil.copy2(src_img, dst_img)
        
        processed_count += 1
    
    print(f"Images with annotations after filtering: {images_with_annotations}")
    
    # Filter categories to include only the first 27
    filtered_categories = [cat for cat in data["categories"] if cat["id"] <= max_class_id]
    
    return processed_count, filtered_categories

# Clean up previous attempts
yolo_dir = DATA_DIR / "yolo"
if yolo_dir.exists():
    shutil.rmtree(yolo_dir)

# Create output directory structure
output_dir = yolo_dir
output_dir.mkdir(exist_ok=True, parents=True)
(output_dir / "labels" / "train").mkdir(exist_ok=True, parents=True)
(output_dir / "labels" / "val").mkdir(exist_ok=True, parents=True)
(output_dir / "images" / "train").mkdir(exist_ok=True, parents=True)
(output_dir / "images" / "val").mkdir(exist_ok=True, parents=True)

# Define paths
train_json = ANNOTATIONS_DIR / "instances_attributes_train2020.json"
val_json = ANNOTATIONS_DIR / "instances_attributes_val2020.json"

train_images_src = DATA_DIR / "00_raw" / "images" / "train"
val_images_src = DATA_DIR / "00_raw" / "images" / "val"

train_images_dst = output_dir / "images" / "train"
val_images_dst = output_dir / "images" / "val"

train_labels_dst = output_dir / "labels" / "train"
val_labels_dst = output_dir / "labels" / "val"

# Convert training set
train_count, categories = convert_coco_to_yolo(
    train_json, 
    train_images_src, 
    train_images_dst, 
    train_labels_dst,
    use_segments=True,
    max_class_id=26
)

# Convert validation set
val_count, _ = convert_coco_to_yolo(
    val_json, 
    val_images_src, 
    val_images_dst, 
    val_labels_dst,
    use_segments=True,
    max_class_id=26
)

# Create dataset.yaml
max_class_id = 26  # 0-indexed, so this will include classes 0-27
category_names = {cat["id"]: cat["name"] for cat in categories if cat["id"] <= max_class_id}

yaml_content = f"""# YOLOv5 dataset config
path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
"""

for cat_id in sorted(category_names.keys()):
    yaml_content += f"  {cat_id-1}: {category_names[cat_id]}\n" 

yaml_path = output_dir / "dataset.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

# Statistics
train_labels = len(list((output_dir / "labels" / "train").glob("*.txt")))
val_labels = len(list((output_dir / "labels" / "val").glob("*.txt")))
train_images = len(list((output_dir / "images" / "train").glob("*")))
val_images = len(list((output_dir / "images" / "val").glob("*")))

print(f"\nConversion completed!")
print(f"Training set: {train_count} images processed, {train_labels} label files, {train_images} images linked")
print(f"Validation set: {val_count} images processed, {val_labels} label files, {val_images} images linked")
print(f"Results saved to {output_dir}")
print(f"Dataset config: {yaml_path}")