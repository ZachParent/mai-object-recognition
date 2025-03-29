"""
YOLO Dataset Validation Script

This script checks if a YOLO dataset is correctly formatted for segmentation tasks.
It verifies:
1. YAML configuration
2. Sample label files to ensure they contain segmentation data
3. Directory structure
"""

import os
import sys
import yaml
import glob
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

def validate_yaml(yaml_path):
    """Validate the YAML configuration file"""
    print(f"Validating YAML file: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        print(f"ERROR: YAML file not found at {yaml_path}")
        return False
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data:
                print(f"ERROR: Missing required field '{field}' in YAML")
                return False
                
        # Check task field for segmentation
        if 'task' not in data:
            print(f"WARNING: 'task' field not found in YAML, should be 'segment' for segmentation")
        elif data['task'] != 'segment':
            print(f"ERROR: 'task' field is '{data['task']}', but should be 'segment' for segmentation")
            return False
            
        # Check class names
        if not isinstance(data['names'], dict):
            print(f"ERROR: 'names' should be a dictionary mapping class IDs to names")
            return False
            
        if len(data['names']) != data['nc']:
            print(f"WARNING: Number of classes in 'names' ({len(data['names'])}) doesn't match 'nc' ({data['nc']})")
            
        # Check paths
        yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        dataset_path = os.path.join(yaml_dir, data['path']) if not os.path.isabs(data['path']) else data['path']
        
        train_path = os.path.join(dataset_path, data['train'])
        val_path = os.path.join(dataset_path, data['val'])
        
        if not os.path.exists(train_path):
            print(f"ERROR: Train path not found: {train_path}")
            return False
            
        if not os.path.exists(val_path):
            print(f"ERROR: Validation path not found: {val_path}")
            return False
            
        print(f"YAML validation successful!")
        print(f"Dataset path: {dataset_path}")
        print(f"Train path: {train_path}")
        print(f"Val path: {val_path}")
        print(f"Number of classes: {data['nc']}")
        print(f"Task: {data.get('task', 'Not specified (should be segment)')}")
        
        return {
            'dataset_path': dataset_path,
            'train_path': train_path,
            'val_path': val_path,
            'yaml_data': data
        }
        
    except Exception as e:
        print(f"ERROR: Failed to parse YAML file: {str(e)}")
        return False

def validate_label_files(yaml_data, num_to_check=5):
    """Validate a sample of label files to ensure they contain segmentation data"""
    print(f"\nValidating label files...")
    
    dataset_path = yaml_data['dataset_path']
    train_path = yaml_data['train_path']
    val_path = yaml_data['val_path']
    
    # Get label paths corresponding to the train/val image paths
    train_label_path = os.path.join(dataset_path, 'labels', os.path.basename(train_path))
    val_label_path = os.path.join(dataset_path, 'labels', os.path.basename(val_path))
    
    if not os.path.exists(train_label_path):
        print(f"ERROR: Train label path not found: {train_label_path}")
        return False
        
    if not os.path.exists(val_label_path):
        print(f"ERROR: Validation label path not found: {val_label_path}")
        return False
    
    # Check some label files in both train and val
    all_correct = True
    
    for label_path, name in [(train_label_path, "train"), (val_label_path, "val")]:
        label_files = glob.glob(os.path.join(label_path, "*.txt"))
        
        if not label_files:
            print(f"ERROR: No label files found in {label_path}")
            all_correct = False
            continue
            
        print(f"Found {len(label_files)} {name} label files")
        
        # Check a sample of files
        sample_size = min(num_to_check, len(label_files))
        sample_files = label_files[:sample_size]
        
        for label_file in sample_files:
            print(f"\nChecking {os.path.basename(label_file)}:")
            
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                if not lines:
                    print(f"  WARNING: Empty label file")
                    continue
                    
                has_segmentation_data = False
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    
                    if len(parts) < 5:
                        print(f"  ERROR: Line {i+1} has only {len(parts)} parts, not enough for a polygon")
                        continue
                        
                    # A valid segmentation line has a class index followed by point coordinates (x1 y1 x2 y2 ...)
                    # We expect more than 5 values for a polygon (class + at least 2 points)
                    if len(parts) > 5:
                        # First part should be the class index
                        try:
                            class_idx = int(float(parts[0]))
                            point_count = (len(parts) - 1) // 2
                            print(f"  Line {i+1}: Class {class_idx}, {point_count} points - OK")
                            has_segmentation_data = True
                        except ValueError:
                            print(f"  ERROR: Line {i+1} has invalid class index: {parts[0]}")
                    else:
                        # This might be a bounding box format (class, x, y, w, h) - not segmentation
                        print(f"  WARNING: Line {i+1} has exactly 5 values - this looks like a bounding box, not a segmentation")
                
                if not has_segmentation_data:
                    print(f"  ERROR: No valid segmentation data found in this file!")
                    all_correct = False
                    
            except Exception as e:
                print(f"  ERROR: Failed to parse label file: {str(e)}")
                all_correct = False
    
    return all_correct

def visualize_sample_labels(yaml_data, num_to_visualize=2, output_dir='segmentation_samples'):
    """Visualize a few segmentation masks to check their correctness"""
    print(f"\nVisualizing sample segmentation masks...")
    
    dataset_path = yaml_data['dataset_path']
    train_path = yaml_data['train_path']
    class_names = yaml_data['yaml_data']['names']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image paths
    image_files = glob.glob(os.path.join(train_path, "*.jpg")) + glob.glob(os.path.join(train_path, "*.png"))
    
    if not image_files:
        print(f"ERROR: No image files found in {train_path}")
        return False
        
    # Check a sample of files
    sample_size = min(num_to_visualize, len(image_files))
    sample_files = image_files[:sample_size]
    
    for img_path in sample_files:
        print(f"\nVisualizing {os.path.basename(img_path)}:")
        
        # Get corresponding label path
        label_path = os.path.join(
            dataset_path, 
            'labels', 
            os.path.basename(train_path), 
            os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        )
        
        if not os.path.exists(label_path):
            print(f"  ERROR: Corresponding label file not found: {label_path}")
            continue
            
        try:
            # Load image
            image = Image.open(img_path)
            img_width, img_height = image.size
            
            # Create a visualization image
            vis_img = image.copy()
            draw = ImageDraw.Draw(vis_img)
            
            # Load label file
            polygons_by_class = {}
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    # First part is the class index
                    class_idx = int(float(parts[0]))
                    
                    # Get segmentation points (normalized)
                    seg_points = list(map(float, parts[1:]))
                    
                    # Convert normalized coordinates to absolute coordinates
                    points = []
                    for i in range(0, len(seg_points), 2):
                        x = int(seg_points[i] * img_width)
                        y = int(seg_points[i+1] * img_height)
                        points.append((x, y))
                    
                    # Store polygon by class
                    if class_idx not in polygons_by_class:
                        polygons_by_class[class_idx] = []
                    
                    polygons_by_class[class_idx].append(points)
            
            # Draw polygons with different colors per class
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
            ]
            
            for class_idx, polygons in polygons_by_class.items():
                color = colors[class_idx % len(colors)]
                class_name = class_names.get(str(class_idx), f"Class {class_idx}")
                
                for poly in polygons:
                    if len(poly) >= 3:  # Need at least 3 points for a polygon
                        draw.polygon(poly, outline=color, fill=None)
                        
                        # Add class label near first point
                        if poly:
                            draw.text((poly[0][0], poly[0][1]), class_name, fill=color)
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"vis_{os.path.basename(img_path)}")
            vis_img.save(output_path)
            print(f"  Saved visualization to {output_path}")
            
            # Also create a semantic mask visualization
            semantic_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            for class_idx, polygons in polygons_by_class.items():
                color = colors[class_idx % len(colors)]
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                for poly in polygons:
                    if len(poly) >= 3:
                        cv2_poly = np.array([poly], dtype=np.int32)
                        cv2.fillPoly(mask, cv2_poly, 1)
                
                # Apply color to mask areas
                semantic_mask[mask > 0] = color
            
            # Save semantic mask
            mask_output_path = os.path.join(output_dir, f"mask_{os.path.basename(img_path)}")
            cv2.imwrite(mask_output_path, semantic_mask)
            print(f"  Saved semantic mask to {mask_output_path}")
            
        except Exception as e:
            print(f"  ERROR: Failed to visualize: {str(e)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Validate YOLO dataset for segmentation')
    parser.add_argument('yaml_path', type=str, help='Path to dataset.yaml file')
    parser.add_argument('--samples', type=int, default=3, help='Number of sample files to check')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample segmentation masks')
    parser.add_argument('--output-dir', type=str, default='segmentation_samples', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Validate YAML file
    yaml_data = validate_yaml(args.yaml_path)
    if not yaml_data:
        sys.exit(1)
        
    # Validate label files
    if not validate_label_files(yaml_data, args.samples):
        print("\nWARNING: Some label files may not be properly formatted for segmentation!")
    else:
        print("\nAll checked label files appear to be correctly formatted for segmentation.")
        
    # Visualize sample labels if requested
    if args.visualize:
        visualize_sample_labels(yaml_data, args.samples, args.output_dir)
    
if __name__ == "__main__":
    main()