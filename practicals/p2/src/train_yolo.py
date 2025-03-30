from ultralytics import YOLO
import yaml
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm

# Import our metrics module
from yolo_comprehensive_metrics import ComprehensiveMetricsCallback, evaluate_model_comprehensive
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import (
    DATA_DIR,
    ANNOTATIONS_DIR
)

def visualize_dataset_samples(data_yaml_path, num_samples=3, output_dir='dataset_samples'):
    """
    Visualize sample images and labels from the dataset to verify segmentation format
    
    Args:
        data_yaml_path (str): Path to dataset YAML file
        num_samples (int): Number of samples to visualize
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset info
    with open(data_yaml_path, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    # Get paths
    dataset_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    train_images_dir = os.path.join(dataset_dir, dataset_info['train'])
    train_labels_dir = os.path.join(dataset_dir, 'labels', os.path.basename(dataset_info['train']))
    
    # Get class names
    class_names = dataset_info['names']
    
    # Get sample images
    import glob
    image_files = glob.glob(os.path.join(train_images_dir, '*.jpg')) + glob.glob(os.path.join(train_images_dir, '*.png'))
    if len(image_files) == 0:
        print(f"No images found in {train_images_dir}")
        return
    
    # Limit number of samples
    image_files = image_files[:num_samples]
    
    # Process each sample
    for img_path in image_files:
        try:
            print(f"Processing {os.path.basename(img_path)}")
            
            # Get corresponding label path
            label_path = os.path.join(train_labels_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            
            if not os.path.exists(label_path):
                print(f"Label file not found: {label_path}")
                continue
            
            # Read image
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            # Read label file
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            print(f"  Found {len(lines)} annotation lines")
            
            # Create visualization
            vis_img = img_rgb.copy()
            
            # Colors for different classes
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
            ]
            
            # Draw segmentation polygons
            has_segmentation = False
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                
                if len(parts) < 5:  # Skip if not enough points for polygon
                    continue
                
                # Get class index and color
                class_idx = int(float(parts[0]))
                color = colors[class_idx % len(colors)]
                
                # Get segmentation points (normalized)
                seg_points = list(map(float, parts[1:]))
                
                # Check if this looks like segmentation (more than 4 coordinates)
                if len(seg_points) <= 4:
                    continue
                
                has_segmentation = True
                
                # Convert normalized coordinates to absolute coordinates
                points = []
                for j in range(0, len(seg_points), 2):
                    if j+1 < len(seg_points):  # Ensure we have both x and y
                        x = int(seg_points[j] * width)
                        y = int(seg_points[j+1] * height)
                        points.append([x, y])
                
                # Draw polygon
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    points_array = np.array([points], dtype=np.int32)
                    cv2.polylines(vis_img, points_array, True, color, 2)
                    
                    # Add class label
                    class_name = class_names.get(str(class_idx), f"Class {class_idx}")
                    cv2.putText(vis_img, class_name, (points[0][0], points[0][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save visualization
            if has_segmentation:
                output_path = os.path.join(output_dir, f"vis_{os.path.basename(img_path)}")
                plt.figure(figsize=(10, 8))
                plt.imshow(vis_img)
                plt.title(f"Segmentation Visualization - {os.path.basename(img_path)}")
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                print(f"  Saved visualization to {output_path}")
            else:
                print(f"  No valid segmentation data found in {label_path}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Visualizations saved to {output_dir}")

def train_yolo_with_metrics(
    data_yaml_path,
    model_name='yolov8n-seg.pt',
    epochs=100,
    image_size=640,
    batch_size=16,
    device=0,
    project_name='fashionpedia_segmentation',
    output_dir='./results',
    debug=True
):
    """
    Train a YOLO segmentation model with comprehensive metrics tracking
    
    Args:
        data_yaml_path (str): Path to dataset YAML file
        model_name (str): Pretrained model name or path
        epochs (int): Number of training epochs
        image_size (int): Input image size
        batch_size (int): Batch size
        device (int): Device number (0 for first GPU, 'cpu' for CPU)
        project_name (str): Project name for saving results
        output_dir (str): Directory to save metrics and visualizations
        debug (bool): Whether to enable debug mode
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # First, visualize some dataset samples to verify it's correctly formatted
    if debug:
        print("\n=== Visualizing dataset samples to verify segmentation format ===")
        visualize_dataset_samples(data_yaml_path, 3, os.path.join(output_dir, 'dataset_samples'))
    
    # Load dataset info to get number of classes
    with open(data_yaml_path, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    if debug:
        print("\n=== Dataset YAML Content ===")
        for key, value in dataset_info.items():
            if key != 'names':
                print(f"{key}: {value}")
            else:
                print(f"names: {len(value)} classes")
    
    num_classes = len(dataset_info['names'])  # Number of classes in the dataset
    
    # Initialize model
    if debug:
        print(f"\n=== Initializing model {model_name} ===")
        print(f"Using device: {device}")
    
    try:
        model = YOLO(model_name)
        
        if debug:
            print(f"Model loaded successfully")
            print(f"Model task: {model.task}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise e
    
    # Create metrics callback
    metrics_callback = ComprehensiveMetricsCallback(
        num_classes=num_classes,
        output_dir=os.path.join(output_dir, 'metrics_log')
    )
    
    def debug_callback(trainer):
        # This will confirm if callbacks are being called
        print("Callback triggered during training!")
        
        # Check what's available in the trainer object
        if hasattr(trainer, 'batch'):
            if isinstance(trainer.batch, dict):
                print(f"Batch keys: {list(trainer.batch.keys())}")
                
                # Check if 'mask' is in the batch
                if 'mask' in trainer.batch:
                    print(f"Mask shape: {trainer.batch['mask'].shape}")
                    print(f"Mask unique values: {torch.unique(trainer.batch['mask']).tolist()}")
                else:
                    print("No 'mask' key found in batch")
            else:
                # If it's not a dict, see what type it is
                print(f"Batch is not a dict, it's: {type(trainer.batch)}")
                
                # If it's a list or tuple, check the first item
                if isinstance(trainer.batch, (list, tuple)) and len(trainer.batch) > 0:
                    print(f"First batch item type: {type(trainer.batch[0])}")
        else:
            print("Trainer has no 'batch' attribute")
            
        # Also check for loss values
        if hasattr(trainer, 'loss'):
            print(f"Loss: {trainer.loss}")
    
    # Register the debug callback
    model.add_callback("on_train_batch_end", debug_callback)
    model.add_callback("on_train_epoch_end", lambda trainer: print(f"\n=== End of training epoch {trainer.epoch} ===\n"))
    

    # Register the callbacks
    model.add_callback("on_train_epoch_end", metrics_callback.on_train_epoch_end)
    model.add_callback("on_val_end", metrics_callback.on_val_end)
    
    # Train model with callbacks
    if debug:
        print("\n=== Starting training ===")
        print(f"Epochs: {epochs}")
        print(f"Image size: {image_size}")
        print(f"Batch size: {batch_size}")
    
    # Check for CUDA/GPU
    if device != 'cpu' and not torch.cuda.is_available():
        print("Warning: CUDA is not available, falling back to CPU")
        device = 'cpu'
    
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=2, #epochs,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            project=project_name,
            name='with_comprehensive_metrics',
            save=True,
            patience=50,  # Early stopping patience
            verbose=True,
            task='segment',  # Explicitly specify segmentation task
            fraction=0.04  # Use 10% of data instead of 1%
        )
        
        if debug:
            print("\n=== Training complete ===")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e
    
    # Get best model path
    best_model_path = model.trainer.best
    
    # Run comprehensive evaluation on best model
    if debug:
        print(f"\n=== Evaluating best model: {best_model_path} ===")
    
    val_data_path = os.path.join(os.path.dirname(data_yaml_path), dataset_info['val'])
    
    final_metrics = evaluate_model_comprehensive(
        model_path=best_model_path,
        val_data_path=val_data_path,
        dataset_yaml=data_yaml_path,
        output_dir=os.path.join(output_dir, 'final_visualizations'),
        conf_threshold=0.001,  # Lower threshold to capture more predictions
        iou_threshold=0.1     # Lower threshold to be more lenient with overlaps
    )
    
    # Save final metrics to CSV
    import pandas as pd
    metrics_df = pd.DataFrame({k: [v] for k, v in final_metrics.items() if isinstance(v, (int, float))})
    metrics_df.to_csv(os.path.join(output_dir, "final_metrics.csv"), index=False)
    
    # Print final results
    print("\nFinal Metrics:")
    for name, value in final_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")
    
    return best_model_path, final_metrics

if __name__ == "__main__":
    # Example usage
    data_yaml_path = DATA_DIR / "yolo" / "dataset.yaml"
    
    # Enable printing basic information about the process
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    
    # Train model with metrics and debugging enabled
    best_model, metrics = train_yolo_with_metrics(
        data_yaml_path=data_yaml_path,
        model_name='yolov8n-seg.pt',  # Using YOLOv8n-seg which is a standard model
        epochs=100,                   # Train for more epochs
        image_size=640,
        batch_size=16,
        device='cpu',                 # Use CPU for compatibility 
        project_name='fashionpedia_segmentation',
        output_dir= DATA_DIR / "02_metrics" / "yolo_comprehensive_metrics_results",
        debug=True
    )
    
    print(f"Training complete. Best model: {best_model}")