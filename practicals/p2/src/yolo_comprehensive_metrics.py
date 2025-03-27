import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Define our own Callback base class since it might not be directly available
class Callback:
    """Base callback class to create custom callbacks for YOLO training"""
    def __init__(self):
        pass
    
    def on_train_start(self, trainer):
        pass
    
    def on_train_epoch_start(self, trainer):
        pass
    
    def on_train_batch_start(self, trainer):
        pass
    
    def on_train_batch_end(self, trainer):
        pass
    
    def on_train_epoch_end(self, trainer):
        pass
    
    def on_train_end(self, trainer):
        pass
    
    def on_val_start(self, trainer):
        pass
    
    def on_val_batch_start(self, trainer):
        pass
    
    def on_val_batch_end(self, trainer):
        pass
    
    def on_val_end(self, trainer):
        pass
    
    def on_fit_epoch_end(self, trainer):
        pass
    
    def on_predict_start(self, predictor):
        pass
    
    def on_predict_batch_start(self, predictor):
        pass
    
    def on_predict_batch_end(self, predictor):
        pass
    
    def on_predict_end(self, predictor):
        pass

class SegmentationMetrics:
    def __init__(self, num_classes, include_background=True):
        """
        Initialize segmentation metrics calculator
        
        Args:
            num_classes (int): Number of classes in the dataset
            include_background (bool): Whether to include background class (class 0) in metrics
        """
        self.num_classes = num_classes
        self.include_background = include_background
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_pixels = 0
        self.total_pixels_per_class = np.zeros(self.num_classes)
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        
    def update(self, pred_mask, gt_mask):
        """
        Update metrics with new prediction and ground truth masks
        
        Args:
            pred_mask (np.ndarray): Prediction mask with class indices
            gt_mask (np.ndarray): Ground truth mask with class indices
        """
        # Flatten masks
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        # Update confusion matrix
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.sum((gt_flat == i) & (pred_flat == j))
                
        # Update total pixels
        self.total_pixels += gt_flat.size
        
        # Update total pixels per class
        for i in range(self.num_classes):
            self.total_pixels_per_class[i] += np.sum(gt_flat == i)
            
        # Update intersection and union for Dice and IoU
        for i in range(self.num_classes):
            self.intersection[i] += np.sum((pred_flat == i) & (gt_flat == i))
            self.union[i] += np.sum((pred_flat == i) | (gt_flat == i))
        
    def compute_metrics(self):
        """
        Compute all metrics
        
        Returns:
            dict: Dictionary with all metrics
        """
        # Determine which classes to include
        classes_range = range(self.num_classes)
        if not self.include_background:
            classes_range = range(1, self.num_classes)
            
        # Calculate per-class metrics
        dice_per_class = np.zeros(self.num_classes)
        precision_per_class = np.zeros(self.num_classes)
        recall_per_class = np.zeros(self.num_classes)
        accuracy_per_class = np.zeros(self.num_classes)
        f1_per_class = np.zeros(self.num_classes)
        
        # Compute per-class metrics
        for i in range(self.num_classes):
            # Dice coefficient (2*TP / (2*TP + FP + FN))
            if self.union[i] > 0:
                dice_per_class[i] = 2 * self.intersection[i] / self.union[i]
                
            # Precision (TP / (TP + FP))
            true_positive = self.confusion_matrix[i, i]
            false_positive = np.sum(self.confusion_matrix[:, i]) - true_positive
            if (true_positive + false_positive) > 0:
                precision_per_class[i] = true_positive / (true_positive + false_positive)
                
            # Recall (TP / (TP + FN))
            false_negative = np.sum(self.confusion_matrix[i, :]) - true_positive
            if (true_positive + false_negative) > 0:
                recall_per_class[i] = true_positive / (true_positive + false_negative)
                
            # F1 Score (2 * Precision * Recall / (Precision + Recall))
            if (precision_per_class[i] + recall_per_class[i]) > 0:
                f1_per_class[i] = 2 * precision_per_class[i] * recall_per_class[i] / (precision_per_class[i] + recall_per_class[i])
                
            # Accuracy ((TP + TN) / (TP + TN + FP + FN))
            true_negative = np.sum(self.confusion_matrix) - np.sum(self.confusion_matrix[i, :]) - np.sum(self.confusion_matrix[:, i]) + true_positive
            denominator = true_positive + true_negative + false_positive + false_negative
            if denominator > 0:
                accuracy_per_class[i] = (true_positive + true_negative) / denominator
                
        # Calculate averaged metrics
        metrics = {}
        
        # With background
        metrics['dice_w_bg'] = np.mean(dice_per_class)
        metrics['precision_w_bg'] = np.mean(precision_per_class)
        metrics['recall_w_bg'] = np.mean(recall_per_class)
        metrics['f1_w_bg'] = np.mean(f1_per_class)
        metrics['accuracy_w_bg'] = np.mean(accuracy_per_class)
        
        # Without background (if applicable)
        if self.num_classes > 1:
            metrics['dice'] = np.mean(dice_per_class[1:]) if not self.include_background else metrics['dice_w_bg']
            metrics['precision'] = np.mean(precision_per_class[1:]) if not self.include_background else metrics['precision_w_bg']
            metrics['recall'] = np.mean(recall_per_class[1:]) if not self.include_background else metrics['recall_w_bg']
            metrics['f1'] = np.mean(f1_per_class[1:]) if not self.include_background else metrics['f1_w_bg']
            metrics['accuracy'] = np.mean(accuracy_per_class[1:]) if not self.include_background else metrics['accuracy_w_bg']
        else:
            # If there's only one class (binary problem)
            metrics['dice'] = metrics['dice_w_bg']
            metrics['precision'] = metrics['precision_w_bg']
            metrics['recall'] = metrics['recall_w_bg']
            metrics['f1'] = metrics['f1_w_bg']
            metrics['accuracy'] = metrics['accuracy_w_bg']
            
        # Per-class metrics
        metrics['dice_per_class'] = dice_per_class
        metrics['precision_per_class'] = precision_per_class
        metrics['recall_per_class'] = recall_per_class
        metrics['f1_per_class'] = f1_per_class
        metrics['accuracy_per_class'] = accuracy_per_class
        
        return metrics
        
def convert_yolo_masks_to_semantic_mask(results, img_shape, num_classes):
    """
    Convert YOLO instance segmentation to semantic segmentation mask
    
    Args:
        results: YOLO prediction results
        img_shape: Tuple of (height, width)
        num_classes: Number of classes
        
    Returns:
        np.ndarray: Semantic segmentation mask with class indices
    """
    height, width = img_shape
    semantic_mask = np.zeros((height, width), dtype=np.uint8)
    
    # If no masks detected, return empty mask
    if results.masks is None or len(results.masks) == 0:
        return semantic_mask
    
    # Process each instance
    for i, (mask, cls) in enumerate(zip(results.masks.data, results.boxes.cls)):
        # Convert tensor mask to numpy
        mask_np = mask.cpu().numpy()[0]
        
        # Resize mask if needed
        if mask_np.shape != (height, width):
            mask_np = cv2.resize(mask_np, (width, height))
        
        # Get class index
        class_idx = int(cls.item()) + 1  # Add 1 because 0 is background in semantic segmentation
        
        # Update semantic mask (higher class indices override lower ones)
        # This handles overlapping instances of different classes
        semantic_mask = np.where(mask_np > 0.5, class_idx, semantic_mask)
    
    return semantic_mask

def parse_yolo_labels_to_semantic_mask(label_path, img_shape, num_classes):
    """
    Parse YOLO format labels to create semantic segmentation mask
    
    Args:
        label_path: Path to YOLO label file
        img_shape: Tuple of (height, width)
        num_classes: Number of classes
        
    Returns:
        np.ndarray: Semantic segmentation mask with class indices
    """
    height, width = img_shape
    semantic_mask = np.zeros((height, width), dtype=np.uint8)
    
    if not os.path.exists(label_path):
        return semantic_mask
    
    # Read YOLO format labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:  # Skip if not enough points for polygon
                continue
                
            # Get class index
            class_idx = int(parts[0]) + 1  # Add 1 because 0 is background in semantic segmentation
            
            # Get segmentation points
            seg_points = list(map(float, parts[1:]))
            
            # Convert normalized coordinates to absolute coordinates
            points = []
            for i in range(0, len(seg_points), 2):
                x = int(seg_points[i] * width)
                y = int(seg_points[i+1] * height)
                points.append([x, y])
            
            # Convert points to mask
            if len(points) > 2:  # Need at least 3 points for a polygon
                instance_mask = np.zeros((height, width), dtype=np.uint8)
                points_array = np.array([points], dtype=np.int32)
                cv2.fillPoly(instance_mask, points_array, 1)
                
                # Update semantic mask (higher class indices override lower ones)
                semantic_mask = np.where(instance_mask > 0, class_idx, semantic_mask)
    
    return semantic_mask

def evaluate_model_comprehensive(model_path, val_data_path, dataset_yaml, output_dir=None, conf_threshold=0.25, 
                                iou_threshold=0.7, max_samples=None):
    """
    Evaluate a trained YOLO segmentation model using comprehensive metrics
    
    Args:
        model_path (str): Path to trained YOLO model
        val_data_path (str): Path to validation data directory
        dataset_yaml (str): Path to dataset.yaml
        output_dir (str): Path to save visualizations (optional)
        conf_threshold (float): Confidence threshold for predictions
        iou_threshold (float): IoU threshold for predictions
        max_samples (int): Maximum number of samples to evaluate (optional)
    
    Returns:
        dict: Dictionary with all metrics
    """
    # Load model
    model = YOLO(model_path)
    
    # Load dataset info
    import yaml
    with open(dataset_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    # Get class names
    class_names = dataset_info['names']
    num_classes = len(class_names) + 1  # +1 for background class
    
    # Initialize metrics (with and without background)
    metrics_with_bg = SegmentationMetrics(num_classes, include_background=True)
    metrics_without_bg = SegmentationMetrics(num_classes, include_background=False)
    
    # Get validation image paths
    val_images_dir = Path(val_data_path)
    val_images = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
    
    # Limit samples if specified
    if max_samples and max_samples < len(val_images):
        val_images = val_images[:max_samples]
    
    print(f"Evaluating on {len(val_images)} validation images")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each validation image
    losses = []
    for img_path in tqdm(val_images):
        # Get corresponding label path (YOLO format)
        label_path = Path(str(img_path).replace('images', 'labels').replace(img_path.suffix, '.txt'))
        
        # Load image to get shape
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]
        
        # Run inference
        results = model(img_path, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        
        # Convert YOLO predictions to semantic mask
        pred_mask = convert_yolo_masks_to_semantic_mask(results, (img_height, img_width), num_classes)
        
        # Convert YOLO ground truth to semantic mask
        gt_mask = parse_yolo_labels_to_semantic_mask(label_path, (img_height, img_width), num_classes)
        
        # Update metrics
        metrics_with_bg.update(pred_mask, gt_mask)
        metrics_without_bg.update(pred_mask, gt_mask)
        
        # Save visualization if requested
        if output_dir:
            # Create visualization
            vis_img = img.copy()
            
            # Overlay ground truth mask (green with transparency)
            gt_overlay = vis_img.copy()
            for cls in range(1, num_classes):  # Skip background
                gt_overlay[gt_mask == cls] = [0, 255, 0]  # Green for ground truth
            vis_img = cv2.addWeighted(vis_img, 0.7, gt_overlay, 0.3, 0)
            
            # Overlay prediction mask (red with transparency)
            pred_overlay = vis_img.copy()
            for cls in range(1, num_classes):  # Skip background
                pred_overlay[pred_mask == cls] = [0, 0, 255]  # Red for prediction
            vis_img = cv2.addWeighted(vis_img, 0.7, pred_overlay, 0.3, 0)
            
            # Save visualization
            vis_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(vis_path, vis_img)
    
    # Compute final metrics
    metrics_bg = metrics_with_bg.compute_metrics()
    metrics_no_bg = metrics_without_bg.compute_metrics()
    
    # Combine all metrics
    final_metrics = {
        'train_loss': 0,  # Not available in evaluation mode
        'val_loss': 0,    # Not available in evaluation mode
        'val_dice': metrics_no_bg['dice'],
        'val_f1': metrics_no_bg['f1'],
        'val_accuracy': metrics_no_bg['accuracy'],
        'val_precision': metrics_no_bg['precision'],
        'val_recall': metrics_no_bg['recall'],
        'val_dice_w_bg': metrics_bg['dice_w_bg'],
        'val_f1_w_bg': metrics_bg['f1_w_bg'],
        'val_accuracy_w_bg': metrics_bg['accuracy_w_bg'],
        'val_precision_w_bg': metrics_bg['precision_w_bg'],
        'val_recall_w_bg': metrics_bg['recall_w_bg'],
        # Per-class metrics
        'val_dice_per_class': metrics_no_bg['dice_per_class'],
        'val_f1_per_class': metrics_no_bg['f1_per_class'],
        'val_accuracy_per_class': metrics_no_bg['accuracy_per_class'],
        'val_precision_per_class': metrics_no_bg['precision_per_class'],
        'val_recall_per_class': metrics_no_bg['recall_per_class']
    }
    
    return final_metrics

class ComprehensiveMetricsCallback(Callback):
    """Custom callback to track comprehensive metrics during training"""
    def __init__(self, num_classes, output_dir=None):
        super().__init__()
        self.num_classes = num_classes
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.reset()
        
    def reset(self):
        self.metrics_with_bg = SegmentationMetrics(self.num_classes, include_background=True)
        self.metrics_without_bg = SegmentationMetrics(self.num_classes, include_background=False)
        
    def on_train_epoch_end(self, trainer):
        """Called at the end of training epoch"""
        # Compute metrics from training data
        train_metrics = self._compute_metrics()
        
        # Add metrics to trainer
        for name, value in train_metrics.items():
            if isinstance(value, (int, float)):
                trainer.metrics[name] = value
        
        # Log to console
        print(f"\nTrain metrics:")
        for name, value in train_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{name}: {value:.4f}")
        
        # Reset for next epoch
        self.reset()
        
    def on_val_end(self, trainer):
        """Called at the end of validation"""
        # Compute metrics from validation data
        val_metrics = self._compute_metrics(prefix='val_')
        
        # Add metrics to trainer
        for name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                trainer.metrics[name] = value
        
        # Log to console
        print(f"\nValidation metrics:")
        for name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{name}: {value:.4f}")
        
        # Log to tensorboard if available
        if hasattr(trainer, 'tb') and trainer.tb:
            for name, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    trainer.tb.add_scalar(f'Metrics/{name}', value, trainer.epoch)
        
        # Reset for next validation
        self.reset()
        
    def _compute_metrics(self, prefix='train_'):
        """Compute and return all metrics with specified prefix"""
        metrics_bg = self.metrics_with_bg.compute_metrics()
        metrics_no_bg = self.metrics_without_bg.compute_metrics()
        
        # Combine all metrics with prefix
        final_metrics = {
            f'{prefix}dice': metrics_no_bg['dice'],
            f'{prefix}f1': metrics_no_bg['f1'],
            f'{prefix}accuracy': metrics_no_bg['accuracy'],
            f'{prefix}precision': metrics_no_bg['precision'],
            f'{prefix}recall': metrics_no_bg['recall'],
            f'{prefix}dice_w_bg': metrics_bg['dice_w_bg'],
            f'{prefix}f1_w_bg': metrics_bg['f1_w_bg'],
            f'{prefix}accuracy_w_bg': metrics_bg['accuracy_w_bg'],
            f'{prefix}precision_w_bg': metrics_bg['precision_w_bg'],
            f'{prefix}recall_w_bg': metrics_bg['recall_w_bg'],
            # Per-class metrics (these won't be logged to tensorboard)
            f'{prefix}dice_per_class': metrics_no_bg['dice_per_class'],
            f'{prefix}f1_per_class': metrics_no_bg['f1_per_class'],
            f'{prefix}accuracy_per_class': metrics_no_bg['accuracy_per_class'],
            f'{prefix}precision_per_class': metrics_no_bg['precision_per_class'],
            f'{prefix}recall_per_class': metrics_no_bg['recall_per_class']
        }
        
        return final_metrics

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/trained/model.pt"
    val_data_path = "path/to/output/fashionpedia_yolo/images/val"
    dataset_yaml = "path/to/output/fashionpedia_yolo/dataset.yaml"
    output_dir = "path/to/visualizations"
    
    # Run evaluation
    metrics = evaluate_model_comprehensive(
        model_path=model_path,
        val_data_path=val_data_path,
        dataset_yaml=dataset_yaml,
        output_dir=output_dir,
        conf_threshold=0.25,
        iou_threshold=0.7
    )
    
    # Print results
    print("\nFinal Metrics:")
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() if isinstance(v, (int, float))})
    metrics_df.to_csv("yolo_segmentation_metrics.csv", index=False)
    
    # Generate visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot metrics comparison
    metrics_to_plot = [
        ('dice', 'dice_w_bg'),
        ('f1', 'f1_w_bg'),
        ('accuracy', 'accuracy_w_bg'),
        ('precision', 'precision_w_bg'),
        ('recall', 'recall_w_bg')
    ]
    
    for i, (metric, metric_w_bg) in enumerate(metrics_to_plot):
        metric_name = metric.split('_')[-1]
        plt.subplot(2, 3, i+1)
        plt.bar(['Without BG', 'With BG'], [metrics[f'val_{metric}'], metrics[f'val_{metric_w_bg}']])
        plt.title(f'{metric_name.capitalize()} Comparison')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.close()
    
    print(f"Results saved to yolo_segmentation_metrics.csv and metrics_comparison.png")