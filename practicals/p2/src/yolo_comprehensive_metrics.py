import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2

# Define our own Callback base class since it might not be directly available
class Callback:
    """Base callback class to create custom callbacks for YOLO training"""
    def __init__(self):
        # Initialize any necessary attributes
        self.training_metrics = {}
        self.validation_metrics = {}
        self.epoch = 0
    
    def on_train_start(self, trainer):
        """Called when training starts."""
        if hasattr(trainer, 'epoch'):
            self.epoch = trainer.epoch
    
    def on_train_epoch_start(self, trainer):
        """Called at the start of each training epoch."""
        if hasattr(trainer, 'epoch'):
            self.epoch = trainer.epoch
    
    def on_train_batch_start(self, trainer):
        """Called at the start of each training batch."""
        if hasattr(trainer, 'batch_idx'):
            self.batch_idx = trainer.batch_idx
    
    def on_train_batch_end(self, trainer):
        """Called at the end of each training batch."""
        if hasattr(trainer, 'loss') and isinstance(trainer.loss, dict):
            for key, value in trainer.loss.items():
                if key not in self.training_metrics:
                    self.training_metrics[key] = []
                if isinstance(value, (int, float)):
                    self.training_metrics[key].append(value)
    
    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        pass
    
    def on_train_end(self, trainer):
        """Called when training ends."""
        pass
    
    def on_val_start(self, trainer):
        """Called when validation starts."""
        self.validation_metrics = {}
    
    def on_val_batch_start(self, trainer):
        """Called at the start of each validation batch."""
        pass
    
    def on_val_batch_end(self, trainer):
        """Called at the end of each validation batch."""
        if hasattr(trainer, 'loss') and isinstance(trainer.loss, dict):
            for key, value in trainer.loss.items():
                val_key = f"val_{key}"
                if val_key not in self.validation_metrics:
                    self.validation_metrics[val_key] = []
                if isinstance(value, (int, float)):
                    self.validation_metrics[val_key].append(value)
    
    def on_val_end(self, trainer):
        """Called at the end of validation."""
        pass
    
    def on_fit_epoch_end(self, trainer):
        """Called at the end of each fit epoch (after validation)."""
        pass
    
    def on_predict_start(self, predictor):
        """Called when prediction starts."""
        pass
    
    def on_predict_batch_start(self, predictor):
        """Called at the start of each prediction batch."""
        pass
    
    def on_predict_batch_end(self, predictor):
        """Called at the end of each prediction batch."""
        pass
    
    def on_predict_end(self, predictor):
        """Called when prediction ends."""
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
        self.update_counter = 0
        
    def update(self, pred_mask, gt_mask):
        """
        Update metrics with new prediction and ground truth masks
        
        Args:
            pred_mask (np.ndarray): Prediction mask with class indices
            gt_mask (np.ndarray): Ground truth mask with class indices
        """
        self.update_counter += 1
        
        if self.update_counter % 10 == 0:
            print(f"Metrics update #{self.update_counter}")
            print(f"  Pred mask shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}")
            print(f"  GT mask shape: {gt_mask.shape}, unique values: {np.unique(gt_mask)}")

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
    
    # Check if results have masks
    if not hasattr(results, 'masks') or results.masks is None:
        return semantic_mask
    
    if len(results.masks) == 0:
        return semantic_mask
    
    # Process each instance
    for i, (mask, cls) in enumerate(zip(results.masks.data, results.boxes.cls)):
        # Convert tensor mask to numpy
        mask_np = mask.cpu().numpy()
        
        # Resize mask if needed
        if mask_np.shape != (height, width):
            mask_np = cv2.resize(mask_np, (width, height))
        
        # Get class index
        class_idx = int(cls.item()) + 1  # Add 1 because 0 is background in semantic segmentation
        
        # Update semantic mask (higher class indices override lower ones)
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
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:  # Skip if not enough points for polygon
            continue
            
        # Get class index
        class_idx = int(float(parts[0])) + 1  # Add 1 because 0 is background in semantic segmentation
        
        # Get segmentation points
        seg_points = list(map(float, parts[1:]))
        
        # Check number of points
        if len(seg_points) % 2 != 0:
            continue
            
        # Convert normalized coordinates to absolute coordinates
        points = []
        for i in range(0, len(seg_points), 2):
            x = int(seg_points[i] * width)
            y = int(seg_points[i+1] * height)
            points.append([x, y])
        
        # Check for valid polygon
        if len(points) < 3:  # Need at least 3 points for a polygon
            continue
            
        # Convert points to mask
        try:
            instance_mask = np.zeros((height, width), dtype=np.uint8)
            points_array = np.array([points], dtype=np.int32)
            cv2.fillPoly(instance_mask, points_array, 1)
            
            # Update semantic mask (higher class indices override lower ones)
            semantic_mask = np.where(instance_mask > 0, class_idx, semantic_mask)
        except Exception:
            pass
    
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
    model.to('cpu')  # Force CPU to avoid device issues
    
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
    def __init__(self, num_classes, output_dir=None, dataset_yaml=None, val_data_path=None, eval_fraction=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.dataset_yaml = dataset_yaml
        self.val_data_path = val_data_path
        self.eval_fraction = eval_fraction  # Fraction of validation set to use
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        self.reset()
        self.all_metrics = {}
        self.model = None
        self.val_images = None
        
    def reset(self):
        """Reset metric calculators"""
        self.metrics_with_bg = SegmentationMetrics(self.num_classes, include_background=True)
        self.metrics_without_bg = SegmentationMetrics(self.num_classes, include_background=False)
    
    def on_train_start(self, trainer):
        """Called when training starts - store model reference and prepare validation images"""
        super().on_train_start(trainer)
        self.model = trainer.model
        
        # Prepare validation image list once at the start
        if self.val_data_path and self.eval_fraction > 0:
            self._prepare_validation_set()
    
    def _prepare_validation_set(self):
        """Prepare the validation set - done once at the beginning of training"""
        # Get validation image paths
        val_images_dir = Path(self.val_data_path)
        all_val_images = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
        
        if len(all_val_images) == 0:
            print(f"Warning: No validation images found in {self.val_data_path}")
            self.val_images = []
            return
            
        # Calculate number of images to use based on fraction
        num_eval_images = max(1, int(len(all_val_images) * self.eval_fraction))
        
        # Use consistent random seed for reproducibility
        np.random.seed(42)
        
        # Sample images
        indices = np.random.choice(len(all_val_images), num_eval_images, replace=False)
        self.val_images = [all_val_images[i] for i in indices]
        
        print(f"Prepared evaluation set with {len(self.val_images)} images " +
              f"({self.eval_fraction:.1%} of {len(all_val_images)} total)")
    
    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch - just display epoch number"""
        print(f"\n=== End of training epoch {trainer.epoch} ===\n")
    
    def on_val_end(self, validator):
        """Called at the end of validation - evaluate model on validation subset"""
        # Get the current epoch
        epoch = getattr(validator.trainer, 'epoch', 0) if hasattr(validator, 'trainer') else 0
        
        # Skip if we don't have the necessary data
        if self.model is None or self.val_images is None or len(self.val_images) == 0:
            print("Warning: Cannot calculate metrics - missing model or validation data")
            return
            
        try:
            print(f"\nCalculating metrics for epoch {epoch} on {len(self.val_images)} validation images...")
            
            # Reset metrics for this epoch
            self.reset()
            
            # Process each validation image
            for img_path in tqdm(self.val_images, desc=f"Evaluating epoch {epoch}"):
                # Get corresponding label path (YOLO format)
                label_path = Path(str(img_path).replace('images', 'labels').replace(img_path.suffix, '.txt'))
                
                if not label_path.exists():
                    continue
                    
                # Load image to get shape
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img_height, img_width = img.shape[:2]
                
                # Run inference with current model weights
                with torch.no_grad():
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                    results = self.model(img_tensor)
                

                # Convert raw model outputs to Results format
                from ultralytics.engine.results import Results
                results = Results(
                    orig_img=img,
                    path=str(img_path),
                    names=self.model.names if hasattr(self.model, 'names') else None,
                    boxes=results[0] if isinstance(results, tuple) and len(results) > 0 else None,
                    masks=results[1] if isinstance(results, tuple) and len(results) > 1 else None
                )
                                                
                # Convert predictions to semantic mask
                pred_mask = convert_yolo_masks_to_semantic_mask(results, (img_height, img_width), self.num_classes)
                
                # Convert ground truth to semantic mask
                gt_mask = parse_yolo_labels_to_semantic_mask(label_path, (img_height, img_width), self.num_classes)
                
                # Update metrics
                self.metrics_with_bg.update(pred_mask, gt_mask)
                self.metrics_without_bg.update(pred_mask, gt_mask)
            
            # Skip metrics calculation if no updates occurred
            if self.metrics_with_bg.update_counter == 0:
                print("Warning: No valid mask pairs were processed for metrics calculation")
                return
                
            # Compute metrics
            metrics = self._compute_metrics(prefix='val_')
            
            # Store metrics history
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.all_metrics[name] = value
            
            # Log to console
            print(f"\nEpoch {epoch} - Validation metrics:")
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{name}: {value:.4f}")
            
            # Save to CSV for tracking
            self._save_metrics_to_csv(epoch)
            
        except Exception as e:
            import traceback
            print(f"Error calculating metrics: {str(e)}")
            print(traceback.format_exc())
        
    def _compute_metrics(self, prefix='val_'):
        """Compute all metrics from the current state"""
        metrics_bg = self.metrics_with_bg.compute_metrics()
        metrics_no_bg = self.metrics_without_bg.compute_metrics()
        
        # Format metrics with prefix
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
            # Add IoU (Jaccard) metrics
            f'{prefix}iou': np.mean(metrics_no_bg['dice_per_class'] / (2 - metrics_no_bg['dice_per_class'])),
            f'{prefix}iou_w_bg': np.mean(metrics_bg['dice_per_class'] / (2 - metrics_bg['dice_per_class']))
        }
        
        # Store class-wise metrics
        self.class_metrics = {
            f'{prefix}dice_per_class': metrics_no_bg['dice_per_class'],
            f'{prefix}f1_per_class': metrics_no_bg['f1_per_class'],
            f'{prefix}accuracy_per_class': metrics_no_bg['accuracy_per_class'],
            f'{prefix}precision_per_class': metrics_no_bg['precision_per_class'],
            f'{prefix}recall_per_class': metrics_no_bg['recall_per_class']
        }
        
        return final_metrics
    
    def _save_metrics_to_csv(self, epoch):
        """Save current epoch metrics to CSV file"""
        if not self.output_dir:
            return
            
        # Create metrics dictionary
        metrics_dict = {'epoch': epoch}
        for name, value in self.all_metrics.items():
            if isinstance(value, (int, float)):
                metrics_dict[name] = value
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics_dict])
        
        # Save to CSV (append mode)
        csv_path = os.path.join(self.output_dir, 'training_metrics.csv')
        
        # Check if file exists to handle header correctly
        if os.path.exists(csv_path):
            metrics_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(csv_path, index=False)


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