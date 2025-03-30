from ultralytics import YOLO
import yaml
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import pandas as pd
from tqdm import tqdm

# Import our metrics module
from yolo_comprehensive_metrics import ComprehensiveMetricsCallback, evaluate_model_comprehensive
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import (
    DATA_DIR,
    ANNOTATIONS_DIR
)


def train_yolo_with_metrics(
    data_yaml_path,
    model_name='yolov8n-seg.pt',
    epochs=100,
    image_size=640,
    batch_size=16,
    device=0,
    project_name='fashionpedia_segmentation',
    output_dir='./results',
    metrics_eval_fraction=0.001,  # Fraction of validation set to use for per-epoch metrics
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
        metrics_eval_fraction (float): Fraction of validation data to use for per-epoch metrics
        debug (bool): Whether to enable debug mode
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    
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
    
    num_classes = len(dataset_info['names']) + 1  # Add 1 for background class
    
    # Get validation data path for metrics calculation
    val_data_path = os.path.join(os.path.dirname(data_yaml_path), dataset_info['val'])
    
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
    
    # Create run directory within the project
    run_name = "with_comprehensive_metrics"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create metrics callback
    metrics_callback = ComprehensiveMetricsCallback(
        num_classes=num_classes,
        output_dir=os.path.join(run_dir, 'metrics_log'),
        dataset_yaml=data_yaml_path,
        val_data_path=val_data_path,
        eval_fraction=metrics_eval_fraction
    )
    
    # Register the callbacks
    #model.add_callback("on_train_start", metrics_callback.on_train_start)
    #model.add_callback("on_train_epoch_end", metrics_callback.on_train_epoch_end)
    #model.add_callback("on_val_end", metrics_callback.on_val_end)
    
    # Train model with callbacks
    if debug:
        print("\n=== Starting training ===")
        print(f"Epochs: {epochs}")
        print(f"Image size: {image_size}")
        print(f"Batch size: {batch_size}")
        print(f"Metrics evaluation: {metrics_eval_fraction:.1%} of validation set")
    
    # Check for CUDA/GPU
    if device != 'cpu' and not torch.cuda.is_available():
        print("Warning: CUDA is not available, falling back to CPU")
        device = 'cpu'

    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            project=project_name,
            name=run_name,
            save=True,
            patience=50,  # Early stopping patience
            verbose=True,
            task='segment',
            fraction=0.001  # Use small fraction of data if in debug mode
        )
        
        if debug:
            print("\n=== Training complete ===")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e
    # Get best model path
    best_model_path = model.trainer.best
    
    # Run comprehensive evaluation on best model with full validation set
    if debug:
        print(f"\n=== Evaluating best model: {best_model_path} ===")
    
    final_metrics = evaluate_model_comprehensive(
        model_path=best_model_path,
        val_data_path=val_data_path,
        dataset_yaml=data_yaml_path,
        output_dir=os.path.join(run_dir, 'final_visualizations'),
        conf_threshold=0.25,  # Standard threshold
        iou_threshold=0.7,     # Standard threshold
        max_samples=10
    )
    
    # Save final metrics to CSV
    metrics_df = pd.DataFrame({k: [v] for k, v in final_metrics.items() if isinstance(v, (int, float))})
    metrics_df.to_csv(os.path.join(run_dir, "final_metrics.csv"), index=False)
    
    # Print final results
    print("\nFinal Metrics:")
    for name, value in final_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")
    
    # Plot metrics history if available
    if os.path.exists(os.path.join(run_dir, 'metrics_log', 'training_metrics.csv')):
        try:
            # Load metrics history
            history_df = pd.read_csv(os.path.join(run_dir, 'metrics_log', 'training_metrics.csv'))
            
            # Plot key metrics over epochs
            plt.figure(figsize=(12, 8))
            
            # Plot Dice score
            if 'val_dice' in history_df.columns:
                plt.subplot(2, 2, 1)
                plt.plot(history_df['epoch'], history_df['val_dice'], 'b-', label='Dice Score')
                plt.title('Dice Score')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.grid(True)
            
            # Plot IoU
            if 'val_iou' in history_df.columns:
                plt.subplot(2, 2, 2)
                plt.plot(history_df['epoch'], history_df['val_iou'], 'g-', label='IoU')
                plt.title('IoU (Jaccard Index)')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.grid(True)
            
            # Plot Precision and Recall
            if 'val_precision' in history_df.columns and 'val_recall' in history_df.columns:
                plt.subplot(2, 2, 3)
                plt.plot(history_df['epoch'], history_df['val_precision'], 'r-', label='Precision')
                plt.plot(history_df['epoch'], history_df['val_recall'], 'y-', label='Recall')
                plt.title('Precision and Recall')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.legend()
                plt.grid(True)
            
            # Plot F1 Score
            if 'val_f1' in history_df.columns:
                plt.subplot(2, 2, 4)
                plt.plot(history_df['epoch'], history_df['val_f1'], 'm-', label='F1 Score')
                plt.title('F1 Score')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'metrics_history.png'))
            print(f"Metrics history plot saved to {os.path.join(run_dir, 'metrics_history.png')}")
            
        except Exception as e:
            print(f"Error plotting metrics history: {str(e)}")
    
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
        epochs=1,#100,                   # Train for more epochs
        image_size=640,
        batch_size=16,
        device='cpu',                 # Use CPU for compatibility 
        project_name='fashionpedia_segmentation',
        output_dir= DATA_DIR / "02_metrics" / "yolo_comprehensive_metrics_results",
        debug=True
    )
    
    print(f"Training complete. Best model: {best_model}")