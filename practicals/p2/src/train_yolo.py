from ultralytics import YOLO
import yaml
import os
import sys
from pathlib import Path
# Import our metrics module
# Assuming you saved the previous code as comprehensive_metrics.py

from yolo_comprehensive_metrics import ComprehensiveMetricsCallback, evaluate_model_comprehensive
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import (
    DATA_DIR,
    ANNOTATIONS_DIR
)
def train_yolo_with_metrics(
    data_yaml_path,
    model_name='yolov11n-seg.pt',
    epochs=100,
    image_size=640,
    batch_size=16,
    device=0,
    project_name='fashionpedia_segmentation',
    output_dir='./results'
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
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset info to get number of classes
    with open(data_yaml_path, 'r') as f:
        dataset_info = yaml.safe_load(f)
    num_classes = len(dataset_info['names'])  # Number of classes in the dataset
    
    # Initialize model
    model = YOLO(model_name)
    
    # Create metrics callback
    metrics_callback = ComprehensiveMetricsCallback(
        num_classes=num_classes,
        output_dir=os.path.join(output_dir, 'visualizations')
    )
    
    # Register the callbacks
    model.add_callback("on_train_epoch_end", metrics_callback.on_train_epoch_end)
    model.add_callback("on_val_end", metrics_callback.on_val_end)
    
    # Train model with callbacks
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=project_name,
        name='with_comprehensive_metrics',
        save=True,
        patience=50,  # Early stopping patience
        verbose=True,
        task='segment'  # Explicitly specify segmentation task
    )
    
    # Get best model path
    best_model_path = model.trainer.best
    
    # Run comprehensive evaluation on best model
    print(f"\nEvaluating best model: {best_model_path}")
    val_data_path = os.path.join(os.path.dirname(data_yaml_path), dataset_info['val'])
    
    final_metrics = evaluate_model_comprehensive(
        model_path=best_model_path,
        val_data_path=val_data_path,
        dataset_yaml=data_yaml_path,
        output_dir=os.path.join(output_dir, 'final_visualizations'),
        conf_threshold=0.25,
        iou_threshold=0.7
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
    
    # Train model with metrics
    best_model, metrics = train_yolo_with_metrics(
        data_yaml_path=data_yaml_path,
        model_name='yolov11n-seg.pt',  # Make sure this is the correct model name
        epochs=100,
        image_size=640,
        batch_size=16,
        device=0,
        project_name='fashionpedia_segmentation',
        output_dir= DATA_DIR / "02_metrics" / "yolo_comprehensive_metrics_results"
    )
    
    print(f"Training complete. Best model: {best_model}")