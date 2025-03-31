import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def compare_models(models_results, output_dir='./comparison_results'):
    """
    Compare metrics between different segmentation models
    
    Args:
        models_results (dict): Dictionary with model names as keys and paths to metrics CSV files as values
        output_dir (str): Directory to save comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics for each model
    models_data = {}
    for model_name, csv_path in models_results.items():
        models_data[model_name] = pd.read_csv(csv_path)
    
    # List of metrics to compare
    val_metrics = [
        'val_dice', 'val_f1', 'val_accuracy', 'val_precision', 'val_recall',
        'val_dice_w_bg', 'val_f1_w_bg', 'val_accuracy_w_bg', 'val_precision_w_bg', 'val_recall_w_bg'
    ]
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, data in models_data.items():
        model_metrics = {metric: data[metric].values[0] for metric in val_metrics if metric in data.columns}
        model_metrics['model'] = model_name
        comparison_data.append(model_metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(output_dir, 'models_comparison.csv'), index=False)
    
    # Create bar plots for each metric
    plt.figure(figsize=(20, 15))
    
    for i, metric in enumerate(val_metrics):
        if all(metric in data.columns for model_name, data in models_data.items()):
            plt.subplot(3, 4, i+1)
            
            # Extract values for this metric
            values = [data[metric].values[0] for model_name, data in models_data.items()]
            model_names = list(models_results.keys())
            
            # Create bar chart
            bars = plt.bar(model_names, values)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', rotation=0)
            
            plt.title(f'{metric}')
            plt.ylim(0, 1.1)  # Metrics are typically between 0 and 1
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()
    
    # Create radar chart for model comparison
    metrics_display = {
        'val_dice': 'Dice',
        'val_f1': 'F1',
        'val_accuracy': 'Accuracy',
        'val_precision': 'Precision',
        'val_recall': 'Recall',
        'val_dice_w_bg': 'Dice w/ BG',
        'val_f1_w_bg': 'F1 w/ BG',
        'val_accuracy_w_bg': 'Acc w/ BG',
        'val_precision_w_bg': 'Prec w/ BG',
        'val_recall_w_bg': 'Recall w/ BG'
    }
    
    # Create two radar charts: with and without background
    for bg_suffix, title_suffix in [('', 'without Background'), ('_w_bg', 'with Background')]:
        # Filter metrics
        radar_metrics = [m for m in val_metrics if m.endswith(bg_suffix)]
        if not radar_metrics:
            continue
            
        # Prepare data
        categories = [metrics_display[m] for m in radar_metrics]
        N = len(categories)
        
        # Create angles for the radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add each model to the chart
        for model_name, data in models_data.items():
            values = [data[m].values[0] for m in radar_metrics]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Add radial grid lines
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1)
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f'Model Comparison {title_suffix}', size=15, y=1.1)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'radar_comparison{bg_suffix}.png'))
        plt.close()
    
    # Create a heatmap to visualize all metrics
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = comparison_df.set_index('model')
    
    # Rename columns for better display
    heatmap_data.columns = [metrics_display.get(col, col) for col in heatmap_data.columns]
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5, vmin=0, vmax=1)
    plt.title('Metrics Comparison Across Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
    plt.close()
    
    print(f"Comparison results saved to {output_dir}")
    
    return comparison_df

def extract_per_class_metrics(yolo_results_path, other_models_results, class_names, output_dir='./class_comparison'):
    """
    Extract and compare per-class metrics between YOLO and other models
    
    Args:
        yolo_results_path (str): Path to YOLO metrics with per-class information
        other_models_results (dict): Dictionary with model names as keys and paths to metrics with per-class info
        class_names (list): List of class names
        output_dir (str): Directory to save comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO metrics
    yolo_data = pd.read_csv(yolo_results_path)
    
    # Load other models' metrics
    models_data = {'YOLO': yolo_data}
    for model_name, csv_path in other_models_results.items():
        models_data[model_name] = pd.read_csv(csv_path)
    
    # Per-class metrics to extract (if available)
    class_metrics = ['dice', 'f1', 'precision', 'recall']
    
    # For each metric, create a comparison chart
    for metric in class_metrics:
        # Check if per-class data is available
        if f'val_{metric}_per_class' not in yolo_data.columns:
            continue
            
        # Extract per-class values for each model
        class_comparison = {}
        for model_name, data in models_data.items():
            if f'val_{metric}_per_class' in data.columns:
                # Parse string representation of array to actual array
                values_str = data[f'val_{metric}_per_class'].values[0]
                if isinstance(values_str, str):
                    # Handle various string formats
                    values_str = values_str.replace('[', '').replace(']', '').replace('\n', '')
                    values = [float(x) for x in values_str.split() if x.strip()]
                else:
                    values = []
                    
                # Skip background class (index 0) if needed
                class_comparison[model_name] = values[1:] if len(values) > len(class_names) else values
        
        # Create comparison dataframe
        if class_comparison:
            df = pd.DataFrame(class_comparison, index=class_names)
            
            # Create bar chart
            plt.figure(figsize=(14, 8))
            df.plot(kind='bar', rot=45)
            plt.title(f'Per-class {metric.capitalize()} Comparison')
            plt.xlabel('Class')
            plt.ylabel(f'{metric.capitalize()} Score')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'per_class_{metric}_comparison.png'))
            plt.close()
            
            # Save to CSV
            df.to_csv(os.path.join(output_dir, f'per_class_{metric}_comparison.csv'))
    
    print(f"Per-class comparison results saved to {output_dir}")
