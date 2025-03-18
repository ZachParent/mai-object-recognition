import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.classification import Accuracy, Dice
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import logging
from datetime import datetime
from tqdm import tqdm
# Import missing functions
from data_load_test import setup_fashionpedia, load_fashionpedia_categories, create_data_loaders

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SegmentationTrainer")

# Create directories for saving results
os.makedirs("training_results", exist_ok=True)
os.makedirs("training_results/plots", exist_ok=True)
os.makedirs("training_results/checkpoints", exist_ok=True)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")

def choose_model(model_name, num_classes):
    """
    Initialize a semantic segmentation model based on model name.
    
    Args:
        model_name (str): Name of the model to initialize
        num_classes (int): Number of classes for segmentation
        
    Returns:
        torch.nn.Module: Initialized model
    """
    if model_name == 'deeplab':
        model = models.segmentation.deeplabv3_resnet101(
            weights_backbone="ResNet101_Weights.DEFAULT", 
            num_classes=num_classes
        )
    elif model_name == 'segformer':
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )
    elif model_name == 'lraspp':
        model = models.segmentation.lraspp_mobilenet_v3_large(
            weights_backbone="MobileNet_V3_Large_Weights.DEFAULT", 
            num_classes=num_classes
        )
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    
    model.to(device)
    return model

def calculate_accuracy(predictions, ground_truth):
    """
    Calculate pixel-wise accuracy between predictions and ground truth.
    
    Args:
        predictions (torch.Tensor): Tensor of predicted class indices
        ground_truth (torch.Tensor): Tensor of ground truth class indices
        
    Returns:
        float: Accuracy as a percentage
    """
    correct_pixels = (predictions == ground_truth).sum().item()
    total_pixels = ground_truth.numel()
    return (correct_pixels / max(total_pixels, 1)) * 100

def calculate_mean_dice_score(predictions, ground_truth, num_classes=None):
    """
    Calculate mean Dice coefficient (F1 score) across all classes.
    
    Args:
        predictions (torch.Tensor): Tensor of predicted class indices
        ground_truth (torch.Tensor): Tensor of ground truth class indices
        num_classes (int, optional): Number of classes in the dataset
        
    Returns:
        float: Mean Dice score across all classes
    """
    # If num_classes not provided, determine from data
    if num_classes is None:
        num_classes = max(predictions.max().item(), ground_truth.max().item()) + 1
    
    # Calculate per-class Dice scores
    dice_scores = []
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_mask = (predictions == class_idx)
        gt_mask = (ground_truth == class_idx)
        
        # Calculate intersection and union
        intersection = (pred_mask & gt_mask).sum().item()
        pred_area = pred_mask.sum().item()
        gt_area = gt_mask.sum().item()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection) / max(pred_area + gt_area, 1e-8)
        dice_scores.append(dice)
    
    # Return mean of Dice scores
    return np.mean(dice_scores)

def calculate_per_class_dice(predictions, ground_truth, num_classes):
    """
    Calculate Dice coefficient (F1 score) for each class separately.
    
    Args:
        predictions (torch.Tensor): Tensor of predicted class indices
        ground_truth (torch.Tensor): Tensor of ground truth class indices
        num_classes (int): Number of classes in the dataset
        
    Returns:
        list: List of Dice scores for each class
    """
    dice_scores = []
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_mask = (predictions == class_idx)
        gt_mask = (ground_truth == class_idx)
        
        # Calculate intersection and union
        intersection = (pred_mask & gt_mask).sum().item()
        pred_area = pred_mask.sum().item()
        gt_area = gt_mask.sum().item()
        
        # Calculate Dice coefficient
        denominator = pred_area + gt_area
        if denominator > 0:
            dice = (2.0 * intersection) / denominator
        else:
            # If class not present in either prediction or ground truth
            dice = 1.0 if pred_area == 0 and gt_area == 0 else 0.0
            
        dice_scores.append(dice)
    
    return dice_scores

def train(model, train_loader, optimizer, criterion, device, epoch_num=None, num_classes=None):
    """
    Comprehensive training function for semantic segmentation models.
    
    Args:
        model (torch.nn.Module): The neural network model to train
        train_loader (DataLoader): PyTorch DataLoader containing training data
        optimizer (torch.optim.Optimizer): Optimization algorithm
        criterion (torch.nn.Module): Loss function
        device (torch.device): Computation device (CPU/GPU)
        epoch_num (int, optional): Current epoch number for logging
        num_classes (int, optional): Number of classes for detailed metrics
        
    Returns:
        tuple: Contains (average_loss, accuracy, mean_dice_score, class_dice_scores)
    """
    # Set model to training mode
    model.train()
    
    # Initialize tracking variables
    total_samples_processed = 0
    correctly_classified_pixels = 0
    cumulative_loss = 0.0
    batch_count = 0
    
    # Lists to store predictions and ground truth for metrics
    all_predictions = []
    all_ground_truth_labels = []
    
    # Track execution time
    epoch_start_time = time.time()
    
    # Create progress bar
    progress_bar = tqdm(
        train_loader, 
        desc=f"Training Epoch {epoch_num if epoch_num is not None else ''}",
        leave=True
    )
    
    # Iterate through each batch in the training dataset
    for batch_idx, (input_images, target_labels) in enumerate(progress_bar):
        # Record batch start time
        batch_start_time = time.time()
        
        # Transfer data to device
        input_images = input_images.to(device)
        target_labels = target_labels.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        model_outputs = model(input_images)
        
        # Handle different model output formats
        if isinstance(model_outputs, dict) and 'out' in model_outputs:
            segmentation_logits = model_outputs['out']
        else:
            segmentation_logits = model_outputs
            
        # Calculate loss
        current_batch_loss = criterion(segmentation_logits, target_labels)
        
        # Backward pass
        current_batch_loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Accumulate batch loss
        cumulative_loss += current_batch_loss.item()
        batch_count += 1
        
        # Compute predicted class indices
        _, predicted_class_indices = torch.max(segmentation_logits, dim=1)
        
        # Calculate pixel-level accuracy
        correctly_classified_pixels += (predicted_class_indices == target_labels).sum().item()
        total_samples_processed += target_labels.numel()
        
        # Store predictions and ground truth
        all_predictions.append(predicted_class_indices.detach().cpu())
        all_ground_truth_labels.append(target_labels.detach().cpu())
        
        # Calculate batch processing time
        batch_processing_time = time.time() - batch_start_time
        
        # Update progress bar
        current_accuracy = correctly_classified_pixels / max(total_samples_processed, 1) * 100
        progress_bar.set_postfix({
            'loss': f"{current_batch_loss.item():.4f}",
            'accuracy': f"{current_accuracy:.2f}%",
            'batch_time': f"{batch_processing_time:.3f}s"
        })
        
        # Detailed logging for every 10th batch
        if batch_idx % 10 == 0:
            logger.info(
                f"Training Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {current_batch_loss.item():.4f} | "
                f"Accuracy: {current_accuracy:.2f}% | "
                f"Processing Time: {batch_processing_time:.3f}s"
            )
    
    # Concatenate all batches into single tensors
    all_predictions_tensor = torch.cat(all_predictions)
    all_ground_truth_tensor = torch.cat(all_ground_truth_labels)
    
    # Calculate final metrics
    final_epoch_accuracy = calculate_accuracy(all_predictions_tensor, all_ground_truth_tensor)
    final_epoch_mean_dice = calculate_mean_dice_score(all_predictions_tensor, all_ground_truth_tensor, num_classes)
    
    # Calculate per-class dice scores if num_classes is provided
    per_class_dice_scores = None
    if num_classes is not None:
        per_class_dice_scores = calculate_per_class_dice(all_predictions_tensor, all_ground_truth_tensor, num_classes)
        
        # Log per-class performance
        logger.info("Per-class Dice Scores:")
        for class_idx, dice_score in enumerate(per_class_dice_scores):
            logger.info(f"  Class {class_idx}: {dice_score:.4f}")
    
    # Calculate average loss
    average_epoch_loss = cumulative_loss / max(batch_count, 1)
    
    # Calculate total epoch time
    total_epoch_time = time.time() - epoch_start_time
    
    # Log detailed epoch summary
    logger.info(
        f"\nTraining Epoch Summary {'(Epoch '+str(epoch_num)+')' if epoch_num is not None else ''} | "
        f"Loss: {average_epoch_loss:.4f} | "
        f"Accuracy: {final_epoch_accuracy:.2f}% | "
        f"Mean Dice: {final_epoch_mean_dice:.4f} | "
        f"Duration: {total_epoch_time:.2f}s | "
        f"Images/sec: {len(train_loader.dataset)/total_epoch_time:.2f}"
    )
    
    return average_epoch_loss, final_epoch_accuracy, final_epoch_mean_dice, per_class_dice_scores

def test(model, validation_loader, criterion, device, phase_name="Validation", num_classes=None):
    """
    Comprehensive evaluation function for semantic segmentation models.
    
    Args:
        model (torch.nn.Module): The neural network model to evaluate
        validation_loader (DataLoader): PyTorch DataLoader containing validation/test data
        criterion (torch.nn.Module): Loss function
        device (torch.device): Computation device (CPU/GPU)
        phase_name (str): Name of the evaluation phase (e.g., "Validation" or "Test")
        num_classes (int, optional): Number of classes for detailed metrics
        
    Returns:
        tuple: Contains (average_loss, accuracy, mean_dice_score, class_dice_scores)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize tracking variables
    total_samples_processed = 0
    correctly_classified_pixels = 0
    cumulative_loss = 0.0
    batch_count = 0
    
    # Lists to store predictions and ground truth
    all_predictions = []
    all_ground_truth_labels = []
    
    # Track execution time
    evaluation_start_time = time.time()
    
    # Create progress bar
    progress_bar = tqdm(
        validation_loader, 
        desc=f"{phase_name} Phase",
        leave=True
    )
    
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        # Iterate through each batch
        for batch_idx, (input_images, target_labels) in enumerate(progress_bar):
            # Record batch start time
            batch_start_time = time.time()
            
            # Transfer data to device
            input_images = input_images.to(device)
            target_labels = target_labels.to(device)
            
            # Forward pass
            model_outputs = model(input_images)
            
            # Handle different model output formats
            if isinstance(model_outputs, dict) and 'out' in model_outputs:
                segmentation_logits = model_outputs['out']
            else:
                segmentation_logits = model_outputs
                
            # Calculate loss
            current_batch_loss = criterion(segmentation_logits, target_labels)
            
            # Accumulate batch loss
            cumulative_loss += current_batch_loss.item()
            batch_count += 1
            
            # Compute predicted class indices
            _, predicted_class_indices = torch.max(segmentation_logits, dim=1)
            
            # Calculate pixel-level accuracy
            correctly_classified_pixels += (predicted_class_indices == target_labels).sum().item()
            total_samples_processed += target_labels.numel()
            
            # Store predictions and ground truth
            all_predictions.append(predicted_class_indices.detach().cpu())
            all_ground_truth_labels.append(target_labels.detach().cpu())
            
            # Calculate batch processing time
            batch_processing_time = time.time() - batch_start_time
            
            # Update progress bar
            current_accuracy = correctly_classified_pixels / max(total_samples_processed, 1) * 100
            progress_bar.set_postfix({
                'loss': f"{current_batch_loss.item():.4f}",
                'accuracy': f"{current_accuracy:.2f}%",
                'batch_time': f"{batch_processing_time:.3f}s"
            })
    
    # Concatenate all batches into single tensors
    all_predictions_tensor = torch.cat(all_predictions)
    all_ground_truth_tensor = torch.cat(all_ground_truth_labels)
    
    # Calculate final metrics
    final_dataset_accuracy = calculate_accuracy(all_predictions_tensor, all_ground_truth_tensor)
    final_dataset_mean_dice = calculate_mean_dice_score(all_predictions_tensor, all_ground_truth_tensor, num_classes)
    
    # Calculate per-class dice scores if num_classes is provided
    per_class_dice_scores = None
    if num_classes is not None:
        per_class_dice_scores = calculate_per_class_dice(all_predictions_tensor, all_ground_truth_tensor, num_classes)
        
        # Log per-class performance
        logger.info(f"Per-class {phase_name} Dice Scores:")
        for class_idx, dice_score in enumerate(per_class_dice_scores):
            logger.info(f"  Class {class_idx}: {dice_score:.4f}")
    
    # Calculate average loss
    average_dataset_loss = cumulative_loss / max(batch_count, 1)
    
    # Calculate total evaluation time
    total_evaluation_time = time.time() - evaluation_start_time
    
    # Log detailed evaluation summary
    logger.info(
        f"\n{phase_name} Summary | "
        f"Loss: {average_dataset_loss:.4f} | "
        f"Accuracy: {final_dataset_accuracy:.2f}% | "
        f"Mean Dice: {final_dataset_mean_dice:.4f} | "
        f"Duration: {total_evaluation_time:.2f}s | "
        f"Images/sec: {len(validation_loader.dataset)/total_evaluation_time:.2f}"
    )
    
    return average_dataset_loss, final_dataset_accuracy, final_dataset_mean_dice, per_class_dice_scores

def plot_training_results(performance_history, experiment_name, num_classes=None):
    """
    Create and save plots visualizing training performance.
    
    Args:
        performance_history (dict): Dictionary containing training metrics
        experiment_name (str): Name of the experiment for file naming
        num_classes (int, optional): Number of classes for per-class plots
    """
    # Create performance plots
    plt.figure(figsize=(15, 10))

    # Plot training and validation losses
    plt.subplot(2, 2, 1)
    plt.plot(performance_history['epoch_numbers'], performance_history['training']['losses'], 'b-', label='Training Loss')
    plt.plot(performance_history['epoch_numbers'], performance_history['validation']['losses'], 'r-', label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracies
    plt.subplot(2, 2, 2)
    plt.plot(performance_history['epoch_numbers'], performance_history['training']['accuracies'], 'b-', label='Training Accuracy')
    plt.plot(performance_history['epoch_numbers'], performance_history['validation']['accuracies'], 'r-', label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot training and validation mean Dice scores
    plt.subplot(2, 2, 3)
    plt.plot(performance_history['epoch_numbers'], performance_history['training']['mean_dice_scores'], 'b-', label='Training Mean Dice')
    plt.plot(performance_history['epoch_numbers'], performance_history['validation']['mean_dice_scores'], 'r-', label='Validation Mean Dice')
    plt.title('Mean Dice Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Dice Score')
    plt.legend()
    plt.grid(True)

    # Plot epoch durations
    plt.subplot(2, 2, 4)
    plt.plot(performance_history['epoch_numbers'], performance_history['training']['epoch_durations'], 'b-', label='Training Time')
    plt.plot(performance_history['epoch_numbers'], performance_history['validation']['epoch_durations'], 'r-', label='Validation Time')
    plt.title('Computation Time Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    # Save the performance plots
    plt.tight_layout()
    plt.savefig(os.path.join("training_results/plots", f"{experiment_name}_performance.png"), dpi=300)
    plt.close()

    # If per-class dice scores are available, create per-class performance plots
    if performance_history['training']['per_class_dice_scores'] and num_classes is not None:
        plt.figure(figsize=(15, 10))
        
        # Extract last epoch's per-class dice scores
        final_train_per_class_dice = performance_history['training']['per_class_dice_scores'][-1]
        final_val_per_class_dice = performance_history['validation']['per_class_dice_scores'][-1]
        
        # Create bar chart of per-class performance
        class_indices = np.arange(num_classes)
        bar_width = 0.35
        
        plt.bar(class_indices - bar_width/2, final_train_per_class_dice, bar_width, label='Training')
        plt.bar(class_indices + bar_width/2, final_val_per_class_dice, bar_width, label='Validation')
        
        plt.xlabel('Class Index')
        plt.ylabel('Dice Score')
        plt.title('Per-Class Dice Scores (Final Epoch)')
        plt.xticks(class_indices)
        plt.legend()
        plt.grid(True, axis='y')
        
        # Save the per-class performance plot
        plt.tight_layout()
        plt.savefig(os.path.join("training_results/plots", f"{experiment_name}_per_class_dice.png"), dpi=300)
        plt.close()

    logger.info(f"Performance visualizations saved to training_results/plots/{experiment_name}_performance.png")

def main():
    # Ask for dataset directory
    data_dir = "../data/00_raw"

    # Setup paths
    data_paths = setup_fashionpedia(data_dir)

    # Load category mappings
    category_mappings = load_fashionpedia_categories(data_paths['train_ann_file'])
    print(f"Total number of classes: {category_mappings['num_classes']}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(data_paths, category_mappings, batch_size=2)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Set number of classes
    num_classes = 28

    # Choose model
    model_options = ["deeplab", "segformer", "lraspp"]
    model = choose_model(model_options[2], num_classes)

    # Define criterion, optimizer and metrics
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    mDice = Dice(average='macro', num_classes=num_classes)

    # Initialize Performance Metric Tracking
    performance_history = {
        'epoch_numbers': [],
        'training': {
            'losses': [],
            'accuracies': [],
            'mean_dice_scores': [],
            'per_class_dice_scores': [],
            'epoch_durations': []
        },
        'validation': {
            'losses': [],
            'accuracies': [],
            'mean_dice_scores': [],
            'per_class_dice_scores': [],
            'epoch_durations': []
        },
        'best_model': {
            'epoch': 0,
            'validation_dice': 0.0,
            'validation_loss': float('inf')
        }
    }

    # Configure training parameters
    num_epochs = 2
    save_checkpoint_frequency = 1
    experiment_name = f"segmentation_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Log training configuration
    logger.info("=" * 80)
    logger.info(f"STARTING TRAINING EXPERIMENT: {experiment_name}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training dataset size: {len(train_loader.dataset)} samples")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)} samples")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)

    # Record overall training start time
    training_start_time = time.time()

    # Tracking simple metrics for final plot
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    train_mdice, test_mdice = [], []

    # Execute training over specified number of epochs
    for epoch_index in range(num_epochs):
        current_epoch = epoch_index + 1
        logger.info(f"\n{'='*30} EPOCH {current_epoch}/{num_epochs} {'='*30}")
        
        # Record epoch start time
        epoch_start_time = time.time()
        
        # Training Phase
        logger.info(f"Beginning training phase for epoch {current_epoch}...")
        train_loss, train_accuracy, train_mean_dice, train_per_class_dice = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_num=current_epoch,
            num_classes=num_classes
        )
        
        # Record training phase duration
        training_phase_duration = time.time() - epoch_start_time
        
        # Validation Phase
        logger.info(f"Beginning validation phase for epoch {current_epoch}...")
        validation_phase_start_time = time.time()
        
        test_loss, test_accuracy, test_mean_dice, test_per_class_dice = test(
            model=model,
            validation_loader=val_loader,
            criterion=criterion,
            device=device,
            phase_name="Validation",
            num_classes=num_classes
        )
        
        # Record validation phase duration
        validation_phase_duration = time.time() - validation_phase_start_time
        
        # Update Performance History
        # Store epoch number
        performance_history['epoch_numbers'].append(current_epoch)
        
        # Store training metrics
        performance_history['training']['losses'].append(train_loss)
        performance_history['training']['accuracies'].append(train_accuracy)
        performance_history['training']['mean_dice_scores'].append(train_mean_dice)
        performance_history['training']['per_class_dice_scores'].append(train_per_class_dice)
        performance_history['training']['epoch_durations'].append(training_phase_duration)
        
        # Store validation metrics
        performance_history['validation']['losses'].append(test_loss)
        performance_history['validation']['accuracies'].append(test_accuracy)
        performance_history['validation']['mean_dice_scores'].append(test_mean_dice)
        performance_history['validation']['per_class_dice_scores'].append(test_per_class_dice)
        performance_history['validation']['epoch_durations'].append(validation_phase_duration)
        
        # Calculate total epoch time
        total_epoch_duration = time.time() - epoch_start_time
        
        # Format metrics with consistent decimal precision
        formatted_metrics = {
            'train_loss': f"{train_loss:.4f}",
            'train_accuracy': f"{train_accuracy:.2f}%",
            'train_mean_dice': f"{train_mean_dice:.4f}",
            'test_loss': f"{test_loss:.4f}",
            'test_accuracy': f"{test_accuracy:.2f}%",
            'test_mean_dice': f"{test_mean_dice:.4f}",
            'epoch_duration': f"{total_epoch_duration:.2f}s"
        }
        
        # Generate comprehensive epoch summary
        epoch_summary = (
            f"Epoch {current_epoch}/{num_epochs} Summary:\n"
            f"  Training:   Loss: {formatted_metrics['train_loss']}, "
            f"Accuracy: {formatted_metrics['train_accuracy']}, "
            f"Mean Dice: {formatted_metrics['train_mean_dice']}\n"
            f"  Validation: Loss: {formatted_metrics['test_loss']}, "
            f"Accuracy: {formatted_metrics['test_accuracy']}, "
            f"Mean Dice: {formatted_metrics['test_mean_dice']}\n"
            f"  Time: {formatted_metrics['epoch_duration']}"
        )
        
        # Log and print epoch summary
        logger.info(epoch_summary)
        print(epoch_summary)
        
        # Store metrics for simple final plot
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_mdice.append(train_mean_dice)
        test_mdice.append(test_mean_dice)
        
        # Best Model Tracking and Checkpointing
        # Check if current model is the best so far
        is_best_model = test_mean_dice > performance_history['best_model']['validation_dice']
        
        if is_best_model:
            performance_history['best_model']['epoch'] = current_epoch
            performance_history['best_model']['validation_dice'] = test_mean_dice
            performance_history['best_model']['validation_loss'] = test_loss
            
            # Save best model checkpoint
            best_model_path = os.path.join("training_results/checkpoints", f"{experiment_name}_best_model.pth")
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_mean_dice': test_mean_dice,
                'performance_history': performance_history
            }, best_model_path)
            
            logger.info(f"New best model saved! Validation Dice: {test_mean_dice:.4f}")
        
        # Save regular checkpoint based on specified frequency
        if current_epoch % save_checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                "training_results/checkpoints", 
                f"{experiment_name}_epoch_{current_epoch}.pth"
            )
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'performance_history': performance_history
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved at epoch {current_epoch}")

    # Calculate total training time
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Generate comprehensive training summary
    training_summary = (
        f"\n{'='*30} TRAINING COMPLETE {'='*30}\n"
        f"Experiment name: {experiment_name}\n"
        f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n"
        f"Best model performance (Epoch {performance_history['best_model']['epoch']}):\n"
        f"  Validation Loss: {performance_history['best_model']['validation_loss']:.4f}\n"
        f"  Validation Mean Dice: {performance_history['best_model']['validation_dice']:.4f}\n"
        f"Final model performance (Epoch {num_epochs}):\n"
        f"  Validation Loss: {test_loss:.4f}\n"
        f"  Validation Mean Dice: {test_mean_dice:.4f}"
    )
    
    print(training_summary)
    logger.info(training_summary)
    
    # Generate and save training performance plots
    plot_training_results(performance_history, experiment_name, num_classes)
    
    logger.info("Training experiment completed successfully!")
    
if __name__ == '__main__':
    main()