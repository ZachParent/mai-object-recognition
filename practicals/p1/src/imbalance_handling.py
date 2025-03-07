import tensorflow as tf
import numpy as np
from config import num_classes, voc_classes, RAW_DATA_DIR, img_size
import xml.etree.ElementTree as ET
from pathlib import Path
import random
from load_data import load_and_preprocess_image, parse_xml_annotation, get_file_paths
from augmentation import create_data_pipeline


def analyze_class_distribution(file_list):
    """
    Analyze the class distribution in the dataset.
    
    Args:
        file_list: List of file IDs
        
    Returns:
        class_counts: Dictionary mapping class index to count
    """
    class_counts = {i: 0 for i in range(num_classes)}
    annotations_dir = RAW_DATA_DIR / "Annotations"
    
    for file_id in file_list:
        xml_path = annotations_dir / f"{file_id}.xml"
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for boxes in root.iter("object"):
                classname = boxes.find("name").text
                class_idx = voc_classes[classname]
                class_counts[class_idx] += 1
                
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
    
    return class_counts


def create_class_to_files_mapping(file_list):
    """
    Create a mapping from class indices to file IDs containing that class.
    
    Args:
        file_list: List of file IDs
        
    Returns:
        class_to_files: Dictionary mapping class indices to lists of file IDs
    """
    class_to_files = {i: [] for i in range(num_classes)}
    annotations_dir = RAW_DATA_DIR / "Annotations"
    
    for file_id in file_list:
        xml_path = annotations_dir / f"{file_id}.xml"
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for boxes in root.iter("object"):
                classname = boxes.find("name").text
                class_idx = voc_classes[classname]
                
                # Add this file to the corresponding class list if not already there
                if file_id not in class_to_files[class_idx]:
                    class_to_files[class_idx].append(file_id)
                    
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
    
    return class_to_files


def generate_balanced_batch_ids(file_list, batch_size=32, min_samples_per_class=1):
    """
    Generate a batch of file IDs with better class balance.
    
    Args:
        file_list: List of all file IDs
        batch_size: Size of the batch to generate
        min_samples_per_class: Minimum samples to include for each class
        
    Returns:
        balanced_file_list: List of file IDs with improved class balance
    """
    # Create mapping of classes to files
    class_to_files = create_class_to_files_mapping(file_list)
    
    # Count files per class
    files_per_class = {cls: len(files) for cls, files in class_to_files.items()}
    
    # Calculate target number of files per class using square root scaling
    total_sqrt = sum(np.sqrt(count) for count in files_per_class.values() if count > 0)
    targets_per_class = {
        cls: max(min_samples_per_class, int((np.sqrt(count) / total_sqrt) * batch_size))
        if count > 0 else 0
        for cls, count in files_per_class.items()
    }
    
    # Adjust if total exceeds batch size
    total_target = sum(targets_per_class.values())
    if total_target > batch_size:
        scale = batch_size / total_target
        targets_per_class = {
            cls: max(min_samples_per_class, int(target * scale))
            if cls in class_to_files and len(class_to_files[cls]) > 0 else 0
            for cls, target in targets_per_class.items()
        }
    
    # Select files for each class
    selected_files = set()
    for cls, target in targets_per_class.items():
        if target <= 0 or len(class_to_files[cls]) == 0:
            continue
            
        # Randomly select files for this class
        cls_files = class_to_files[cls]
        selection = random.sample(
            cls_files, 
            min(target, len(cls_files))
        )
        selected_files.update(selection)
    
    # If we need more files, randomly select from remaining files
    remaining = batch_size - len(selected_files)
    if remaining > 0:
        remaining_files = list(set(file_list) - selected_files)
        if remaining_files:
            additional = random.sample(
                remaining_files,
                min(remaining, len(remaining_files))
            )
            selected_files.update(additional)
    
    # Convert to list and ensure batch_size length
    balanced_files = list(selected_files)
    
    # If still not enough, allow duplicates (focus on minority classes)
    if len(balanced_files) < batch_size:
        # Sort classes by frequency (ascending)
        rare_classes = sorted(files_per_class.keys(), key=lambda c: files_per_class[c])
        
        # Add samples from rare classes
        needed = batch_size - len(balanced_files)
        for cls in rare_classes:
            if needed <= 0:
                break
                
            if len(class_to_files[cls]) > 0:
                # Determine how many to add
                to_add = min(needed, len(class_to_files[cls]))
                
                # Select random samples (with replacement possible)
                additional = random.choices(class_to_files[cls], k=to_add)
                balanced_files.extend(additional)
                needed -= to_add
    
    # Shuffle final balanced batch
    random.shuffle(balanced_files)
    
    # Ensure exactly batch_size elements
    if len(balanced_files) > batch_size:
        balanced_files = balanced_files[:batch_size]
    
    return balanced_files


def load_balanced_batch(file_ids):
    """
    Load a batch of balanced images and labels.
    
    Args:
        file_ids: List of file IDs to load
        
    Returns:
        images: List of loaded and preprocessed images
        labels: List of parsed labels
    """
    images = []
    labels = []
    
    # Get full paths for each file ID
    image_paths, annotation_paths = get_file_paths(file_ids)
    
    for img_path, xml_path in zip(image_paths, annotation_paths):
        # Load and preprocess image
        image = load_and_preprocess_image(img_path)
        images.append(image)
        
        # Parse annotation
        label = parse_xml_annotation(xml_path)
        labels.append(label)
    
    return np.array(images), np.array(labels)


def create_balanced_dataset_generator(file_list, batch_size=32):
    """
    Create a generator that yields balanced batches of opened files.
    
    Args:
        file_list: List of file IDs
        batch_size: Size of batches to generate
        
    Returns:
        generator: A generator yielding (images, labels) tuples
    """
    def generator():
        while True:
            # Generate balanced batch IDs
            balanced_ids = generate_balanced_batch_ids(file_list, batch_size)
            
            # Load the actual images and labels
            images, labels = load_balanced_batch(balanced_ids)
            
            yield images, labels
    
    return generator


def create_balanced_dataset(file_list, is_training=True, batch_size=32, augmentation=None):
    """
    Create a TensorFlow dataset with balanced class distribution.
    
    Args:
        file_list: List of file IDs
        is_training: Whether this is a training dataset
        batch_size: Batch size
        augmentation: Type of augmentation to apply
        
    Returns:
        dataset: A TensorFlow dataset with balanced batches
    """
    # Create generator function
    gen = create_balanced_dataset_generator(file_list, batch_size)
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.float32)
        )
    )
    
    # Create and apply the data pipeline
    data_pipeline = create_data_pipeline(is_training, augmentation)

    # Apply the pipeline to the images
    dataset = dataset.map(
        lambda x, y: (data_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# Method 1: Class-Weighted Loss
def create_class_weights(file_list, power=0.5):
    """
    Create weights for each class based on inverse frequency.
    
    Args:
        file_list: List of file IDs
        power: Power factor to control weight magnitude
        
    Returns:
        class_weights: Dictionary of class weights
    """
    class_counts = analyze_class_distribution(file_list)
    
    # Calculate inverse frequency with smoothing
    total_samples = sum(class_counts.values())
    inv_frequencies = {
        cls: (total_samples / (count + 1))
        for cls, count in class_counts.items()
    }
    
    # Apply power to control weight magnitude
    class_weights = {
        cls: (freq ** power)
        for cls, freq in inv_frequencies.items()
    }
    
    # Normalize weights
    max_weight = max(class_weights.values())
    class_weights = {
        cls: weight / max_weight 
        for cls, weight in class_weights.items()
    }
    
    return class_weights


def create_weighted_binary_crossentropy(file_list, power=0.5):
    """
    Create a weighted binary crossentropy loss function.
    
    Args:
        file_list: List of file IDs
        power: Power factor for class weight calculation
        
    Returns:
        weighted_loss: Custom loss function
    """
    # Calculate class weights
    class_weights = create_class_weights(file_list, power)
    
    # Convert to tensor
    weight_values = [class_weights[i] for i in range(num_classes)]
    weight_tensor = tf.constant(weight_values, dtype=tf.float32)
    
    def weighted_loss(y_true, y_pred):
        # Base binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Apply weights based on true classes
        weights = tf.reduce_sum(
            y_true * tf.reshape(weight_tensor, [1, num_classes]) + 
            (1 - y_true) * 1.0,
            axis=-1
        )
        
        # Return weighted mean
        return tf.reduce_mean(bce * weights)
    
    return weighted_loss