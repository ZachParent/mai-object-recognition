import xml.etree.ElementTree as ET
import tensorflow as tf
from keras import layers, Sequential
from config import VOC_CLASSES, NUM_CLASSES, RAW_DATA_DIR, IMG_SIZE
import numpy as np


def parse_xml_annotation(xml_path):
    """Parse XML annotation file and return list of object classes."""
    try:
        # Decode tensor to string if needed
        if isinstance(xml_path, tf.Tensor):
            xml_path = xml_path.numpy().decode("utf-8")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Initialize multi-hot encoding array
        classes = np.zeros(NUM_CLASSES, dtype=np.float32)

        # Find all object classes and set corresponding indices to 1
        for boxes in root.iter("object"):
            classname = boxes.find("name").text
            class_idx = VOC_CLASSES[classname]  # Convert string label to index
            classes[class_idx] = 1.0

        return classes

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return np.zeros(NUM_CLASSES, dtype=np.float32)


@tf.function
def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    # Read image file
    img = tf.io.read_file(image_path)
    # Decode JPEG
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize to fixed dimensions
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])  # Use consistent size from config
    # Convert to float32
    img = tf.cast(img, tf.float32)
    return img


def get_dataset_from_paths(image_paths, annotation_paths):
    """Create a tf.data.Dataset from lists of image and annotation paths."""

    def load_sample(img_path, xml_path):
        # Load and preprocess image
        image = load_and_preprocess_image(img_path)

        # Load and parse annotation
        labels = tf.py_function(
            parse_xml_annotation, [xml_path], tf.float32
        )  # Change back to float32
        labels.set_shape([NUM_CLASSES])

        return image, labels

    # Create dataset from paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotation_paths))

    # Load and preprocess samples
    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    return dataset


def get_file_paths(file_list):
    """Convert list of file IDs to full image and annotation paths."""
    image_paths = [str(RAW_DATA_DIR / "JPEGImages" / f"{id}.jpg") for id in file_list]
    annotation_paths = [
        str(RAW_DATA_DIR / "Annotations" / f"{id}.xml") for id in file_list
    ]
    return image_paths, annotation_paths


def get_preprocessing_pipeline(use_normalization=True, per_sample_normalization=False):
    """Returns a Keras Sequential pipeline for preprocessing."""
    if not use_normalization:
        return layers.Identity()

    preprocessing_layers = []
    if per_sample_normalization:
        # Use Lambda layer for per-sample normalization
        def normalize(x):
            mean = tf.reduce_mean(x, axis=[0, 1], keepdims=True)
            std = tf.math.reduce_std(x, axis=[0, 1], keepdims=True)
            return (x - mean) / (std + 1e-7)

        preprocessing_layers.append(layers.Lambda(normalize))
    else:
        preprocessing_layers.append(layers.Rescaling(1.0 / 255))

    return Sequential(preprocessing_layers, name="preprocessing_pipeline")


def create_dataset(file_list, batch_size):
    """Create a tf.data.Dataset from a list of file paths."""
    # Get full paths for images and annotations
    image_paths, annotation_paths = get_file_paths(file_list)

    # Create base dataset
    dataset = get_dataset_from_paths(image_paths, annotation_paths)

    # Create and apply the data pipeline
    preprocessing_pipeline = get_preprocessing_pipeline()

    # Apply the pipeline to the images
    dataset = dataset.map(
        lambda x, y: (preprocessing_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    # # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_data(file_path):
    """Load data from a text file."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]
