import cv2
import xml.etree.ElementTree as ET
import tensorflow as tf
from config import voc_classes, num_classes, RAW_DATA_DIR, img_size
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
        classes = np.zeros(num_classes, dtype=np.float32)

        # Find all object classes and set corresponding indices to 1
        for boxes in root.iter("object"):
            classname = boxes.find("name").text
            class_idx = voc_classes[classname]  # Convert string label to index
            classes[class_idx] = 1.0

        return classes

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return np.zeros(num_classes, dtype=np.float32)


@tf.function
def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    # Read image file
    img = tf.io.read_file(image_path)
    # Decode JPEG
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize to fixed dimensions
    img = tf.image.resize(img, [img_size, img_size])  # Use consistent size from config
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
        labels.set_shape([num_classes])

        return image, labels

    # Create dataset from paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotation_paths))

    # Load and preprocess samples
    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def get_file_paths(file_list):
    """Convert list of file IDs to full image and annotation paths."""
    image_paths = [str(RAW_DATA_DIR / "JPEGImages" / f"{id}.jpg") for id in file_list]
    annotation_paths = [
        str(RAW_DATA_DIR / "Annotations" / f"{id}.xml") for id in file_list
    ]
    return image_paths, annotation_paths


def read_content(xml_file: str):
    """Legacy function to read XML content."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_objects = []
    for boxes in root.iter("object"):
        classname = boxes.find("name").text
        list_with_all_objects.append(voc_classes[classname])

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_objects, list_with_all_boxes


def load_batch(data_list, step, batch_size, raw_data_dir, img_size):
    """Legacy function to load a batch of data."""
    X, Y = [], []
    for f in data_list[step * batch_size : (step + 1) * batch_size]:
        img = cv2.imread(raw_data_dir / "JPEGImages" / (f + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)

        classes = np.zeros(num_classes)
        try:
            cnames, _ = read_content(raw_data_dir / "Annotations" / (f + ".xml"))
        except:
            print(f)
        for c in cnames:
            classes[c] = 1.0
        Y.append(classes)

    return (np.array(X), np.array(Y))
