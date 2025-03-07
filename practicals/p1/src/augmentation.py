import tensorflow as tf
from keras import layers, Sequential
from config import *


def color_jitter_layer(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """Creates a layer that randomly adjusts color properties.

    Args:
        brightness: Maximum delta for brightness adjustment
        contrast: Maximum delta for contrast adjustment
        saturation: Maximum delta for saturation adjustment
        hue: Maximum delta for hue adjustment

    Returns:
        A Lambda layer that applies random color jittering
    """

    def color_jitter(image):
        # Apply random brightness
        if brightness > 0:
            image = tf.image.random_brightness(image, brightness)

        # Apply random contrast
        if contrast > 0:
            image = tf.image.random_contrast(image, 1 - contrast, 1 + contrast)

        # Apply random saturation
        if saturation > 0:
            image = tf.image.random_saturation(image, 1 - saturation, 1 + saturation)

        # Apply random hue
        if hue > 0:
            image = tf.image.random_hue(image, hue)

        # Make sure values stay in valid range
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image

    return layers.Lambda(color_jitter, name="color_jitter")


def random_erasing_layer(
    p=0.5, area_range=(0.02, 0.2), aspect_ratio_range=(0.3, 3.0), value=0
):
    """Creates a layer that randomly erases rectangles from the image.

    Args:
        p: Probability of applying the erasing
        area_range: Range of area ratio to erase (min, max)
        aspect_ratio_range: Range of aspect ratio for the erased area
        value: Value to fill the erased area with

    Returns:
        A Lambda layer that applies random erasing
    """

    def random_erasing(image):
        if tf.random.uniform(()) > p:
            return image

        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        image_area = tf.cast(height * width, tf.float32)

        # Random erasing area
        area_ratio = tf.random.uniform((), area_range[0], area_range[1])
        erase_area = area_ratio * image_area

        # Random aspect ratio
        aspect_ratio = tf.random.uniform(
            (), aspect_ratio_range[0], aspect_ratio_range[1]
        )

        # Calculate height and width of the erase rectangle
        h = tf.math.sqrt(erase_area / aspect_ratio)
        w = aspect_ratio * h

        # Convert to integers
        h = tf.cast(h, tf.int32)
        w = tf.cast(w, tf.int32)

        # Ensure h and w are not larger than image dimensions
        h = tf.minimum(h, height)
        w = tf.minimum(w, width)

        # Random position
        x = tf.random.uniform((), 0, width - w, dtype=tf.int32)
        y = tf.random.uniform((), 0, height - h, dtype=tf.int32)

        # Create mask for erasing
        mask = tf.ones((height, width, 3), dtype=image.dtype) * value

        # Create box indices
        top_left_y = y
        top_left_x = x
        bottom_right_y = y + h
        bottom_right_x = x + w

        # Create the erased image
        erased_area = mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]

        # Update the image with the erased area
        indices = tf.TensorArray(tf.int32, size=h * w * 3)
        values = tf.TensorArray(tf.float32, size=h * w * 3)

        # Fill indices and values
        counter = 0
        for i in range(h):
            for j in range(w):
                for k in range(3):
                    indices = indices.write(
                        counter, [top_left_y + i, top_left_x + j, k]
                    )
                    values = values.write(counter, value)
                    counter += 1

        indices = indices.stack()
        values = values.stack()

        # Apply the erasing using scatter_nd
        return tf.tensor_scatter_nd_update(image, indices, values)

    return layers.Lambda(random_erasing, name="random_erasing")


def get_augmentation_pipeline(use_augmentation=True, advanced_augmentation=False):
    """Returns a Keras Sequential pipeline for augmentation.

    Args:
        use_augmentation: Whether to use basic augmentation
        advanced_augmentation: Whether to add the advanced augmentation techniques
    """
    if not use_augmentation:
        return layers.Identity()

    augmentation_layers = [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.2),
    ]

    # Add advanced augmentation layers if requested
    if advanced_augmentation:
        # Add color jittering
        augmentation_layers.append(color_jitter_layer())

        # Add random erasing/cutout
        augmentation_layers.append(random_erasing_layer())

    return Sequential(augmentation_layers, name="augmentation_pipeline")


def get_preprocessing_pipeline(use_normalization=True):
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


def create_data_pipeline(is_training=True, advanced_augmentation=False):
    """Creates a complete data processing pipeline combining preprocessing and augmentation.

    Args:
        is_training: Whether pipeline is used during training (applying augmentation)
        advanced_augmentation: Whether to add advanced augmentation techniques
    """
    preprocessing = get_preprocessing_pipeline()
    augmentation = (
        get_augmentation_pipeline(
            use_augmentation=is_training, advanced_augmentation=advanced_augmentation
        )
        if is_training
        else layers.Identity()
    )

    return Sequential([preprocessing, augmentation], name="data_pipeline")
