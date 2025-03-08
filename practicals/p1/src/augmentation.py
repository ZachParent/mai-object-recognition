import tensorflow as tf
import random
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
    
    This implementation is compatible with TensorFlow graph execution.

    Args:
        p: Probability of applying the erasing
        area_range: Range of area ratio to erase (min, max)
        aspect_ratio_range: Range of aspect ratio for the erased area
        value: Value to fill the erased area with

    Returns:
        A Lambda layer that applies random erasing
    """

    def random_erasing(image):
        # Get image dimensions - handle both batched and unbatched inputs
        shape = tf.shape(image)
        if len(image.shape) == 4:  # Batched input
            batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
            # Process each image in the batch separately
            def process_single_image(img):
                return _apply_random_erasing(img, height, width, channels)
            
            # Apply random erasing to each image in the batch with probability p
            should_apply = tf.random.uniform(shape=[batch_size]) < p
            processed_images = tf.map_fn(
                lambda x: tf.cond(
                    x[0],
                    lambda: process_single_image(x[1]),
                    lambda: x[1]
                ),
                (should_apply, image),
                dtype=image.dtype
            )
            return processed_images
        else:  # Unbatched input
            height, width, channels = shape[0], shape[1], shape[2]
            # Apply random erasing with probability p
            return tf.cond(
                tf.random.uniform(()) < p,
                lambda: _apply_random_erasing(image, height, width, channels),
                lambda: image
            )
    
    def _apply_random_erasing(image, height, width, channels):
        # Calculate area to erase
        image_area = tf.cast(height * width, tf.float32)
        area_ratio = tf.random.uniform(
            shape=(), minval=area_range[0], maxval=area_range[1]
        )
        erase_area = area_ratio * image_area
        
        # Calculate aspect ratio
        aspect_ratio = tf.random.uniform(
            shape=(), minval=aspect_ratio_range[0], maxval=aspect_ratio_range[1]
        )
        
        # Calculate height and width of erasing rectangle
        h = tf.cast(tf.sqrt(erase_area / aspect_ratio), tf.int32)
        w = tf.cast(aspect_ratio * tf.cast(h, tf.float32), tf.int32)
        
        # Ensure h and w are not larger than image dimensions
        h = tf.minimum(h, height)
        w = tf.minimum(w, width)
        
        # Random position
        x = tf.random.uniform(shape=(), minval=0, maxval=width - w + 1, dtype=tf.int32)
        y = tf.random.uniform(shape=(), minval=0, maxval=height - h + 1, dtype=tf.int32)
        
        # Create mask
        # First, create a ones mask for the area to erase
        erase_mask = tf.ones([h, w, channels]) * value
        
        # Then, create indices for that area in the original image
        indices = tf.meshgrid(
            tf.range(y, y + h),
            tf.range(x, x + w),
            tf.range(channels),
            indexing='ij'
        )
        indices = tf.stack(indices, axis=-1)
        indices = tf.reshape(indices, [-1, 3])
        
        # Create a copy of the image
        erased_image = tf.identity(image)
        
        # Use tensor_scatter_nd_update to overwrite the pixels in the erase area
        # We need to use flat values for the update
        updates = tf.ones([h * w * channels], dtype=image.dtype) * value
        
        return tf.tensor_scatter_nd_update(erased_image, indices, updates)
    
    return layers.Lambda(random_erasing, name="random_erasing")


def get_augmentation_pipeline(use_augmentation=True, augmentation="simple"):
    """Returns a Keras Sequential pipeline for augmentation.

    Args:
        use_augmentation: Whether to use basic augmentation
        advanced_augmentation: Whether to add the advanced augmentation techniques
    """
    if not use_augmentation:
        return layers.Identity()

    augmentation_layers = []

    if augmentation == "simple" or augmentation == "all":
        #Add simple augmentation layers
        augmentation_layers.append(layers.RandomFlip("horizontal_and_vertical"))
        augmentation_layers.append(layers.RandomRotation(0.2))
        augmentation_layers.append(layers.RandomTranslation(0.1, 0.1))
        augmentation_layers.append(layers.RandomZoom(0.2))


    # Add advanced augmentation layers if requested
    if augmentation == "color" or augmentation == "all":
        # Add color jittering
        augmentation_layers.append(color_jitter_layer())

    if augmentation == "occlusion" or augmentation == "all":
        # Add random erasing/cutout
        augmentation_layers.append(random_erasing_layer())

    return Sequential(augmentation_layers, name="augmentation_pipeline")




def create_augmentation_pipeline(augmentation="simple"):
    """Creates a complete data processing pipeline combining preprocessing and augmentation.

    Args:
        is_training: Whether pipeline is used during training (applying augmentation)
        advanced_augmentation: Whether to add advanced augmentation techniques
    """
    augmentation = (
        get_augmentation_pipeline(augmentation=augmentation)
        if augmentation is not None
        else layers.Identity()
    )

    return Sequential(augmentation, name="data_pipeline")


def apply_augmentation(dataset, augmentation="simple"):
    """Applies augmentation to a dataset based on the desired type.

        Args:
        dataset: Dataset to modify
        augmentation: Type of augmentation to apply
    """


    augmentation_pipeline = create_augmentation_pipeline(augmentation)
    aug_dataset = dataset.map(
        lambda x, y: (augmentation_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.concatenate(aug_dataset)

    return dataset