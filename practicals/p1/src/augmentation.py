import tensorflow as tf
from keras import layers, Sequential
from config import *


def get_augmentation_pipeline(use_augmentation=True):
    """Returns a Keras Sequential pipeline for augmentation."""
    if not use_augmentation:
        return layers.Identity()

    return Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.2),
        ],
        name="augmentation_pipeline",
    )


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


def create_data_pipeline(is_training=True):
    """Creates a complete data processing pipeline combining preprocessing and augmentation."""
    preprocessing = get_preprocessing_pipeline()
    augmentation = get_augmentation_pipeline() if is_training else layers.Identity()

    return Sequential([preprocessing, augmentation], name="data_pipeline")
