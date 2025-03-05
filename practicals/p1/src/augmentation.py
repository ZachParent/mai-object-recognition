from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

test_data_gen_args = dict(
    rescale=None if per_sample_normalization else 1.0 / 255,
    samplewise_center=True if per_sample_normalization else False,
    samplewise_std_normalization=True if per_sample_normalization else False,
)

train_data_gen_args = (
    dict(
        rescale=None if per_sample_normalization else 1.0 / 255,
        samplewise_center=True if per_sample_normalization else False,
        samplewise_std_normalization=True if per_sample_normalization else False,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
    )
    if data_augmentation
    else test_data_gen_args
)

training_datagen = ImageDataGenerator(**train_data_gen_args)

test_datagen = ImageDataGenerator(**test_data_gen_args)
