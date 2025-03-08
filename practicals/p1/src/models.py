import random
import csv
import os
import tensorflow as tf
from keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    MultiHeadAttention,
    LayerNormalization,
    Average,
    Dropout,
    Layer,
)
from tensorflow.keras.models import Model

from config import *
from experiment_config import ExperimentConfig
import tensorflow.keras.applications as app


class ExpandDims(Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=1)


class Squeeze(Layer):
    def call(self, x):
        return tf.squeeze(x, axis=1)


def setup_best_config(
    exp: ExperimentConfig,
    params_to_reuse: list[str],
    best_experiment_config: ExperimentConfig = None,
):
    """Setup the best configuration based on previous experiments.

    Args:
        exp: The experiment config to update
        params_to_reuse: List of parameter names to copy from best_experiment_config
        best_experiment_config: The experiment config to copy parameters from
    """
    if best_experiment_config is None:
        return exp

    print("Reusing parameters from best experiment:")
    for param in params_to_reuse:
        if hasattr(best_experiment_config, param):
            setattr(exp, param, getattr(best_experiment_config, param))
            print(f"\t{param}: {getattr(best_experiment_config, param)}")

    return exp


def build_base_model(exp: ExperimentConfig):
    """Build the base model based on the experiment configuration."""
    # Select the corresponding network class
    mynet = getattr(getattr(app, exp.net_name[0]), exp.net_name[1])

    # Create the base pre-trained model
    base_model = (
        mynet(include_top=False)
        if exp.train_from_scratch
        else mynet(weights="imagenet", include_top=False)
    )

    return base_model


def add_classifier_head(base_model, exp: ExperimentConfig):
    """Build the model based on the experiment configuration."""
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if exp.classifier_head == "default":
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(NUM_CLASSES, activation=exp.last_layer_activation)(x)

    elif exp.classifier_head == "ensemble":
        # Create the classifier heads
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # First head
        x1 = Dense(512, activation="relu")(x)
        x1 = Dropout(rate=0.2)(x1)
        predictions1 = Dense(NUM_CLASSES, activation=exp.last_layer_activation)(x1)

        # Second head
        x2 = Dense(512, activation="relu")(x)
        x2 = Dropout(rate=0.2)(x2)
        predictions2 = Dense(NUM_CLASSES, activation=exp.last_layer_activation)(x2)

        # Third head
        x3 = Dense(512, activation="relu")(x)
        x3 = Dropout(rate=0.2)(x3)
        predictions3 = Dense(NUM_CLASSES, activation=exp.last_layer_activation)(x3)

        # Average the outputs
        predictions = Average()([predictions1, predictions2, predictions3])

    elif exp.classifier_head == "attention":
        # Transformer Head
        embed_dim = 256  # Embedding dimension
        num_heads = 4  # Number of attention heads
        ff_dim = 512  # Feedforward layer dimension
        num_layers = 1  # Number of Transformer encoder layers

        x = Dense(embed_dim, activation="relu")(x)

        # Expand feature vector to sequence (1, embed_dim) to match Transformer input format
        x = ExpandDims()(x)

        # Transformer Encoder Layer
        for _ in range(num_layers):
            attn_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads
            )(x, x)
            attn_output = LayerNormalization()(attn_output + x)
            ff_output = Dense(ff_dim, activation="relu")(attn_output)
            ff_output = Dense(embed_dim)(ff_output)
            ff_output = Dropout(0.2)(ff_output)
            x = LayerNormalization()(ff_output + attn_output)

        # Remove sequence dimension and project to output classes
        x = Squeeze()(x)
        predictions = Dense(NUM_CLASSES, activation=exp.last_layer_activation)(x)

    else:
        raise ValueError("Invalid classifier head type")

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def create_model(
    exp: ExperimentConfig,
):
    print(f"Defining model: {exp.title}")

    # Build the base model
    base_model = build_base_model(exp)

    # Add the classifier head
    model = add_classifier_head(base_model, exp)

    return base_model, model
