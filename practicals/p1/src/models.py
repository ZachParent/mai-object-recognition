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

tf.config.run_functions_eagerly(True)
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
    exp_name: str,
    best_experiment_config: ExperimentConfig = None,
):
    """Setup the best configuration based on previous experiments."""
    if exp_name == "model-experiments":
        pass

    elif exp_name == "hyperparameter-experiments":
        print(
            f"Reusing from best model experiment:\n"
            f"\tnet_name: {best_experiment_config.net_name}, "
            f"\ttrain_from_scratch: {best_experiment_config.train_from_scratch}, "
            f"\twarm_up: {best_experiment_config.warm_up}"
        )
        exp.net_name = best_experiment_config.net_name
        exp.train_from_scratch = best_experiment_config.train_from_scratch
        exp.warm_up = best_experiment_config.warm_up

    elif exp_name == "augmentation-experiments":
        print(
            f"Reusing from best hyperparameter experiment:\n"
            f"\tnet_name: {best_experiment_config.net_name},\n"
            f"\ttrain_from_scratch: {best_experiment_config.train_from_scratch},\n"
            f"\twarm_up: {best_experiment_config.warm_up},\n"
            f"\tbatch_size: {best_experiment_config.batch_size},\n"
            f"\tlearning_rate: {best_experiment_config.learning_rate},\n"
            f"\tloss: {best_experiment_config.loss},\n"
            f"\tlast_layer_activation: {best_experiment_config.last_layer_activation}"
        )
        exp.net_name = best_experiment_config.net_name
        exp.train_from_scratch = best_experiment_config.train_from_scratch
        exp.warm_up = best_experiment_config.warm_up
        exp.batch_size = best_experiment_config.batch_size
        exp.learning_rate = best_experiment_config.learning_rate
        exp.loss = best_experiment_config.loss
        exp.last_layer_activation = best_experiment_config.last_layer_activation

    elif exp_name == "classfier_head-experiments":
        print(
            f"Reusing from best hyperparameter experiment:\n"
            f"\tnet_name: {best_experiment_config.net_name},\n"
            f"\ttrain_from_scratch: {best_experiment_config.train_from_scratch},\n"
            f"\twarm_up: {best_experiment_config.warm_up},\n"
            f"\tbatch_size: {best_experiment_config.batch_size},\n"
            f"\tlearning_rate: {best_experiment_config.learning_rate},\n"
            f"\tloss: {best_experiment_config.loss},\n"
            f"\tlast_layer_activation: {best_experiment_config.last_layer_activation}"
        )
        exp.net_name = best_experiment_config.net_name
        exp.train_from_scratch = best_experiment_config.train_from_scratch
        exp.warm_up = best_experiment_config.warm_up
        exp.batch_size = best_experiment_config.batch_size
        exp.learning_rate = best_experiment_config.learning_rate
        exp.loss = best_experiment_config.loss
        exp.last_layer_activation = best_experiment_config.last_layer_activation

    return


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
        predictions = Dense(num_classes, activation=exp.last_layer_activation)(x)

    elif exp.classifier_head == "ensemble":
        # Create the classifier heads
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # First head
        x1 = Dense(512, activation="relu")(x)
        x1 = Dropout(rate=0.2)(x1)
        predictions1 = Dense(num_classes, activation=exp.last_layer_activation)(x1)

        # Second head
        x2 = Dense(512, activation="relu")(x)
        x2 = Dropout(rate=0.2)(x2)
        predictions2 = Dense(num_classes, activation=exp.last_layer_activation)(x2)

        # Third head
        x3 = Dense(512, activation="relu")(x)
        x3 = Dropout(rate=0.2)(x3)
        predictions3 = Dense(num_classes, activation=exp.last_layer_activation)(x3)

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
        predictions = Dense(num_classes, activation=exp.last_layer_activation)(x)

    else:
        raise ValueError("Invalid classifier head type")

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def create_model(
    exp: ExperimentConfig,
    exp_name: str,
    best_experiment_config: ExperimentConfig = None,
):
    print(f"Defining model: {exp.title}")

    setup_best_config(exp, exp_name, best_experiment_config)

    # Build the base model
    base_model = build_base_model(exp)

    # Add the classifier head
    model = add_classifier_head(base_model, exp)

    return base_model, model
