import tensorflow as tf
from keras import backend as K
from sklearn.metrics import average_precision_score
import numpy as np


def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def mean_average_precision(y_true, y_pred):
    def map_numpy(y_true_np, y_pred_np):
        return average_precision_score(y_true_np, y_pred_np).astype('float32')

    map_value = tf.numpy_function(map_numpy, [y_true, y_pred], tf.float32)
    return map_value


def subset_accuracy_metric(y_true, y_pred):
    y_pred_bin = tf.cast(tf.greater(y_pred, 0.5), tf.float32)  # Thresholding at 0.5

    exact_match = tf.reduce_all(tf.equal(y_true, y_pred_bin), axis=-1)

    return tf.reduce_mean(tf.cast(exact_match, tf.float32))
