import tensorflow as tf
from keras import backend as K

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

def average_precision(y_true, y_pred):
    sorted_indices = tf.argsort(y_pred, axis=-1, direction='DESCENDING') 
    sorted_true = tf.gather(y_true, sorted_indices, batch_dims=1) 

    cumulative_true = tf.cumsum(sorted_true, axis=-1)
    cumulative_total = tf.range(1, tf.shape(sorted_true)[-1] + 1, dtype=tf.float32)  

    precision_at_k = cumulative_true / cumulative_total
    AP = tf.reduce_sum(precision_at_k * sorted_true, axis=-1) / (tf.reduce_sum(sorted_true, axis=-1) + K.epsilon())

    return AP

def mean_average_precision(y_true, y_pred):
    AP_per_class = average_precision(y_true, y_pred)
    return tf.reduce_mean(AP_per_class)  

def subset_accuracy_metric(y_true, y_pred):
    y_pred_bin = tf.cast(tf.greater(y_pred, 0.5), tf.float32)  # Thresholding at 0.5
    
    exact_match = tf.reduce_all(tf.equal(y_true, y_pred_bin), axis=-1)
    
    return tf.reduce_mean(tf.cast(exact_match, tf.float32))

