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


def mean_average_precision(y_true, y_pred):
    """
    Compute mean average precision (mAP) for a batch of predictions.
    
    Both y_true and y_pred are expected to be TensorFlow symbolic tensors 
    of shape [batch_size, num_predictions]. For each sample, y_true should 
    contain binary values (1 for relevant, 0 for non-relevant) and y_pred 
    contains the predicted scores.
    
    Returns:
        A scalar Tensor representing the mean average precision over the batch.
    """
    # Sort the predictions in descending order and obtain sorted indices.
    sorted_indices = tf.argsort(y_pred, axis=-1, direction='DESCENDING')
    # Rearrange y_true according to the sorted indices.
    sorted_y_true = tf.gather(y_true, sorted_indices, batch_dims=1)
    
    # Compute the cumulative sum of the sorted y_true values.
    cumsum = tf.cumsum(sorted_y_true, axis=-1)
    
    # Create a tensor of rank positions (starting at 1).
    ranks = tf.cast(tf.range(1, tf.shape(y_true)[1] + 1), tf.float32)
    
    # Compute precision at each rank: precision = (# of relevant up to rank k) / k.
    precision_at_k = cumsum / ranks
    
    # Only consider positions with relevant documents.
    precision_at_hits = precision_at_k * sorted_y_true
    
    # Sum the precision scores for each sample.
    sum_precisions = tf.reduce_sum(precision_at_hits, axis=-1)
    
    # Count of relevant items per sample.
    relevant_counts = tf.reduce_sum(sorted_y_true, axis=-1)
    
    # For samples with no relevant items, define average precision as 0.
    average_precision = tf.where(
        tf.equal(relevant_counts, 0),
        tf.zeros_like(sum_precisions),
        sum_precisions / relevant_counts
    )
    
    # Mean average precision over the batch.
    mean_ap = tf.reduce_mean(average_precision)
    return mean_ap


def subset_accuracy_metric(y_true, y_pred):
    y_pred_bin = tf.cast(tf.greater(y_pred, 0.5), tf.float32)  # Thresholding at 0.5

    exact_match = tf.reduce_all(tf.equal(y_true, y_pred_bin), axis=-1)

    return tf.reduce_mean(tf.cast(exact_match, tf.float32))
