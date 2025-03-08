import os
import tensorflow as tf
from config import *
from experiment_config import ExperimentConfig
import time
from itertools import islice
from tensorflow.keras import optimizers
from keras import backend as K
import gc
import numpy as np
import pandas as pd

from metrics import f1_metric, mean_average_precision, subset_accuracy_metric


def train_one_epoch(model, train_dataset, n_train_steps):
    """Train the model for one epoch."""
    train_loss, train_acc, train_f1, train_map, train_subset_acc = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    start_time = time.time()
    for X, Y in islice(train_dataset, n_train_steps):
        loss, acc, f1, map_score, subset_acc = model.train_on_batch(X, Y)
        train_loss += loss
        train_acc += acc
        train_f1 += f1
        train_map += map_score
        train_subset_acc += subset_acc

    elapsed = time.time() - start_time
    print(f"Time taken for training one epoch: {elapsed:.2f}s")

    return (
        train_loss / n_train_steps,
        train_acc / n_train_steps,
        train_f1 / n_train_steps,
        train_map / n_train_steps,
        train_subset_acc / n_train_steps,
    )


def test_one_epoch(model, test_dataset, n_test_steps):
    """Test the model for one epoch."""
    test_loss, test_acc, test_f1, test_map, test_subset_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    start_time = time.time()
    for X, Y in islice(test_dataset, n_test_steps):
        loss, acc, f1, map_score, subset_acc = model.evaluate(X, Y, verbose=0)
        test_loss += loss
        test_acc += acc
        test_f1 += f1
        test_map += map_score
        test_subset_acc += subset_acc

    elapsed = time.time() - start_time
    print(f"Time taken for testing one epoch: {elapsed:.2f}s")

    return (
        test_loss / n_test_steps,
        test_acc / n_test_steps,
        test_f1 / n_test_steps,
        test_map / n_test_steps,
        test_subset_acc / n_test_steps,
    )


def save_results(
    exp,
    exp_name,
    train_loss,
    train_acc,
    train_f1,
    train_map,
    train_subset_acc,
    test_loss,
    test_acc,
    test_f1,
    test_map,
    test_subset_acc,
    training_time
):
    """Save training and testing results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = RESULTS_DIR / f"{exp_name}.csv"
    
    new_row = pd.DataFrame([{
        "id": exp.id,
        "title": exp.title,
        "test_loss": test_loss,
        "test_acc": test_acc, 
        "test_f1": test_f1,
        "test_map": test_map,
        "test_subset_acc": test_subset_acc,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "train_map": train_map,
        "train_subset_acc": train_subset_acc,
        "train_time": training_time
    }])

    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        # Update existing row if ID exists, otherwise append
        df = df[df["ID"] != exp.id]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(results_file, index=False)

    print(f"Results saved to {results_file}")


def save_model(model, exp: ExperimentConfig):
    """Save the model weights to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_filename = f"{MODELS_DIR}/{exp.id:02d}-{exp.net_name[0]}.weights.h5"

    model.save_weights(model_filename)

    print(f"Model saved to {model_filename}")


def save_history(
    train_loss_history,
    train_acc_history,
    train_f1_history,
    train_map_history,
    train_subset_acc_history,
    test_loss_history,
    test_acc_history,
    test_f1_history,
    test_map_history,
    test_subset_acc_history,
    exp,
):
    """Save training and testing histories to CSV files."""
    os.makedirs(HISTORIES_DIR, exist_ok=True)

    history_files = {
        "train_loss": train_loss_history,
        "train_acc": train_acc_history,
        "train_f1": train_f1_history,
        "train_map": train_map_history,
        "train_subset_acc": train_subset_acc_history,
        "test_loss": test_loss_history,
        "test_acc": test_acc_history,
        "test_f1": test_f1_history,
        "test_map": test_map_history,
        "test_subset_acc": test_subset_acc_history,
    }

    for history_type, history_data in history_files.items():
        history_filename = (
            f"{HISTORIES_DIR}/{exp.id:02d}-{exp.net_name[0]}-{history_type}.csv"
        )
        
        df = pd.DataFrame({history_type: history_data})
        df.to_csv(history_filename, index=False)

        print(f"History saved to {history_filename}")
        
def save_predictions_to_csv(exp: ExperimentConfig, model, test_dataset, n_test_steps, test_list):
    rows = []
    global_image_index = 0  # Counter to index into test_list
    os.makedirs(LABELS_DIR, exist_ok=True)

    # Iterate over batches of the test_dataset.
    for X, Y in islice(test_dataset, n_test_steps):
        batch_size = X.shape[0]
        # Get the corresponding image names from test_list.
        image_names = test_list[global_image_index: global_image_index + batch_size]
        global_image_index += batch_size

        # Compute model predictions for the batch.
        predictions = model.predict(X)
        # For multi-label predictions, threshold the probabilities.
        pred_labels = (predictions >= 0.5).astype(int)
        Y = np.array(Y)
        n_classes = Y.shape[1]

        # Create a row for each image in the batch.
        for i in range(batch_size):
            row = {
                "id": exp.id,
                "title": exp.title,
                "image_name": image_names[i]
            }
            # Add true and predicted labels for each class.
            for cls in range(n_classes):
                row[cls] = pred_labels[i, cls]
            rows.append(row)

    # Convert the rows into a DataFrame.
    df = pd.DataFrame(rows)
    csv_path = PREDICTED_CSV

    # Append to CSV if it exists; otherwise, create a new file.
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    print(f"Predictions saved to {csv_path}")


def train_and_test(
    model,
    base_model,
    exp_name,
    exp: ExperimentConfig,
    train_dataset,
    test_dataset,
    train_list,
    test_list,
):
    n_train_steps = len(train_list) // exp.batch_size
    n_test_steps = len(test_list) // exp.batch_size
    warmup_epochs = 3  # Number of epochs to keep the base model frozen

    (
        train_loss_history,
        train_acc_history,
        train_f1_history,
        train_map_history,
        train_subset_acc_history,
    ) = ([], [], [], [], [])
    (
        test_loss_history,
        test_acc_history,
        test_f1_history,
        test_map_history,
        test_subset_acc_history,
    ) = ([], [], [], [], [])

    # Define optimizers
    warmup_optimizer = optimizers.RMSprop(learning_rate=exp.learning_rate * 0.1)
    opt_rms = optimizers.RMSprop(learning_rate=exp.learning_rate)

    # Track previous optimizer to avoid redundant compilation
    prev_optimizer = None

    print(f"In training loop: {exp.title}")
    training_start_time = time.time()

    for epoch in range(exp.n_epochs):
        # Determine optimizer
        if exp.warm_up and epoch < warmup_epochs:
            optimizer = warmup_optimizer
        else:
            optimizer = opt_rms

        # Freeze base model at the start of warmup
        if exp.warm_up and epoch == 0:
            print("Freezing base model layers for warmup.")
            for layer in base_model.layers:
                layer.trainable = False

        # Unfreeze base model after warmup period
        if exp.warm_up and epoch == warmup_epochs:
            print(f"Unfreezing base model at epoch {epoch}")
            for layer in base_model.layers:
                layer.trainable = True  # Unfreeze layers
            should_recompile = True  # Mark for recompilation

        else:
            should_recompile = (
                optimizer != prev_optimizer
            )  # Recompile only if optimizer changes

        # Recompile model only if necessary
        if should_recompile:
            print(f"Recompiling model at epoch {epoch} (Optimizer changed)")
            model.compile(
                loss=exp.loss,
                optimizer=optimizer,
                metrics=[
                    "AUC",
                    f1_metric,
                    mean_average_precision,
                    subset_accuracy_metric,
                ],
            )
            prev_optimizer = optimizer  # Update previous optimizer

        # Train one epoch
        train_loss, train_acc, train_f1, train_map, train_subset_acc = train_one_epoch(
            model, train_dataset, n_train_steps
        )
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        train_map_history.append(train_map)
        train_f1_history.append(train_f1)
        train_subset_acc_history.append(train_subset_acc)

        print(
            f"Epoch {epoch} training loss: {train_loss:.2f}, acc: {train_acc:.2f}, "
            f"f1: {train_f1:.2f}, mAP: {train_map:.2f}"
        )

        # Test one epoch
        test_loss, test_acc, test_f1, test_map, test_subset_acc = test_one_epoch(
            model, test_dataset, n_test_steps
        )
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        test_f1_history.append(test_f1)
        test_map_history.append(test_map)
        test_subset_acc_history.append(test_subset_acc)

        print(
            f"Epoch {epoch} test loss: {test_loss:.2f}, acc: {test_acc:.2f}, "
            f"f1: {test_f1:.2f}, mAP: {test_map:.2f}"
        )

    training_time = time.time() - training_start_time
    print(f"Training ({exp.title}) finished in: {training_time:.2f} seconds")

    # Save final results
    save_results(
        exp,
        exp_name,
        train_loss_history[-1],
        train_acc_history[-1],
        train_f1_history[-1],
        train_map_history[-1],
        train_subset_acc_history[-1],
        test_loss_history[-1],
        test_acc_history[-1],
        test_f1_history[-1],
        test_map_history[-1],
        test_subset_acc_history[-1],
        training_time
    )

    # Save model weights
    # save_model(model, exp)

    # Save training history
    save_history(
        train_loss_history,
        train_acc_history,
        train_f1_history,
        train_map_history,
        train_subset_acc_history,
        test_loss_history,
        test_acc_history,
        test_f1_history,
        test_map_history,
        test_subset_acc_history,
        exp,
    )
    
    # Save predictions to CSV
    save_predictions_to_csv(exp, model, test_dataset, n_test_steps, test_list)

    # Clear memory
    del model
    K.clear_session()
    gc.collect()

    memory_info = tf.config.experimental.get_memory_info("GPU:0")
    print("Current memory usage (bytes):", memory_info["current"])
    print("Peak memory usage (bytes):", memory_info["peak"])
