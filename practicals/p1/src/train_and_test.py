import random
import csv
import os
import tensorflow as tf
from config import *
from load_data import get_file_paths, get_dataset_from_paths
from augmentation import create_data_pipeline
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
):
    """Save training and testing results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = RESULTS_DIR / f"{exp_name}.csv"
    final_results = [
        exp.id,
        test_loss,
        test_acc,
        test_f1,
        test_map,
        test_subset_acc,
        train_loss,
        train_acc,
        train_f1,
        train_map,
        train_subset_acc,
    ]

    file_exists = os.path.exists(results_file)
    updated_rows = []
    found = False

    if file_exists:
        with open(results_file, mode="r") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            for row in reader:
                if int(row[0]) == int(exp.id):
                    updated_rows.append(final_results)
                    found = True
                else:
                    updated_rows.append(row)

    if not found:
        updated_rows.append(final_results)

    with open(results_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ID",
                "Test Loss",
                "Test Accuracy",
                "Test F1",
                "Test mAP",
                "Test Subset Acc",
                "Train Loss",
                "Train Accuracy",
                "Train F1",
                "Train mAP",
                "Train Subset Acc",
            ]
        )
        writer.writerows(updated_rows)

    print(f"Results saved to {results_file}")


def save_model(model, exp: ExperimentConfig):
    """Save the model weights to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_filename = f"{MODELS_DIR}/{exp.net_name[0]}-{exp.id}.weights.h5"

    # start_time = time.time()
    model.save_weights(model_filename)
    # end_time = time.time()

    print(f"Model saved to {model_filename}")
    # print(f"Time taken to save model: {end_time - start_time:.2f}s")


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
            f"{HISTORIES_DIR}/{exp.net_name[0]}-{exp.id}-{history_type}.csv"
        )

        with open(history_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([history_type])
            for value in history_data:
                writer.writerow([value])

        print(f"History saved to {history_filename}")
        

def save_predictions_to_csv(experiment, model, test_dataset, n_test_steps, test_list):
    rows = []
    global_image_index = 0  # Counter to index into test_list

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
                "experiment_id": experiment.title,
                "image_name": image_names[i]
            }
            # Add true and predicted labels for each class.
            for cls in range(n_classes):
                row[f"true_{cls}"] = Y[i, cls]
                row[f"pred_{cls}"] = pred_labels[i, cls]
            rows.append(row)

    # Convert the rows into a DataFrame.
    df = pd.DataFrame(rows)
    csv_path = Path(RESULTS_DIR) / "label_predictions.csv"

    # Append to CSV if it exists; otherwise, create a new file.
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    print(f"Predictions saved to {csv_path}")


def train_and_test(
    model,
    exp_name,
    exp: ExperimentConfig,
    train_dataset,
    test_dataset,
    train_list,
    test_list,
):
    n_train_steps = len(train_list) // exp.batch_size
    n_test_steps = len(test_list) // exp.batch_size
    warmup_epochs = 2  # Number of epochs to keep the base model frozen

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

    print(f"In training loop: {exp.title}")
    start_time = time.time()

    for epoch in range(exp.n_epochs):
        random.shuffle(train_list)

        # Set the optimizer and freeze/unfreeze model layers
        if exp.warm_up and epoch < warmup_epochs:
            optimizer = (
                warmup_optimizer  # Use warmup_optimizer during the warmup period
            )

            # Freeze the base model during warmup (only once at the beginning)
            if epoch == 0:  # Freeze only at the start of the warmup phase
                model.layers[0].trainable = False

        # Unfreeze base model after warmup period
        if exp.warm_up and epoch == warmup_epochs:
            print(f"Unfreezing base model at epoch {epoch}")
            model.layers[0].trainable = True  # Unfreeze the base model

        else:
            optimizer = opt_rms  # Use normal optimizer after the warmup period

        # Recompile the model if the optimizer changes
        model.compile(
            loss=exp.loss,
            optimizer=optimizer,
            metrics=["AUC", f1_metric, mean_average_precision, subset_accuracy_metric],
        )

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

    elapsed_time = time.time() - start_time
    print(f"Training ({exp.title}) finished in: {elapsed_time:.2f} seconds")

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
    )
    
    # Save predicted and true labels
    save_predictions_to_csv(exp, model, test_dataset, n_test_steps, test_list)

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

    # Clear memory
    del model
    K.clear_session()
    gc.collect()

    memory_info = tf.config.experimental.get_memory_info("GPU:0")
    print("Current memory usage (bytes):", memory_info["current"])
    print("Peak memory usage (bytes):", memory_info["peak"])
