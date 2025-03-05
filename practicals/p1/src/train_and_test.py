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


def train_one_epoch(model, train_dataset, n_train_steps):
    """Train the model for one epoch."""
    train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
    start_time = time.time()
    for X, Y in islice(train_dataset, n_train_steps):
        loss, acc, f1 = model.train_on_batch(X, Y)
        train_loss += loss
        train_acc += acc
        train_f1 += f1

    elapsed = time.time() - start_time
    print(f"Time taken for training one epoch: {elapsed:.2f}s")

    return (
        train_loss / n_train_steps,
        train_acc / n_train_steps,
        train_f1 / n_train_steps,
    )


def test_one_epoch(model, test_dataset, n_test_steps):
    """Test the model for one epoch."""
    test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0
    start_time = time.time()
    for X, Y in islice(test_dataset, n_test_steps):
        loss, acc, f1 = model.evaluate(X, Y, verbose=0)
        test_loss += loss
        test_acc += acc
        test_f1 += f1

    elapsed = time.time() - start_time
    print(f"Time taken for testing one epoch: {elapsed:.2f}s")

    return test_loss / n_test_steps, test_acc / n_test_steps, test_f1 / n_test_steps


def save_results(exp_name, exp, train_loss, train_acc, train_f1, test_loss, test_acc, test_f1):
    """Save training and testing results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = RESULTS_DIR / f"{exp_name}.csv"
    final_results = [
        exp.id,
        test_loss,
        test_acc,
        test_f1,
        test_acc,
        train_loss,
        train_acc,
        train_f1,
        train_acc,
    ]

    file_exists = os.path.exists(results_file)
    updated_rows = []
    found = False

    # start_time = time.time()
    if file_exists:
        with open(results_file, mode="r") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            for row in reader:
                # print(row)
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
                "Test AUC",
                "Train Loss",
                "Train Accuracy",
                "Train F1",
                "Train AUC",
            ]
        )
        writer.writerows(updated_rows)
    # print(f"Time taken for writing results: {time.time() - start_time:.2f}s")

    print(f"Results saved to {results_file}")


def save_model(model, exp: ExperimentConfig):
    """Save the model weights to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_filename = f'{MODELS_DIR}/{exp.net_name[0]}-{exp.id}.weights.h5'

    # start_time = time.time()
    model.save_weights(model_filename)
    # end_time = time.time()

    print(f"Model saved to {model_filename}")
    # print(f"Time taken to save model: {end_time - start_time:.2f}s")


def save_history(
    train_loss_history,
    train_acc_history,
    train_f1_history,
    test_loss_history,
    test_acc_history,
    test_f1_history,
    exp: ExperimentConfig,
):
    """Save training and testing histories to CSV files."""
    os.makedirs(HISTORIES_DIR, exist_ok=True)

    history_files = {
        "train_loss": train_loss_history,
        "train_acc": train_acc_history,
        "train_f1": train_f1_history,
        "test_loss": test_loss_history,
        "test_acc": test_acc_history,
        "test_f1": test_f1_history,
    }

    for history_type, history_data in history_files.items():
        history_filename = (
            f'{HISTORIES_DIR}/{exp.net_name[0]}-{exp.id}-{history_type}.csv'
        )

        # start_time = time.time()
        with open(history_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([history_type])
            for value in history_data:
                writer.writerow([value])
        # end_time = time.time()

        print(f"History saved to {history_filename}")
        # print(f"Time taken to save {history_type} history: {end_time - start_time:.2f}s")


def train_and_test(model, exp_name, exp: ExperimentConfig, train_dataset, test_dataset, train_list, test_list):
    n_train_steps = 2
    n_test_steps = 2

    train_loss_history, train_acc_history, train_f1_history = [], [], []
    test_loss_history, test_acc_history, test_f1_history = [], [], []

    print(f"In training loop: {exp.title}")
    start_time = time.time()
    for epoch in range(exp.n_epochs):
        random.shuffle(train_list)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_dataset, n_train_steps
        )
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        train_f1_history.append(train_f1)

        print(
            f"Epoch {epoch} training loss: {train_loss:.2f}, acc: {train_acc:.2f}, f1: {train_f1:.2f}"
        )

        test_loss, test_acc, test_f1 = test_one_epoch(model, test_dataset, n_test_steps)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        test_f1_history.append(test_f1)

        print(
            f"Epoch {epoch} test loss: {test_loss:.2f}, acc: {test_acc:.2f}, f1: {test_f1:.2f}"
        )

    elapsed_time = time.time() - start_time
    print(f"Training ({exp.title}) finished in: {elapsed_time}")

    save_results(
        exp_name,
        exp,
        train_loss_history[-1],
        train_acc_history[-1],
        train_f1_history[-1],
        test_loss_history[-1],
        test_acc_history[-1],
        test_f1_history[-1],
    )

    save_model(model, exp)

    save_history(
        train_loss_history,
        train_acc_history,
        train_f1_history,
        test_loss_history,
        test_acc_history,
        test_f1_history,
        exp,
    )
