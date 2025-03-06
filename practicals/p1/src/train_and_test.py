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
    train_loss, train_acc, train_f1, train_map = 0.0, 0.0, 0.0, 0.0
    start_time = time.time()
    for X, Y in islice(train_dataset, n_train_steps):
        loss, acc, f1, map_score = model.train_on_batch(X, Y)
        train_loss += loss
        train_acc += acc
        train_f1 += f1
        train_map += map_score

    elapsed = time.time() - start_time
    print(f"Time taken for training one epoch: {elapsed:.2f}s")

    return (
        train_loss / n_train_steps,
        train_acc / n_train_steps,
        train_f1 / n_train_steps,
        train_map / n_train_steps,
    )


def test_one_epoch(model, test_dataset, n_test_steps):
    """Test the model for one epoch."""
    test_loss, test_acc, test_f1, test_map = 0.0, 0.0, 0.0, 0.0
    start_time = time.time()
    for X, Y in islice(test_dataset, n_test_steps):
        loss, acc, f1, map_score = model.evaluate(X, Y, verbose=0)
        test_loss += loss
        test_acc += acc
        test_f1 += f1
        test_map += map_score

    elapsed = time.time() - start_time
    print(f"Time taken for testing one epoch: {elapsed:.2f}s")

    return test_loss / n_test_steps, test_acc / n_test_steps, test_f1 / n_test_steps, test_map/n_test_steps


def save_results(exp, train_loss, train_acc, train_f1, train_map, test_loss, test_acc, test_f1, test_map):
    """Save training and testing results to CSV."""
    results_file = f"{RESULTS_DIR}/model-experiments.csv"
    final_results = [
        exp.id,
        test_loss,
        test_acc,
        test_f1,
        test_map,  
        train_loss,
        train_acc,
        train_f1,
        train_map,  
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
                "Train Loss",
                "Train Accuracy",
                "Train F1",
                "Train mAP", 
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
    test_loss_history,
    test_acc_history,
    test_f1_history,
    test_map_history,  
    exp,
):
    """Save training and testing histories to CSV files."""
    os.makedirs(HISTORIES_DIR, exist_ok=True)

    history_files = {
        "train_loss": train_loss_history,
        "train_acc": train_acc_history,
        "train_f1": train_f1_history,
        "train_map": train_map_history,  
        "test_loss": test_loss_history,
        "test_acc": test_acc_history,
        "test_f1": test_f1_history,
        "test_map": test_map_history,  
    }

    for history_type, history_data in history_files.items():
        history_filename = (
            f'{HISTORIES_DIR}/{exp.net_name[0]}-{exp.id}-{history_type}.csv'
        )

        with open(history_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([history_type])
            for value in history_data:
                writer.writerow([value])

        print(f"History saved to {history_filename}")

def train_and_test(
    model,
    exp_name,
    exp: ExperimentConfig,
    train_dataset,
    test_dataset,
    train_list,
    test_list,
):
    n_train_steps = 10  # len(train_list) // exp.batch_size  # TODO remove 10
    n_test_steps = 10   # len(test_list) // exp.batch_size  # TODO remove 10

    train_loss_history, train_acc_history, train_f1_history, train_map_history = [], [], [], []
    test_loss_history, test_acc_history, test_f1_history, test_map_history = [], [], [], []

    print(f"In training loop: {exp.title}")
    start_time = time.time()

    for epoch in range(exp.n_epochs):
        random.shuffle(train_list)

        # Train one epoch
        train_loss, train_acc, train_f1, train_map = train_one_epoch(
            model, train_dataset, n_train_steps
        )
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        train_f1_history.append(train_f1)
        train_map_history.append(train_map)  

        print(
            f"Epoch {epoch} training loss: {train_loss:.2f}, acc: {train_acc:.2f}, "
            f"f1: {train_f1:.2f}, mAP: {train_map:.2f}"
        )

        # Test one epoch
        test_loss, test_acc, test_f1, test_map = test_one_epoch(
            model, test_dataset, n_test_steps
        )
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        test_f1_history.append(test_f1)
        test_map_history.append(test_map) 

        print(
            f"Epoch {epoch} test loss: {test_loss:.2f}, acc: {test_acc:.2f}, "
            f"f1: {test_f1:.2f}, mAP: {test_map:.2f}"
        )

    elapsed_time = time.time() - start_time
    print(f"Training ({exp.title}) finished in: {elapsed_time:.2f} seconds")

    # Save final results
    save_results(
        exp,
        train_loss_history[-1],
        train_acc_history[-1],
        train_f1_history[-1],
        train_map_history[-1],  
        test_loss_history[-1],
        test_acc_history[-1],
        test_f1_history[-1],
        test_map_history[-1], 
    )

    # Save model weights
    save_model(model, exp)

    # Save training history
    save_history(
        train_loss_history,
        train_acc_history,
        train_f1_history,
        train_map_history,  
        test_loss_history,
        test_acc_history,
        test_f1_history,
        test_map_history,  
        exp,
    )
