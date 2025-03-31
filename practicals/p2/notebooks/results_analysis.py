# %%
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from typing import List, Tuple
from pathlib import Path

plt.style.use("default")
sys.path.append("..")
sys.path.append("../src")

from src.config import METRICS_DIR, DATA_DIR, FIGURES_DIR
from src.dataset import MAIN_ITEM_NAMES

CONFUSION_MATRICES_DIR = DATA_DIR / "03_confusion_matrices"
# %%
# load data
labels = ["background"] + MAIN_ITEM_NAMES
metrics_dfs: dict[str, pd.DataFrame] = {}
for path in METRICS_DIR.glob("experiment_*.csv"):
    metrics_dfs[path.stem] = pd.read_csv(path, index_col=0)

confusion_matrix_dfs: dict[str, pd.DataFrame] = {}
for path in CONFUSION_MATRICES_DIR.glob("experiment_*.csv"):
    confusion_matrix_dfs[path.stem] = pd.read_csv(path, index_col=0)

def plot_confusion_matrix(ax: plt.Axes, cm: pd.DataFrame, labels: List[str], title: str, cbar: bool = False):
    sns.heatmap(
        cm,
        cmap="Blues",
        fmt="d",
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="lightgrey",
        cbar=cbar,
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Predicted", fontsize=12, weight="bold")
    ax.set_ylabel("True", fontsize=12, weight="bold")
# %%
# plot confusion matrices
experiment_ids = [11, 24]
for experiment_id in experiment_ids:
    cm = confusion_matrix_dfs[f"experiment_{experiment_id}"]
    # normalize the predictions by the positive counts
    positive_counts = cm.sum(axis=1)
    cm = cm.div(positive_counts, axis=0)

    # sort by the positive counts
    sorted_indices = positive_counts.sort_values(ascending=False).index
    positive_sorted_cm = cm.iloc[sorted_indices, sorted_indices]
    positive_sorted_labels = [labels[i] for i in sorted_indices]
    # sort by the true positive counts
    tp_counts = cm.values.diagonal()
    tp_sorted_indices = tp_counts.argsort()[::-1]
    tp_sorted_cm = cm.iloc[tp_sorted_indices, tp_sorted_indices]
    tp_sorted_labels = [labels[i] for i in tp_sorted_indices]

    # plot both confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(30, 14), width_ratios=[1, 1.25])
    plot_confusion_matrix(ax[0], tp_sorted_cm, tp_sorted_labels, "Sorted by True Positives", cbar=False)
    plot_confusion_matrix(ax[1], positive_sorted_cm, positive_sorted_labels, "Sorted by Ground Truth Positives", cbar=True)
    plt.suptitle("Confusion Matrices, Normalized by Ground Truth Positives", fontsize=20, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrices_experiment_{experiment_id}.png", dpi=300)

    plt.show()

# %%
# plot the metrics
def plot_metrics(metrics_df: pd.DataFrame, metrics: List[Tuple[str, str, str]], title: str, save_path: Path):
    plt.figure(figsize=(12, 6))
    epochs = metrics_df.index
    for metric_name, metric_label, metric_color in metrics:
        plt.plot(epochs, metrics_df[metric_name], label=metric_label, marker='o', color=metric_color)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.xticks(epochs)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.show()

# %%
metrics_df = metrics_dfs["experiment_11"]
metrics: List[Tuple[str, str, str]] = [
    ("val_f1", "Validation F1 Score", "#FF0000"),
    ("val_dice", "Validation Dice Score", "#FF00FF"),
    ("val_loss", "Validation Loss", "#0000FF"),
]
plot_metrics(metrics_df, metrics, "Validation Metrics During Training", FIGURES_DIR / "validation_metrics_experiment_11.png")

# %%
# compare before and after fine tuning
experiment_id = 24
cm = confusion_matrix_dfs[f"experiment_{experiment_id}"]
# normalize the predictions by the positive counts
positive_counts = cm.sum(axis=1)
cm = cm.div(positive_counts, axis=0)

# sort by the true positive counts
tp_counts = cm.values.diagonal()
tp_sorted_indices = tp_counts.argsort()[::-1]
tp_sorted_cm = cm.iloc[tp_sorted_indices, tp_sorted_indices]
tp_sorted_labels = [labels[i] for i in tp_sorted_indices]
worst_12_labels = tp_sorted_labels[-12:]

worst_12_labels

# TODO: replace this with the confusion matrix after fine tuning
subset_cm = cm.iloc[-12:, -12:]
subset_cm

# %%
# TODO
# 1. a confusion matrix, then the subset confusion matrix after fine tuning
# 2. a 4 column plot that shows the mDice performance for each experiment part of hyperparam search