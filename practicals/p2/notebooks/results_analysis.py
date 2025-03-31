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
        vmin=0,
        vmax=1,
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Predicted", fontsize=12, weight="bold")
    ax.set_ylabel("True", fontsize=12, weight="bold")
# %%
# plot confusion matrices
experiment_ids = [24]
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

    ax[0].add_patch(plt.Rectangle((15.95, 15.95), 12, 12, fill=False, color="red", linewidth=4))
    ax[0].text(19.5, 17.5, "Worst 12 Classes", fontsize=14, weight="bold", color="red")
    plt.savefig(FIGURES_DIR / f"confusion_matrices_annotated_experiment_{experiment_id}.png", dpi=300)

    plt.show()

# %%
# compare confusion matrices before and after fine tuning
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
worst_12_labels = tp_sorted_labels[0:1] + tp_sorted_labels[-12:]

worst_12_indices = np.argwhere(np.isin(tp_sorted_labels, worst_12_labels)).flatten()
subset_cm = tp_sorted_cm.iloc[worst_12_indices, worst_12_indices]

fig, axs = plt.subplots(1, 2, figsize=(24, 12), width_ratios=[1, 1.25])
plot_confusion_matrix(axs[0], subset_cm, worst_12_labels, "Before Fine Tuning", cbar=False)
axs[0].add_patch(plt.Rectangle((0.95, 0.95), 12, 12, fill=False, color="red", linewidth=3))
# TODO: replace this with the confusion matrix after fine tuning
plot_confusion_matrix(axs[1], subset_cm, worst_12_labels, "After Fine Tuning", cbar=True)

plt.suptitle("Confusion Matrices of Worst 12 Classes", fontsize=20, weight="bold")
plt.tight_layout()

# %%
# define plot_metrics
def plot_metrics(metrics_df: pd.DataFrame, metrics: List[Tuple[str, str, str, str]], title: str, save_path: Path):
    fig, axs = plt.subplots(1, len(metrics), figsize=(8 * len(metrics), 6))
    epochs = metrics_df.index
    for ax, (val_metric_name, train_metric_name, metric_label, metric_color) in zip(axs, metrics):
        ax.plot(epochs, metrics_df[val_metric_name], marker='o', label='Validation', color=metric_color)
        ax.plot(epochs, metrics_df[train_metric_name], marker='o', label='Training', color=metric_color, linestyle='--')

        ax.set_ylabel(metric_label, fontsize=12, weight="bold")
        ax.set_xlabel('Epoch', fontsize=12, weight="bold")
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid()

    plt.suptitle(title, fontsize=20, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# %%
# plot the training curves
experiment_ids = [24]
for experiment_id in experiment_ids:
    metrics_df = metrics_dfs[f"experiment_{experiment_id}"]
    metrics: List[Tuple[str, str, str, str]] = [
        ("val_accuracy", "train_accuracy", "Accuracy", "#14A3A1"),
        ("val_dice", "train_dice", "Dice Score", "#CD5334"),
        ("val_loss", "train_loss", "Loss", "#4059AD"),
    ]
    plot_metrics(metrics_df, metrics, f"Training Curves", FIGURES_DIR / f"validation_metrics_experiment_{experiment_id}.png")


# %%
# TODO
# 2. a 4 column plot that shows the mDice performance for each experiment part of hyperparam search