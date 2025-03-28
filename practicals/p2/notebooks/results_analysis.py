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


# %%
# normalize confusion matrix

cm = confusion_matrix_dfs["experiment_11"]
# normalize the predictions by the positive counts
positive_counts = cm.sum(axis=1)
cm = cm.div(positive_counts, axis=0)

# %%
# sort by the positive counts
sorted_indices = positive_counts.sort_values(ascending=False).index
positive_sorted_cm = cm.iloc[sorted_indices, sorted_indices]
positive_sorted_labels = [labels[i] for i in sorted_indices]
# %%
# sort by the true positive counts
tp_counts = cm.values.diagonal()
tp_sorted_indices = tp_counts.argsort()[::-1]
tp_sorted_cm = cm.iloc[tp_sorted_indices, tp_sorted_indices]
tp_sorted_labels = [labels[i] for i in tp_sorted_indices]

# %%
# plot both confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(30, 14), width_ratios=[1, 1.25])
sns.heatmap(
    tp_sorted_cm,
    cmap="Blues",
    fmt="d",
    ax=ax[0],
    xticklabels=tp_sorted_labels,
    yticklabels=tp_sorted_labels,
    linewidths=0.5,
    linecolor="lightgrey",
    cbar=False,
)
ax[0].set_title("True Positive-Sorted Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("True")

sns.heatmap(
    positive_sorted_cm,
    cmap="Blues",
    fmt="d",
    ax=ax[1],
    xticklabels=positive_sorted_labels,
    yticklabels=positive_sorted_labels,
    linewidths=0.5,
    linecolor="lightgrey",
)
ax[1].set_title("Positive-Sorted Confusion Matrix")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("True")

plt.suptitle("Confusion Matrices, normalized by positive counts", fontsize=20)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "confusion_matrices_experiment_11.png", dpi=300)
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
