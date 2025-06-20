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

from src.experiment_config import MODELS
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


def plot_confusion_matrix(
    ax: plt.Axes, cm: pd.DataFrame, labels: List[str], title: str, cbar: bool = False
):
    sns.heatmap(
        cm,
        cmap="PuBuGn",
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
    plot_confusion_matrix(
        ax[0], tp_sorted_cm, tp_sorted_labels, "Sorted by True Positives", cbar=False
    )
    plot_confusion_matrix(
        ax[1],
        positive_sorted_cm,
        positive_sorted_labels,
        "Sorted by Ground Truth Positives",
        cbar=True,
    )
    plt.suptitle(
        "Confusion Matrices, Normalized by Ground Truth Positives",
        fontsize=20,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"confusion_matrices_experiment_{experiment_id}.png", dpi=300
    )

    ax[0].add_patch(
        plt.Rectangle((15.95, 15.95), 12, 12, fill=False, color="#CD5334", linewidth=4)
    )
    ax[0].text(
        19.5, 17.5, "Worst 12 Classes", fontsize=14, weight="bold", color="#CD5334"
    )
    plt.savefig(
        FIGURES_DIR / f"confusion_matrices_annotated_experiment_{experiment_id}.png",
        dpi=300,
    )

    plt.show()

# %%
# compare confusion matrices before and after fine tuning
non_fine_tuned_cm = confusion_matrix_dfs[f"experiment_{24}"]
fine_tuned_cm = confusion_matrix_dfs[f"experiment_{25}"]
fine_tuned_cm.columns = [
    "background",
    "cardigan",
    "vest",
    "jumpsuit",
    "cape",
    "headband, head covering, hair accessory",
    "tie",
    "glove",
    "watch",
    "belt",
    "leg warmer",
    "sock",
    "scarf",
]
fine_tuned_cm.index = fine_tuned_cm.columns
# %%
# normalize the predictions by the positive counts
positive_counts = non_fine_tuned_cm.sum(axis=1)
non_fine_tuned_cm = non_fine_tuned_cm.div(positive_counts, axis=0)
positive_counts = fine_tuned_cm.sum(axis=1)
fine_tuned_cm = fine_tuned_cm.div(positive_counts, axis=0)

# sort by the true positive counts
tp_counts = non_fine_tuned_cm.values.diagonal()
tp_sorted_indices = tp_counts.argsort()[::-1]
tp_sorted_cm = non_fine_tuned_cm.iloc[tp_sorted_indices, tp_sorted_indices]
tp_sorted_labels = [labels[i] for i in tp_sorted_indices]
worst_12_labels = tp_sorted_labels[0:1] + tp_sorted_labels[-12:]

worst_12_indices = np.argwhere(np.isin(tp_sorted_labels, worst_12_labels)).flatten()
subset_cm = tp_sorted_cm.iloc[worst_12_indices, worst_12_indices]
fine_tuned_cm = fine_tuned_cm.loc[worst_12_labels, worst_12_labels]

fig, axs = plt.subplots(1, 2, figsize=(24, 11), width_ratios=[1, 1.25])
plot_confusion_matrix(
    axs[0], subset_cm, worst_12_labels, "Before Fine Tuning", cbar=False
)
axs[0].set_xticks(np.array(range(len(worst_12_labels))) + 0.5)
axs[0].set_xticklabels(worst_12_labels, rotation=45, ha="right")

axs[0].add_patch(
    plt.Rectangle((0.98, 0.98), 11.98, 11.98, fill=False, color="#CD5334", linewidth=3)
)
# TODO: replace this with the confusion matrix after fine tuning
plot_confusion_matrix(
    axs[1], fine_tuned_cm, worst_12_labels, "After Fine Tuning", cbar=True
)
axs[1].set_xticks(np.array(range(len(worst_12_labels))) + 0.5)
axs[1].set_xticklabels(worst_12_labels, rotation=45, ha="right")

plt.suptitle("Confusion Matrices of Worst 12 Classes", fontsize=20, weight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrices_fine_tuning.png", dpi=300)
plt.show()


# %%
# define plot_metrics
def plot_metrics(
    metrics_df: pd.DataFrame,
    metrics: List[Tuple[str, str, str, str]],
    title: str,
    save_path: Path,
):
    fig, axs = plt.subplots(1, len(metrics), figsize=(8 * len(metrics), 6))
    epochs = metrics_df.index
    for ax, (val_metric_name, train_metric_name, metric_label, metric_color) in zip(
        axs, metrics
    ):
        ax.plot(
            epochs,
            metrics_df[val_metric_name],
            marker="o",
            label="Validation",
            color=metric_color,
        )
        ax.plot(
            epochs,
            metrics_df[train_metric_name],
            marker="o",
            label="Training",
            color=metric_color,
            linestyle="--",
        )

        ax.set_ylabel(metric_label, fontsize=12, weight="bold")
        ax.set_xlabel("Epoch", fontsize=12, weight="bold")
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid()

    plt.suptitle(title, fontsize=20, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# %%
# plot the training curves
experiment_ids = [24, 25]
for experiment_id in experiment_ids:
    metrics_df = metrics_dfs[f"experiment_{experiment_id}"]
    metrics: List[Tuple[str, str, str, str]] = [
        ("val_dice", "train_dice", "Dice Score", "#CD5334"),
        ("val_accuracy", "train_accuracy", "Accuracy", "#4D7B28"),
        ("val_loss", "train_loss", "Loss", "#4059AD"),
    ]
    plot_metrics(
        metrics_df,
        metrics,
        f"Training Curves",
        FIGURES_DIR / f"validation_metrics_experiment_{experiment_id}.png",
    )

# %%
# compute the mDice performance for each experiment part of hyperparam search
hyperparam_search_config = [
    (range(0, 9), "learning_rate", [0.0005, 0.0001, 0.00005]),
    (range(9, 18), "batch_size", [4, 8, 16]),
    ([11, 18, 12, 19, 17, 20], "with_augmentation", [False, True]),
    ([11, 21, 12, 22, 17, 23], "img_size", [192, 384]),
]

final_dice_scores = []

for exp_ids, hyperparam_name, hyperparam_values in hyperparam_search_config:
    final_dices = [
        metrics_dfs[f"experiment_{exp_id:02d}"]["val_dice"].values[-1]
        for exp_id in exp_ids
    ]
    final_dice_scores.append((exp_ids, hyperparam_name, hyperparam_values, final_dices))
# %%
# Prepare data for heatmaps
heatmap_data = {}
for exp_ids, hyperparam_name, hyperparam_values, final_dices in final_dice_scores:
    for i, exp_id in enumerate(exp_ids):
        model_name = MODELS[i // len(hyperparam_values)]
        hyperparam_value = hyperparam_values[i % len(hyperparam_values)]
        final_dice = final_dices[i]
        if hyperparam_name not in heatmap_data:
            heatmap_data[hyperparam_name] = {}
        if model_name not in heatmap_data[hyperparam_name]:
            heatmap_data[hyperparam_name][model_name] = []
        heatmap_data[hyperparam_name][model_name].append(final_dice)

# %%
# Create a figure with subplots and additional space for the colorbar
fig, axs = plt.subplots(3, 4, figsize=(27, 9), width_ratios=[1, 1, 0.65, 0.65])

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# Plot heatmaps
hyperparam_names = [config[1] for config in hyperparam_search_config]
for row, model in enumerate(MODELS):
    for col, (exp_ids, hyperparam_name, hyperparam_values, final_dices) in enumerate(
        final_dice_scores
    ):
        data = heatmap_data[hyperparam_name][model]

        # Use the global vmin and vmax for consistent color scaling
        im = sns.heatmap(
            [data],
            ax=axs[row, col],
            annot=True,
            cmap="PuBuGn",
            xticklabels=hyperparam_values,
            yticklabels=[model],
            cbar=False,
            vmin=0,
            vmax=0.09,
            annot_kws={"weight": "bold", "fontsize": 20},
        )
        axs[row, col].set_yticklabels(
            axs[row, col].get_yticklabels(), fontweight="bold"
        )

        best_idx = np.argmax(data)
        axs[row, col].add_patch(
            plt.Rectangle(
                (best_idx + 0.01, 0.01),
                0.98,
                0.98,
                fill=False,
                color="#CD5334",
                linewidth=4,
            )
        )

        # axs[row, col].set_title(f"{model} - {hyperparam_name}", fontsize=14)
        axs[row, col].set_xlabel(hyperparam_name, fontsize=12, weight="bold")

fig.colorbar(im.collections[0], cax=cbar_ax, label="Dice Score")

# Add arrows between the plots using figure coordinates
# These arrows will appear in the white space between subplots
arrow_ys = (0.75, 0.41, 0.12)
arrow_xs = [(0.255, 0.295), (0.525, 0.565), (0.705, 0.745)]

for row, arrow_y in enumerate(arrow_ys):
    for col, arrow_x in enumerate(
        arrow_xs
    ):  # Only add arrows between columns, not after the last column
        # Convert axis coordinates to figure coordinates
        fig.canvas.draw()  # This is needed to ensure the renderer exists
        print(arrow_x)
        # Get the positions in figure coordinates
        ax_right = arrow_x[0]
        ax_left = arrow_x[1]

        # Add an arrow annotation using figure coordinates
        fig.add_artist(
            plt.Arrow(
                ax_right,  # Start a bit to the right of the current plot
                arrow_y,
                (ax_left - ax_right),  # End a bit to the left of the next plot
                0,  # No vertical movement
                width=0.03,
                color="#CD5334",
            )
        )

plt.suptitle("Hyperparameter Search Results", fontsize=20, weight="bold")
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave room for the colorbar
plt.savefig(FIGURES_DIR / "hyperparam_search_heatmaps.png", dpi=300)
plt.show()
