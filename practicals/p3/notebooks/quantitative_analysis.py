# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from get_results import get_metrics_dfs, get_runs_df


# %%
# Plot distribution of metrics across all runs
def plot_metric_violin(df, metric_name, title=None):
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="run_id", y=metric_name, inner="quartile")
    plt.title(title or f"Distribution of {metric_name} across runs")
    plt.xlabel("Run ID")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# %%
# Plot correlation heatmap for metrics
def plot_correlation_heatmap(df, metric_columns, title):
    fig = plt.figure(figsize=(12, 10))
    # Ensure only numeric columns are used for correlation
    numeric_df = df[metric_columns].select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title(title)
    plt.tight_layout()
    return fig


# %%
# Plot combined distribution of frame and video metrics across all runs
def plot_combined_metric_distribution(frame_df, video_df, metric_name, title=None):
    """Plots a split violin plot comparing frame-level and video-level metric distributions."""
    fig = plt.figure(figsize=(12, 7))

    # Prepare data for combined plot
    frame_data = frame_df[["run_id", metric_name]].copy()
    frame_data["level"] = "Frame"
    video_data = video_df[["run_id", metric_name]].copy()
    video_data["level"] = "Video"

    combined_data = pd.concat([frame_data, video_data], ignore_index=True)

    sns.violinplot(
        data=combined_data,
        x="run_id",
        y=metric_name,
        hue="level",
        split=True,
        inner="quartile",  # Show quartiles
        palette={"Frame": "lightskyblue", "Video": "lightcoral"},
    )
    plt.title(title or f"Combined Distribution of {metric_name} (Frame vs Video)")
    plt.xlabel("Run ID")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.legend(title="Metric Level")
    plt.tight_layout()
    return fig


# %%
# Plot training curves from runs_df
def plot_training_curves(runs_df, metric_name, title=None):
    """Plots training curves for a given metric from the runs_df,
    with a subplot for each run_id, fixed ylim, and special test set viz."""
    if metric_name not in runs_df.columns:
        print(f"Metric '{metric_name}' not found in runs_df. Skipping plot.")
        return

    metric_data_all_runs = runs_df[metric_name]
    if metric_data_all_runs.isnull().all():
        print(f"All values for metric '{metric_name}' are NaN. Skipping plot.")
        return

    metric_min = metric_data_all_runs.min()
    metric_max = metric_data_all_runs.max()

    range_val = metric_max - metric_min
    padding = range_val * 0.05
    if (
        padding == 0
    ):  # Handles cases where all values are the same or only one point for the metric
        padding = abs(metric_max) * 0.1 if metric_max != 0 else 0.1
    if padding == 0:  # If metric_max was 0, padding is still 0
        padding = 0.1  # Default absolute padding

    ylim_bottom = metric_min - padding
    ylim_top = metric_max + padding

    unique_run_ids = sorted(runs_df["run_id"].unique())
    if not unique_run_ids:
        print(f"No run_ids found for metric '{metric_name}'. Skipping plot.")
        return

    num_runs = len(unique_run_ids)
    ncols = min(num_runs, 2) if num_runs > 1 else 1
    nrows = (num_runs + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False
    )
    fig.suptitle(title or f"Training Curves for {metric_name}", fontsize=16, y=0.98)

    axes_flat = axes.flatten()

    for i, run_id in enumerate(unique_run_ids):
        ax = axes_flat[i]
        run_data = runs_df[
            (runs_df["run_id"] == run_id) & (runs_df[metric_name].notna())
        ]

        # Plot train and validation lines
        train_val_data = run_data[run_data["set"].isin(["train", "val"])]
        if not train_val_data.empty:
            sns.lineplot(
                data=train_val_data,
                x="epoch",
                y=metric_name,
                hue="set",
                style="set",
                palette={"train": "royalblue", "val": "darkorange"},
                linewidth=2,
                ax=ax,
            )

        # Plot test point if available
        test_data = run_data[run_data["set"] == "test"]
        if not test_data.empty:
            # Assuming we want the test point from the latest epoch if multiple exist
            latest_test_point = test_data.loc[test_data["epoch"].idxmax()]
            # Determine the x-coordinate for the test point based on max train/val epoch
            max_train_val_epoch = 0  # Default if no train/val data
            if not train_val_data.empty:
                max_train_val_epoch = train_val_data["epoch"].max()
            else:  # If only test data, use its own epoch for x
                max_train_val_epoch = latest_test_point["epoch"]

            # For visual clarity, place test point slightly after the last train/val epoch
            # Or directly at max_train_val_epoch if preferred. Let's try directly for now.
            test_plot_x_coord = max_train_val_epoch

            test_value = latest_test_point[metric_name]
            if pd.notna(test_value):
                ax.scatter(
                    test_plot_x_coord,
                    test_value,
                    color="forestgreen",
                    s=100,
                    label=f"Test: {test_value:.3f}",
                    marker="*",
                    zorder=5,
                )
                ax.text(
                    test_plot_x_coord,
                    test_value,
                    f" {test_value:.3f}",
                    color="forestgreen",
                    va="center",
                    ha="left",
                )

        ax.set_ylim(ylim_bottom, ylim_top)
        ax.set_title(f"Run ID: {run_id}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)

        # Create a comprehensive legend
        handles, labels = ax.get_legend_handles_labels()
        # Simplify test label for legend, value is already on plot
        simplified_labels = [
            lab.split(":")[0] if "Test:" in lab else lab for lab in labels
        ]

        # Use a dictionary to ensure unique legend entries, preserving order of first appearance
        legend_dict = {}
        for handle, label in zip(handles, simplified_labels):
            if label not in legend_dict:
                legend_dict[label] = handle

        if legend_dict:  # Only show legend if there are items to show
            ax.legend(legend_dict.values(), legend_dict.keys(), title="Set")
        else:
            ax.legend_ = None  # Remove empty legend box if nothing plotted

        ax.grid(True)

    for j in range(num_runs, nrows * ncols):
        fig.delaxes(axes_flat[j])

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# %%
if __name__ == "__main__":
    # Get the metrics dataframes
    frame_metrics, video_metrics = get_metrics_dfs()

    # Get the runs dataframe
    runs_df = get_runs_df()

    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")  # Using a specific seaborn style
    sns.set_palette("husl")
    # Get all metric columns (excluding run_id, video_id, frame_id)
    metric_columns = [
        col
        for col in frame_metrics.columns
        if col not in ["run_id", "video_id", "frame_id"]
    ]

    # Plot distributions for each metric
    for metric in metric_columns:
        plot_metric_violin(frame_metrics, metric, f"Frame-level {metric} Distribution")
        plt.show()
        plot_metric_violin(video_metrics, metric, f"Video-level {metric} Distribution")
        plt.show()

    plot_correlation_heatmap(
        frame_metrics, metric_columns, "Frame-level Metrics Correlation"
    )
    plt.show()
    plot_correlation_heatmap(
        video_metrics, metric_columns, "Video-level Metrics Correlation"
    )
    plt.show()

    # Plot combined distributions for each metric
    for metric in metric_columns:
        plot_combined_metric_distribution(frame_metrics, video_metrics, metric)
        plt.show()

    # Get metrics available in runs_df for plotting training curves
    if not runs_df.empty:
        run_metric_columns = [
            col
            for col in runs_df.columns
            if col not in ["run_id", "epoch", "set"]  # Exclude identifiers
        ]

        # Plot training curves for each relevant metric
        for metric in run_metric_columns:
            plot_training_curves(runs_df, metric)
            plt.show()
    else:
        print("runs_df is empty. Skipping training curve plots.")
