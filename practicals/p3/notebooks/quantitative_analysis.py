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

from src.config import RESULTS_DIR


# %%
# Load the metrics data
def get_metrics_dfs():
    frame_metrics = pd.read_csv(RESULTS_DIR / "frame_metrics.csv")
    video_metrics = pd.read_csv(RESULTS_DIR / "video_metrics.csv")
    return frame_metrics, video_metrics


# %%
# Get the metrics dataframes
frame_metrics, video_metrics = get_metrics_dfs()
# %%
# Set up the plotting style
plt.style.use("seaborn-v0_8-whitegrid")  # Using a specific seaborn style
sns.set_palette("husl")


# %%
# Plot distribution of metrics across all runs
def plot_metric_violin(df, metric_name, title=None):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="run_id", y=metric_name, inner="quartile")
    plt.title(title or f"Distribution of {metric_name} across runs")
    plt.xlabel("Run ID")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
# Get all metric columns (excluding run_id, video_id, frame_id)
metric_columns = [
    col
    for col in frame_metrics.columns
    if col not in ["run_id", "video_id", "frame_id"]
]

# %%
# Plot distributions for each metric
for metric in metric_columns:
    plot_metric_violin(frame_metrics, metric, f"Frame-level {metric} Distribution")
    plot_metric_violin(video_metrics, metric, f"Video-level {metric} Distribution")


# %%
# Plot correlation heatmap for metrics
def plot_correlation_heatmap(df, title):
    plt.figure(figsize=(12, 10))
    # Ensure only numeric columns are used for correlation
    numeric_df = df[metric_columns].select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_correlation_heatmap(frame_metrics, "Frame-level Metrics Correlation")
plot_correlation_heatmap(video_metrics, "Video-level Metrics Correlation")


# %%
# Plot combined distribution of frame and video metrics across all runs
def plot_combined_metric_distribution(frame_df, video_df, metric_name, title=None):
    """Plots a split violin plot comparing frame-level and video-level metric distributions."""
    plt.figure(figsize=(12, 7))

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
    plt.show()


# %%
# Plot combined distributions for each metric
for metric in metric_columns:
    plot_combined_metric_distribution(frame_metrics, video_metrics, metric)
