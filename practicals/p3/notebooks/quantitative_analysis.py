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
frame_metrics = pd.read_csv(RESULTS_DIR / "frame_metrics.csv")
video_metrics = pd.read_csv(RESULTS_DIR / "video_metrics.csv")

# %%
# Set up the plotting style
plt.style.use("seaborn-v0_8-whitegrid")  # Using a specific seaborn style
sns.set_palette("husl")


# %%
# Plot distribution of metrics across all runs
def plot_metric_distribution(df, metric_name, title=None):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="run_id", y=metric_name)
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
    plot_metric_distribution(
        frame_metrics, metric, f"Frame-level {metric} Distribution"
    )
    plot_metric_distribution(
        video_metrics, metric, f"Video-level {metric} Distribution"
    )


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
# Plot metric trends across runs
def plot_metric_trends(df, metric_name, title=None):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="run_id", y=metric_name, errorbar="ci", marker="o")
    plt.title(title or f"{metric_name} Trends Across Runs")
    plt.xlabel("Run ID")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
# Plot trends for each metric
for metric in metric_columns:
    plot_metric_trends(frame_metrics, metric, f"Frame-level {metric} Trends")
    plot_metric_trends(video_metrics, metric, f"Video-level {metric} Trends")
