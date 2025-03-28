# %%
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("..")
sys.path.append("../src")

from src.config import METRICS_DIR, DATA_DIR
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
# compute confusion matrix

cm = confusion_matrix_dfs["experiment_11"]
# normalize the predictions by the true counts
true_counts = cm.sum(axis=1)
cm = cm.div(true_counts, axis=0)

# sort the labels by the true counts
sorted_indices = true_counts.sort_values(ascending=False).index
cm = cm.iloc[sorted_indices, sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]

# %% 
# plot confusion matrix
fig, ax = plt.subplots(1,1, figsize=(12,10))
sns.heatmap(cm, cmap="Blues", fmt="d", ax=ax, xticklabels=sorted_labels, yticklabels=sorted_labels, linewidths=0.5, linecolor="lightgrey")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.show()

# %% 
# compute tp-sorted confusion matrix
tp_counts = cm.values.diagonal()
tp_sorted_indices = tp_counts.argsort()[::-1]
tp_sorted_cm = cm.iloc[tp_sorted_indices, tp_sorted_indices]
tp_sorted_labels = [labels[i] for i in tp_sorted_indices]

# %%
# plot tp-sorted confusion matrix
fig, ax = plt.subplots(1,1, figsize=(12,10))
sns.heatmap(tp_sorted_cm, cmap="Blues", fmt="d", ax=ax, xticklabels=tp_sorted_labels, yticklabels=tp_sorted_labels, linewidths=0.5, linecolor="lightgrey")
ax.set_title("TP-Sorted Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")