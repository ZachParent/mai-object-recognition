# %%
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
labels = ["background"] + MAIN_ITEM_NAMES
metrics_dfs = {}
for path in METRICS_DIR.glob("experiment_*.csv"):
    metrics_dfs[path.stem] = pd.read_csv(path, index_col=0)

confusion_matrix_dfs = {}
for path in CONFUSION_MATRICES_DIR.glob("experiment_*.csv"):
    confusion_matrix_dfs[path.stem] = pd.read_csv(path, index_col=0)


# %%

cm = confusion_matrix_dfs["experiment_11"]
cm = cm.div(cm.sum(axis=1), axis=0)
# %%
fig, ax = plt.subplots(1,1, figsize=(12,10))
logarithmic_cm = np.log(cm + 1)
sns.heatmap(cm, cmap="Blues", fmt="d", ax=ax, xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor="lightgrey")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.show()

