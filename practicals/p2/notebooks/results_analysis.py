# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("..")
sys.path.append("../src")

from src.metrics import METRICS_DIR, CONFUSION_MATRICES_DIR
from src.dataset import MAIN_ITEM_NAMES
# %%
labels = ["background"] + MAIN_ITEM_NAMES
metrics_dfs = [
    pd.read_csv(path, index_col=0) for path in METRICS_DIR.glob("experiment_*.csv")
]
confusion_matrix_dfs = [
    pd.read_csv(path, index_col=0) for path in CONFUSION_MATRICES_DIR.glob("experiment_*.csv")
]

# %%

cm = confusion_matrix_dfs[0]
cm = cm.div(cm.sum(axis=1), axis=0)
# %%
fig, ax = plt.subplots(1,1, figsize=(12,10))
logarithmic_cm = np.log(cm + 1)
sns.heatmap(cm, cmap="Blues", fmt="d", ax=ax, xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor="lightgrey")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.show()

