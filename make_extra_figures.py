"""
Generates the two figures missing from outputs/figures/:
  - fig_confusion_matrix.png
  - fig_feature_importance.png
Reads the existing CSVs the main script already produced.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.makedirs("outputs/figures", exist_ok=True)

# ---------- Confusion matrix ----------
preds = pd.read_csv("outputs/test_predictions.csv")
cm = confusion_matrix(preds["y_true"], preds["y_pred"])
print("Confusion matrix:\n", cm)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"], cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Temporal Test Set, n=486)")
plt.tight_layout()
plt.savefig("outputs/figures/fig_confusion_matrix.png", dpi=150)
plt.close()

# ---------- Feature importance bar chart ----------
imp = pd.read_csv("outputs/feature_importance.csv").head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(imp["Feature"], imp["Importance"], color="steelblue")
for i, (f, v) in enumerate(zip(imp["Feature"], imp["Importance"])):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("Importance Score")
ax.set_title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("outputs/figures/fig_feature_importance.png", dpi=150)
plt.close()

print("Wrote outputs/figures/fig_confusion_matrix.png")
print("Wrote outputs/figures/fig_feature_importance.png")