"""
calibration_analysis.py
Computes and saves the calibration table + Brier score for the temporal test set.

Run from project root:
    python calibration_analysis.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

PRED_PATH = os.path.join("outputs", "test_predictions.csv")
OUT_CSV   = os.path.join("outputs", "calibration_table.csv")

if not os.path.exists(PRED_PATH):
    raise FileNotFoundError(
        f"{PRED_PATH} not found. Run `python random_forest_model_v2.py` first."
    )

preds = pd.read_csv(PRED_PATH)
y_true  = preds["y_true"].values
y_proba = preds["y_proba"].values

brier = brier_score_loss(y_true, y_proba)

# 10 uniform-width bins, matching Section 3.4 of the manuscript
prob_true, prob_pred = calibration_curve(
    y_true, y_proba, n_bins=10, strategy="uniform"
)

# Number of observations per bin
bin_edges = np.linspace(0, 1, 11)
counts, _ = np.histogram(y_proba, bins=bin_edges)

rows = []
for i in range(10):
    rows.append({
        "Bin":           i + 1,
        "Range":         f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
        "MeanPredicted": prob_pred[i],
        "ObservedRate":  prob_true[i],
        "N":             int(counts[i]),
    })

table = pd.DataFrame(rows)
table.to_csv(OUT_CSV, index=False)

# Console summary
print("=" * 70)
print("CALIBRATION ANALYSIS (temporal test set)")
print("=" * 70)
print(f"Brier score:              {brier:.4f}")
print(f"Observations:             {len(y_true)}")
print(f"Positive events:          {int(y_true.sum())}")
print(f"Positive rate:            {y_true.mean():.4f}")
print()
print(table.to_string(index=False))
print()

# Save for reproducibility
pd.DataFrame({"y_true": y_true, "y_proba": y_proba}).to_csv(
    os.path.join("outputs", "calibration_raw.csv"), index=False
)
print(f"Wrote {OUT_CSV}")
print("Wrote outputs/calibration_raw.csv")