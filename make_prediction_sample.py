"""
make_prediction_sample.py — slide-ready 6-row prediction table (v2, no emoji).

Run from project root:
    python make_prediction_sample.py
Outputs:
    outputs/figures/fig_prediction_sample.png
    outputs/prediction_sample.csv
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

preds = pd.read_csv("outputs/test_predictions.csv")

# --- Choose 6 representative rows: 3 TP, 2 TN, 1 FP (your console output) ---
tp = preds[(preds.y_true == 1) & (preds.y_pred == 1)].head(3)
tn = preds[(preds.y_true == 0) & (preds.y_pred == 0)].head(2)
fp = preds[(preds.y_true == 0) & (preds.y_pred == 1)].head(1)

sample = pd.concat([tp, tn, fp]).reset_index(drop=True)
sample["Date"]       = pd.to_datetime(sample["Sampling_Date"]).dt.strftime("%d-%b-%Y")
sample["Actual"]     = sample["y_true"].map({1: "Hypoxia", 0: "Normal"})
sample["Prob"]       = sample["y_proba"].round(3)
sample["Model says"] = sample["y_pred"].map({1: "ALERT", 0: "Safe"})

# ASCII labels (no emoji — avoids font warnings)
def _verdict(row):
    if row["y_true"] == 1 and row["y_pred"] == 1:   return "CORRECT"
    if row["y_true"] == 0 and row["y_pred"] == 0:   return "CORRECT"
    if row["y_true"] == 0 and row["y_pred"] == 1:   return "FALSE ALARM"
    if row["y_true"] == 1 and row["y_pred"] == 0:   return "MISSED"
    return "?"

sample["Result"] = sample.apply(_verdict, axis=1)

display = sample[["Pond_ID", "Date", "Actual", "Prob", "Model says", "Result"]]

# --- Render as a figure ---
fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
ax.axis("off")
tbl = ax.table(
    cellText=display.values,
    colLabels=display.columns,
    cellLoc="center", loc="center",
    colColours=["#0B3D91"] * len(display.columns),
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.7)

# Colour-code: green = correct, orange = false alarm, red = missed
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold", color="white", size=11)
        cell.set_facecolor("#0B3D91")
        continue
    verdict = sample.iloc[row - 1]["Result"]
    if verdict == "CORRECT":
        cell.set_facecolor("#E8F5E9")
    elif verdict == "FALSE ALARM":
        cell.set_facecolor("#FFF3E0")
    elif verdict == "MISSED":
        cell.set_facecolor("#FFCDD2")

# Highlight the "Prob" column in teal to draw the eye
for row in range(1, len(sample) + 1):
    cell = tbl[(row, 3)]
    cell.set_text_props(weight="bold", color="#008080")

ax.set_title(
    "Real predictions on 6 unseen samples (July–August 2026 test set)",
    fontsize=13, pad=8, weight="bold"
)
plt.tight_layout()
plt.savefig("outputs/figures/fig_prediction_sample.png", dpi=200, bbox_inches="tight")
plt.close()

display.to_csv("outputs/prediction_sample.csv", index=False)
print("Wrote outputs/figures/fig_prediction_sample.png")
print(display.to_string(index=False))