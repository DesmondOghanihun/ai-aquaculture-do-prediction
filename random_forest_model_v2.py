"""
AI-Driven Precision Aquaculture
Random Forest Model for DO Depletion Prediction — v2.2

Fixes vs. v2.1:
  - Suppress / handle UndefinedMetricWarning in grouped CV folds
  - Fix SHAP 3-D array issue with NumPy 2.5+ (TreeExplainer multi-output)
  - Fix threshold-selection bug for the 95%-recall threshold
  - Suppress SHAP FutureWarning on np.random.seed (global RNG deprecation)
  - Add a per-fold report so you can see high-risk counts per validation fold
  - Consistent output paths; script is idempotent (safe to re-run)
"""

import os
import sys
import platform
import warnings

import numpy as np
import pandas as pd
import sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. SHAP analysis will be skipped.")
    print("         Install with: pip install shap")

# ----------------------------------------------------------------
# Silence known benign warnings
# ----------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")
# Suppress the flood of "F-score is ill-defined" warnings during grouped CV
warnings.filterwarnings("ignore", message=".*F-score is ill-defined.*")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directories
os.makedirs("outputs", exist_ok=True)
os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"SHAP available: {HAS_SHAP}")


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
CSV_PATH = os.path.join("data", "water_quality_data_Mar_Aug_2026.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = "water_quality_data_Mar_Aug_2026.csv"

print(f"\nLoading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
df = df.sort_values("Sampling_Date").reset_index(drop=True)

print("=" * 70)
print("DATA OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Date range: {df['Sampling_Date'].min().date()} to {df['Sampling_Date'].max().date()}")
print(f"Ponds: {df['Pond_ID'].nunique()}")
print(f"High-risk events: {int(df['DO_Risk'].sum())} ({df['DO_Risk'].mean()*100:.1f}%)")


# ============================================================
# STEP 2: FEATURES
# ============================================================
feature_columns = [
    "Water_Temp_C", "Air_Temp_C", "Humidity_pct",
    "Barometric_Pressure_hPa", "Precipitation_mm",
    "Solar_Radiation_W_m2", "Turbidity_NTU", "Ammonia_mgL",
    "Nitrite_mgL", "Phytoplankton_Density_cells_ml",
    "Stocking_Density_fish_m2", "pH",
]

pond_dummies = pd.get_dummies(df["Pond_Type"], prefix="Pond", drop_first=False)
df = pd.concat([df, pond_dummies], axis=1)
feature_columns += pond_dummies.columns.tolist()

X = df[feature_columns].fillna(df[feature_columns].median())
y = df["DO_Risk"].astype(int)
groups = df["Pond_ID"]


# ============================================================
# STEP 3A: TEMPORAL SPLIT
# ============================================================
print("\n" + "=" * 70)
print("TEMPORAL SPLIT (train Mar-Jun, test Jul-Aug)")
print("=" * 70)

train_mask = df["Sampling_Date"] < "2026-07-01"
test_mask  = df["Sampling_Date"] >= "2026-07-01"

X_train_t = X.loc[train_mask].reset_index(drop=True)
y_train_t = y.loc[train_mask].reset_index(drop=True)
X_test_t  = X.loc[test_mask].reset_index(drop=True)
y_test_t  = y.loc[test_mask].reset_index(drop=True)
groups_train = groups.loc[train_mask].reset_index(drop=True).values

print(f"Train: {len(X_train_t)} obs, {int(y_train_t.sum())} high-risk ({y_train_t.mean()*100:.1f}%)")
print(f"Test:  {len(X_test_t)} obs, {int(y_test_t.sum())} high-risk ({y_test_t.mean()*100:.1f}%)")


# ============================================================
# STEP 3B: GROUPED 5-FOLD CV (pickle-safe, warning-safe)
# ============================================================
print("\n" + "=" * 70)
print("GROUPED 5-FOLD CV (by Pond_ID) for hyperparameter tuning")
print("=" * 70)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

rf_base = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=1,
)

gkf = GroupKFold(n_splits=5)

# Materialise generator -> list of (train_idx, val_idx) — required for n_jobs > 1
cv_splits = list(gkf.split(X_train_t, y_train_t, groups=groups_train))

# Report fold composition (also explains why some warnings were firing)
print("Fold composition:")
for i, (tr_idx, va_idx) in enumerate(cv_splits):
    n_hi_tr = int(y_train_t.iloc[tr_idx].sum())
    n_hi_va = int(y_train_t.iloc[va_idx].sum())
    print(f"  Fold {i+1}: train n={len(tr_idx)} (high-risk={n_hi_tr}), "
          f"val n={len(va_idx)} (high-risk={n_hi_va})")
    if n_hi_va == 0:
        print(f"    WARNING: fold {i+1} has 0 high-risk samples in validation — "
              f"F1 undefined for this fold, will be skipped by CV")

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=cv_splits,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    error_score=np.nan,     # <-- don't crash on a bad fold
)
grid_search.fit(X_train_t, y_train_t)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best grouped-CV F1: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_


# ============================================================
# STEP 4: BASELINE COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

baselines = {
    "Always Low Risk":  DummyClassifier(strategy="most_frequent"),
    "Stratified Dummy": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    "Prior Dummy":      DummyClassifier(strategy="prior"),
}

baseline_results = []
for name, clf in baselines.items():
    clf.fit(X_train_t, y_train_t)
    yp = clf.predict(X_test_t)
    baseline_results.append({
        "Model": name,
        "Accuracy":  accuracy_score(y_test_t, yp),
        "Precision": precision_score(y_test_t, yp, zero_division=0),
        "Recall":    recall_score(y_test_t, yp, zero_division=0),
        "F1":        f1_score(y_test_t, yp, zero_division=0),
    })

y_pred_rf = best_rf.predict(X_test_t)
y_proba_rf = best_rf.predict_proba(X_test_t)[:, 1]
baseline_results.append({
    "Model": "Random Forest (this study)",
    "Accuracy":  accuracy_score(y_test_t, y_pred_rf),
    "Precision": precision_score(y_test_t, y_pred_rf),
    "Recall":    recall_score(y_test_t, y_pred_rf),
    "F1":        f1_score(y_test_t, y_pred_rf),
})

baseline_df = pd.DataFrame(baseline_results).round(4)
print(baseline_df.to_string(index=False))
baseline_df.to_csv(os.path.join("outputs", "baseline_comparison.csv"), index=False)


# ============================================================
# STEP 5: FULL TEST-SET METRICS
# ============================================================
print("\n" + "=" * 70)
print("RANDOM FOREST — TEMPORAL TEST SET (Jul-Aug 2026)")
print("=" * 70)

acc  = accuracy_score(y_test_t, y_pred_rf)
prec = precision_score(y_test_t, y_pred_rf)
rec  = recall_score(y_test_t, y_pred_rf)
f1   = f1_score(y_test_t, y_pred_rf)
auc  = roc_auc_score(y_test_t, y_proba_rf)
ap   = average_precision_score(y_test_t, y_proba_rf)
brier = brier_score_loss(y_test_t, y_proba_rf)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"PR-AUC:    {ap:.4f}")
print(f"Brier:     {brier:.4f}")

cm = confusion_matrix(y_test_t, y_pred_rf)
print(f"\nConfusion matrix:\n{cm}")
print("\n" + classification_report(y_test_t, y_pred_rf, target_names=["Low Risk", "High Risk"]))

pd.DataFrame({
    "Pond_ID": df.loc[test_mask, "Pond_ID"].values,
    "Sampling_Date": df.loc[test_mask, "Sampling_Date"].dt.strftime("%Y-%m-%d").values,
    "y_true": y_test_t.values,
    "y_proba": y_proba_rf,
    "y_pred": y_pred_rf,
}).to_csv(os.path.join("outputs", "test_predictions.csv"), index=False)


# ============================================================
# STEP 6: PRECISION-RECALL CURVE + THRESHOLD SELECTION
# ============================================================
print("\n" + "=" * 70)
print("THRESHOLD SELECTION")
print("=" * 70)

precisions, recalls, thresholds = precision_recall_curve(y_test_t, y_proba_rf)
# NOTE: len(thresholds) == len(precisions) - 1; recalls has one extra (recall=0 at the end)
f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
best_idx = int(np.argmax(f1s))
thr_f1_opt = float(thresholds[best_idx])

# Threshold for >= 95% recall — pick the LOWEST threshold that satisfies recall>=0.95
# (lower threshold = more positives = higher recall)
recall_target = 0.95
# Thresholds are ascending; recalls are descending from 1.0 down.
# Find the largest threshold where recall is still >= 0.95.
candidates = np.where(recalls[:-1] >= recall_target)[0]
if len(candidates):
    # Among thresholds that satisfy recall>=0.95, pick the largest (highest precision)
    thr_95recall = float(thresholds[candidates[-1]])
    # Compute metrics at that threshold
    y_pred_95 = (y_proba_rf >= thr_95recall).astype(int)
    prec_95 = precision_score(y_test_t, y_pred_95, zero_division=0)
    rec_95  = recall_score(y_test_t, y_pred_95, zero_division=0)
    f1_95   = f1_score(y_test_t, y_pred_95, zero_division=0)
else:
    thr_95recall = 0.0
    prec_95, rec_95, f1_95 = 0.0, 0.0, 0.0

print(f"Default threshold 0.50       -> recall={rec:.3f}, precision={prec:.3f}, F1={f1:.3f}")
print(f"F1-optimal threshold {thr_f1_opt:.3f}  -> recall={recalls[best_idx]:.3f}, "
      f"precision={precisions[best_idx]:.3f}, F1={f1s[best_idx]:.3f}")
print(f"95%-recall threshold {thr_95recall:.3f} -> recall={rec_95:.3f}, "
      f"precision={prec_95:.3f}, F1={f1_95:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recalls, precisions, label=f"RF (AP={ap:.3f})")
ax.axhline(y_test_t.mean(), color="grey", linestyle="--", label="No-skill baseline")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (Temporal Test Set)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join("outputs", "figures", "fig_precision_recall_curve.png"), dpi=150)
plt.close()


# ============================================================
# STEP 7: CALIBRATION CURVE
# ============================================================
prob_true, prob_pred = calibration_curve(y_test_t, y_proba_rf, n_bins=10, strategy="uniform")
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(prob_pred, prob_true, marker="o", label="RF")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title(f"Calibration (Brier = {brier:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join("outputs", "figures", "fig_calibration_curve.png"), dpi=150)
plt.close()


# ============================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================
imp = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": best_rf.feature_importances_,
}).sort_values("Importance", ascending=False).reset_index(drop=True)
print("\nFeature importances:")
print(imp.to_string(index=False))
imp.to_csv(os.path.join("outputs", "feature_importance.csv"), index=False)


# ============================================================
# STEP 9: SHAP ANALYSIS (with NumPy 2.x / SHAP 0.46+ fix)
# ============================================================
if HAS_SHAP:
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)
    try:
        explainer = shap.TreeExplainer(best_rf)
        shap_vals = explainer.shap_values(X_test_t, check_additivity=False)

        # SHAP 0.46+ with scikit-learn 1.6+ can return:
        #   - a 2-D array (n_samples, n_features) for binary classification, OR
        #   - a list [class0_array, class1_array], OR
        #   - a 3-D array (n_samples, n_features, n_classes)
        # We want the "high-risk" class contributions.
        if isinstance(shap_vals, list):
            sv_high = shap_vals[1]          # class 1 = high risk
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sv_high = shap_vals[:, :, 1]    # (n_samples, n_features)
        else:
            sv_high = shap_vals

        sv_high = np.asarray(sv_high)
        if sv_high.ndim != 2:
            raise ValueError(f"Unexpected SHAP shape: {sv_high.shape}")
        print(f"SHAP values shape: {sv_high.shape}")

        # Save mean |SHAP|
        shap_imp = pd.DataFrame({
            "Feature": X_test_t.columns,
            "MeanAbsSHAP": np.abs(sv_high).mean(axis=0),
        }).sort_values("MeanAbsSHAP", ascending=False)
        shap_imp.to_csv(os.path.join("outputs", "shap_importance.csv"), index=False)
        print("\nMean |SHAP| per feature:")
        print(shap_imp.to_string(index=False))

        # Beeswarm plot
        plt.figure()
        shap.summary_plot(sv_high, X_test_t, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(os.path.join("outputs", "figures", "fig_shap_summary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Bar plot
        plt.figure()
        shap.summary_plot(sv_high, X_test_t, plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(os.path.join("outputs", "figures", "fig_shap_bar.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        print("SHAP outputs saved.")
    except Exception as e:
        print(f"SHAP computation failed (non-fatal): {type(e).__name__}: {e}")


# ============================================================
# STEP 10: SAVE METRICS
# ============================================================
pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Brier"],
    "Value":  [acc, prec, rec, f1, auc, ap, brier],
}).to_csv(os.path.join("outputs", "model_performance_metrics.csv"), index=False)

print("\n" + "=" * 70)
print("DONE. Outputs written to ./outputs/")
print("=" * 70)