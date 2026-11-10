# AI-Driven Precision Aquaculture: DO Depletion Prediction

Random Forest classifier predicting hypoxia events (DO < 4.0 mg/L) in tropical fish ponds in Delta State, Nigeria.

## Dataset
- 1,422 observations, 18 ponds, 3 pond types (Tarpaulin, Concrete, Earthen)
- Sampling period: 2026-03-02 to 2026-08-31
- Six localities in Asaba and environs: Okwe, Ibusa, Issele Asagba, Okpanam, Anwai, Ugbolu

## Methodology
- Binary target: DO_Risk = 1 if DO < 4.0 mg/L, else 0
- 15 features (12 numeric + 3 one-hot pond-type dummies)
- Temporal train/test split: train Mar–Jun 2026, test Jul–Aug 2026 (no leakage)
- GroupKFold CV by Pond_ID (accounts for pond clustering)
- Grid search on n_estimators, max_depth, min_samples_split, min_samples_leaf
- class_weight='balanced' for the 12.9% minority class
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier, calibration
- SHAP analysis (if the `shap` package is available)

## Reproduce

```bash
pip install -r requirements.txt
python random_forest_model_v2.py