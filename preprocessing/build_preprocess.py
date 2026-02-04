import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

from preprocessing.preprocessors import PreprocessSpec, Preprocessor

REFERENCE_PATH = "data/reference.csv"
OUT_DIR = "artifacts"
OUT_PATH = os.path.join(OUT_DIR, "preprocess.pkl")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 2) Load reference data
    df = pd.read_csv(REFERENCE_PATH)

    spec = PreprocessSpec()
    pre = Preprocessor(spec)

    # 3) Fit preprocessor
    pre.fit(df)

    skipped = getattr(pre, "_skipped_term_purpose_groups_", 0)
    print("Skipped (Term,Purpose) groups with NaN median:", skipped)

    # 4) Transform reference
    df_clean = pre.transform(df)

    # 5) FINAL FEATURES FOR CNN
    IDENTIFYING_COLS = ["Loan ID", "Customer ID", "Loan Status"]
    feature_columns = [c for c in df_clean.columns if c not in IDENTIFYING_COLS]

    pre.final_feature_list_ = feature_columns
    df_model = df_clean[feature_columns]

    print("Final feature count:", len(feature_columns))
    print("Final features:", feature_columns[:10], "...")

    # (optional) keep indicators for UI/reporting only
    ui_only_cols = []
    for c in ["Annual Income__missing", "Credit Score__missing"]:
        if c in df_clean.columns:
            ui_only_cols.append(c)

    # 6) Fit StandardScaler ONLY on CNN features
    X_ref = df_model.astype(float).values
    # ---- Sanity check BEFORE fitting scaler ----
    if not np.isfinite(X_ref).all():
        bad = np.isnan(X_ref).sum() + np.isinf(X_ref).sum()
        raise ValueError(f"Reference features contain NaN/Inf before scaler fit: {bad} bad values. Fix preprocessing first.")

    scaler = StandardScaler()
    scaler.fit(X_ref)

    # attach
    pre.scaler_ = scaler
    pre.ui_only_feature_list_ = ui_only_cols

    # 7) Save artifact
    joblib.dump(pre, OUT_PATH)

    # 8) Summary
    print("✅ Saved:", OUT_PATH)
    print("Final feature count:", len(feature_columns))
    print("Final features:", feature_columns[:10], "...")
    X_scaled = pre.scaler_.transform(X_ref[:5])
    print("Smoke test scaled shape:", X_scaled.shape)


if __name__ == "__main__":
    main()
