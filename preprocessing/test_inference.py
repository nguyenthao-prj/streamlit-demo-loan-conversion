import joblib
import pandas as pd
import numpy as np

PREPROCESS_PATH = "artifacts/preprocess.pkl"
INPUT_PATH = "data/reference.csv"   

def main():
    # 1) Load preprocess artifact
    pre = joblib.load(PREPROCESS_PATH)

    # 2) Load raw inference data
    df_raw = pd.read_csv(INPUT_PATH)

    n_total = len(df_raw)

    # 3) Basic diagnostics BEFORE preprocess
    placeholder_mask = df_raw["Current Loan Amount"] == 99999999
    n_placeholder = placeholder_mask.sum()

    n_missing_income = df_raw["Annual Income"].isna().sum()
    n_missing_score = df_raw["Credit Score"].isna().sum()

    # 4) Transform
    df_clean = pre.transform(df_raw)
    # --- Find remaining NaNs after preprocess (before scaler)
    nan_cols = df_clean[pre.final_feature_list_].isna().sum()
    nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)

    print("\n--- Remaining NaNs after preprocess (before scaler) ---")
    if len(nan_cols) == 0:
        print("None ✅")
    else:
        print(nan_cols.to_string())
    
    # 5) Diagnostics AFTER preprocess
    n_income_imputed = df_clean["Annual Income__missing"].sum()
    n_score_imputed = df_clean["Credit Score__missing"].sum()

    fill_zero_cols = [
        "Months since last delinquent",
        "Years in current job",
        "Bankruptcies",
        "Tax Liens",
    ]
    fill_zero_stats = {
        c: int((df_raw[c].isna()).sum()) if c in df_raw.columns else 0
        for c in fill_zero_cols
    }

    # 6) Final feature matrix
    X = df_clean[pre.final_feature_list_].astype(float).values
    X_scaled = pre.scaler_.transform(X)

    # 7) Sanity checks
    has_nan = np.isnan(X_scaled).any()
    has_inf = np.isinf(X_scaled).any()

    # 8) REPORT
    print("========== INFERENCE PREPROCESS REPORT ==========")
    print(f"Total records: {n_total}")
    print(f"Loan Amount placeholder (99,999,999): {n_placeholder}")

    print("\n--- Imputation ---")
    print(f"Annual Income imputed: {n_income_imputed}")
    print(f"Credit Score imputed: {n_score_imputed}")

    print("\n--- Fill-zero (from NaN) ---")
    for k, v in fill_zero_stats.items():
        print(f"{k}: {v}")

    print("\n--- Final matrix ---")
    print("Feature count:", X_scaled.shape[1])
    print("Has NaN:", has_nan)
    print("Has Inf:", has_inf)

    print("===============================================")

if __name__ == "__main__":
    main()
