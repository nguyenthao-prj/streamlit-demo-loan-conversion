
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

PREPROCESS_PATH = "artifacts/preprocess.pkl"

COLAB_MODEL_PATH = "assets/colab/best_model.keras"
COLAB_SCALER_PATH = "assets/colab/scaler.pkl"
COLAB_FEATURES_PATH = "assets/colab/feature_columns.json"
DEFAULT_INPUT_PATH = "data/reference.csv"


def score_dataframe(df_raw: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    # ----------------------------
    # 1) Load artifacts
    # ----------------------------
    pre = joblib.load(PREPROCESS_PATH)

    with open(COLAB_FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    scaler = joblib.load(COLAB_SCALER_PATH)
    model = load_model(COLAB_MODEL_PATH)

    # ----------------------------
    # 2) Preprocess -> df_clean
    # ----------------------------
    df_clean = pre.transform(df_raw)
    if len(df_clean) == 0:
        raise ValueError(
            "After preprocessing, 0 rows remain. "
            "Check uploaded CSV: core numeric columns may be missing/invalid."
        )

    # Ensure Customer ID exists (for output)
    if "Customer ID" in df_clean.columns:
        cust_id = df_clean["Customer ID"].values
    elif "Customer ID" in df_raw.columns:
        cust_id = df_raw["Customer ID"].values[: len(df_clean)]
    else:
        cust_id = np.arange(len(df_clean))

    # ----------------------------
    # 3) Build X for COLAB model (24 features)
    #    - Reindex by colab feature list
    #    - Missing one-hot columns => 0
    # ----------------------------
    X_raw = df_clean.reindex(columns=feature_cols)
    X_raw = X_raw.fillna(0.0).astype(float)

    X_scaled = scaler.transform(X_raw.values)

    # sanity check
    if not np.isfinite(X_scaled).all():
        raise ValueError("Scaled features contain NaN/Inf. Check colab scaler and preprocessing consistency.")

    # CNN input: (N, 24, 1)
    X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # ----------------------------
    # 4) Predict + reconstruction error (match Colab)
    #    Colab trained with target = X_scaled (2D)
    #    Model output expected: (N, 24)
    # ----------------------------
    X_hat = model.predict(X_cnn, verbose=0)
    X_hat = np.asarray(X_hat, dtype=float)

    if X_hat.ndim != 2 or X_hat.shape[1] != X_scaled.shape[1]:
        raise ValueError(f"Unexpected model output shape: {X_hat.shape} (expected (N, {X_scaled.shape[1]}))")

    recon_error = np.mean((X_scaled - X_hat) ** 2, axis=1)

    # Percentile score [0, 100] (higher = better)
    conv_pct = pd.Series(-recon_error).rank(pct=True).to_numpy(dtype=float) * 100.0

    valid = np.isfinite(conv_pct)
    if valid.sum() == 0:
        raise ValueError("All conversion scores are NaN/Inf even after Colab model. Check X_scaled and model output.")

    # Priority threshold
    cut = np.percentile(conv_pct[valid], 100 - top_k)
    priority = np.where(conv_pct >= cut, f"Top {top_k}%", "Remaining")

    # ----------------------------
    # 5) Output
    # ----------------------------
    out = pd.DataFrame({
        "Customer ID": cust_id,
        "Reconstruction Error": recon_error,
        "Conversion Probability (%)": conv_pct,
        "Priority Group": priority,
    })
    return out

def main():
    df_raw = pd.read_csv(DEFAULT_INPUT_PATH)
    results = score_dataframe(df_raw, top_k=5)
    print("========== CNN SCORING SUMMARY ==========")
    print(results["Priority Group"].value_counts())
    print(results.head(10))


if __name__ == "__main__":
    main()
