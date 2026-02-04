import joblib
import pandas as pd
import numpy as np
from tensorflow import keras

PREPROCESS_PATH = "artifacts/preprocess.pkl"
INPUT_PATH = "data/reference.csv"       # hoặc file inference_sample.csv
MODEL_PATH = "assets/best_model.keras"

def main():
    pre = joblib.load(PREPROCESS_PATH)
    model = keras.models.load_model(MODEL_PATH)

    df_raw = pd.read_csv(INPUT_PATH)
    df_clean = pre.transform(df_raw)

    X = df_clean[pre.final_feature_list_].astype(float).values
    X_scaled = pre.scaler_.transform(X)

    print("========== CNN MODEL INSPECTION ==========")
    print("Model input_shape:", model.input_shape)
    print("Model output_shape:", model.output_shape)
    print("X_scaled shape:", X_scaled.shape)

    # Try common reshape patterns
    candidates = []

    # 2D (n, features)
    candidates.append(("2D", X_scaled))

    # 3D (n, features, 1)
    candidates.append(("3D_feat1", X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)))

    # 3D (n, 1, features)
    candidates.append(("3D_1feat", X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])))

    for name, Xin in candidates:
        try:
            # predict small batch
            yhat = model.predict(Xin[:8], verbose=0)
            yhat = np.asarray(yhat)
            print(f"\n✅ Predict OK with {name}")
            print("Input batch shape:", Xin[:8].shape)
            print("Output shape:", yhat.shape)

            # Compute reconstruction MSE per row (handle 2D or 3D)
            if Xin[:8].ndim == 2:
                mse = np.mean((Xin[:8] - yhat) ** 2, axis=1)
            else:
                mse = np.mean((Xin[:8] - yhat) ** 2, axis=(1, 2))

            print("MSE sample:", mse[:5])
        except Exception as e:
            print(f"\n❌ Predict FAILED with {name}: {type(e).__name__}: {e}")

    print("==========================================")

if __name__ == "__main__":
    main()
