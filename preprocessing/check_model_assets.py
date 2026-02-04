import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

PREPROCESS_PATH = "artifacts/preprocess.pkl"
MODEL_PATH = "assets/best_model.keras"
DATA_PATH = "data/reference.csv"

pre = joblib.load(PREPROCESS_PATH)
model = load_model(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
df_clean = pre.transform(df)

X = df_clean[pre.final_feature_list_].astype(float).values
X_scaled = pre.scaler_.transform(X)
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

y = model.predict(X_cnn[:32], verbose=0)
y = np.asarray(y)

print("Model input_shape:", getattr(model, "input_shape", None))
print("Model output_shape:", getattr(model, "output_shape", None))
print("Output batch shape:", y.shape)
print("Has NaN:", np.isnan(y).any())
print("NaN count:", int(np.isnan(y).sum()))
