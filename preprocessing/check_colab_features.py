import json
import joblib
import pandas as pd

PREPROCESS_PATH = "artifacts/preprocess.pkl"
FEATURES_PATH = "assets/colab/feature_columns.json"
DATA_PATH = "data/reference.csv"

pre = joblib.load(PREPROCESS_PATH)
df = pd.read_csv(DATA_PATH)
df_clean = pre.transform(df)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    colab_features = json.load(f)

missing = [c for c in colab_features if c not in df_clean.columns]
extra = [c for c in df_clean.columns if c in colab_features]

print("df_clean columns:", len(df_clean.columns))
print("colab_features:", len(colab_features))
print("present colab features:", len(extra))
print("missing colab features:", len(missing))

if missing:
    print("\nMISSING FEATURES:")
    for c in missing:
        print("-", c)
else:
    print("\nAll colab features are present ✅")
