import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# -------------------------
# Paths
# -------------------------
PREPROCESS_PATH = "artifacts/preprocess.pkl"
REFERENCE_PATH = 'https://drive.google.com/uc?id=1xmV4epAeF1hHN-8lvpoOVYvZkTOLCHqH'

# -------------------------
# Load preprocessor + data
# -------------------------
pre = joblib.load(PREPROCESS_PATH)

df = pd.read_csv(REFERENCE_PATH)
df_clean = pre.transform(df)

feature_cols = pre.final_feature_list_
X = df_clean[feature_cols].astype(float).values

# Scale
X_scaled = pre.scaler_.transform(X)

# Reshape for CNN
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

print("X_cnn shape:", X_cnn.shape)  # EXPECT: (N, 36, 1)

# -------------------------
# Build CNN (baseline)
# -------------------------
n_features = X_cnn.shape[1]

model = Sequential([
    Conv1D(filters=16, kernel_size=3, activation="relu",
           input_shape=(n_features, 1)),
    Conv1D(filters=8, kernel_size=3, activation="relu"),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(n_features)   # reconstruction
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"
)

# -------------------------
# Inspect
# -------------------------
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------
# Train
# -------------------------
callbacks = [
    EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(
    filepath="artifacts/best_model.keras",
    monitor="loss",
    save_best_only=True
)

]

history = model.fit(
    X_cnn, X_scaled,
    epochs=10,
    batch_size=32,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)

print("✅ Training done. Best model saved to artifacts/best_model.pkl")
