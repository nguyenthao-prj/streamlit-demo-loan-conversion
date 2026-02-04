import pandas as pd
import numpy as np

# Đọc dữ liệu
df = pd.read_csv("data/credit_test.csv")

# Random dữ liệu (shuffle)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Chia thành 12 phần gần bằng nhau
splits = np.array_split(df, 12)

months = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]

# Lưu ra file
for split, name in zip(splits, months):
    split.to_csv(f"data/credit_test_{name}.csv", index=False)
