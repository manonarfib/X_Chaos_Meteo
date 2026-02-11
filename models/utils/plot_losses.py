import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path_train = "checkpoints/run_141848/train_log.csv"
csv_path_val   = "checkpoints/run_141848/validation_log.csv"

df_train = pd.read_csv(csv_path_train)
df_val   = pd.read_csv(csv_path_val)

tensor_re = re.compile(r"tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def parse_loss(x):
    # Already numeric
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)

    # Missing values
    if pd.isna(x):
        return np.nan

    # String cases
    s = str(x).strip()

    # tensor(...) format
    m = tensor_re.search(s)
    if m:
        return float(m.group(1))

    # plain numeric string (e.g., "2.34")
    return float(s)

df_train["loss"] = df_train["loss"].apply(parse_loss)
df_val["loss"]   = df_val["loss"].apply(parse_loss)

# If you want to avoid hardcoding 3470, infer "batches per epoch" from train log
batches_per_epoch = int(df_train["batch_idx"].max()) + 1

x_train = (df_train["epoch"] - 1) * batches_per_epoch + df_train["batch_idx"]
x_val   = (df_val["epoch"]   - 1) * batches_per_epoch + df_val["batch_idx"]

plt.figure(figsize=(8, 5))
plt.plot(x_train, df_train["loss"], label="Train loss")
plt.plot(x_val,   df_val["loss"],   label="Validation loss")

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig('/'.join(csv_path_train.split('/')[:-1])+"/loss_curve.png", dpi=150)
plt.show()
