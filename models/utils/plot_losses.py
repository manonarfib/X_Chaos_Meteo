import pandas as pd
import matplotlib.pyplot as plt

csv_path_train = "checkpoints/train_log.csv"
csv_path_val = "checkpoints/validation_log.csv"
df_train = pd.read_csv(csv_path_train)
df_val = pd.read_csv(csv_path_val)

def parse_loss(x):
    # x = "tensor(2.2915, device='cuda:0', grad_fn=<...>)"
    return float(x.split("tensor(")[1].split(",")[0])

df_train["loss"] = df_train["loss"].apply(parse_loss)
df_val["loss"] = df_val["loss"].apply(parse_loss)

plt.figure(figsize=(8, 5))
plt.plot((df_train["epoch"]-1)*3470+df_train["batch_idx"], df_train["loss"], label="Train loss", marker="o")
plt.plot((df_val["epoch"]-1)*3470+df_val["batch_idx"], df_val["loss"], label="Validation loss", marker="s")

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("checkpoints/loss_curve.png", dpi=150)
plt.show()
