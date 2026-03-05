# https://scikit-learn.org/stable/modules/permutation_importance.html
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
     
     
from sklearn.base import BaseEstimator
import torch
from torch.utils.data import DataLoader

class TorchRegressorWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            X_np = np.asarray(X, dtype=np.float32)
            X_torch = torch.from_numpy(X_np).to(device)
            return self.model(X_torch).cpu().numpy().ravel()


    def score(self, X, y):
        y_pred = self.predict(X)
        return -np.mean((y - y_pred) ** 2)  # ex: -MSE   
        
import matplotlib

from sklearn.inspection import permutation_importance
from sklearn.utils.fixes import parse_version

from models.utils.ERA5_dataset_from_local import ERA5Dataset


def plot_permutation_importance(clf, X, y, ax, scoring="neg_mean_squared_error"):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2, scoring=scoring)
    perm_sorted_idx = result.importances_mean.argsort()

    # `labels` argument in boxplot is deprecated in matplotlib 3.9 and has been
    # renamed to `tick_labels`. The following code handles this, but as a
    # scikit-learn user you probably can write simpler code by using `labels=...`
    # (matplotlib < 3.9) or `tick_labels=...` (matplotlib >= 3.9).
    tick_labels_parameter_name = (
        "tick_labels"
        if parse_version(matplotlib.__version__) >= parse_version("3.9")
        else "labels"
    )
    tick_labels_dict = {tick_labels_parameter_name: X.columns[perm_sorted_idx]}
    ax.boxplot(result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
    ax.axvline(x=0, color="k", linestyle="--")
    return ax

import numpy as np
import torch

def compute_permutation_importance(model, X, y, metric):
    """
    X: torch.Tensor ou np.ndarray, shape (n_samples, n_features)
    y: torch.Tensor ou np.ndarray, shape (n_samples,)
    metric: fonction(y_true, y_pred) -> float (plus petit = meilleur)
    """
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    # baseline
    y_pred = model(torch.from_numpy(X).float()).detach().numpy().ravel()
    baseline_score = metric(y, y_pred)

    for f in range(n_features):
        X_perm = X.copy()
        idx = np.random.permutation(n_samples)
        X_perm[:, f] = X_perm[idx, f]

        y_pred_perm = model(torch.from_numpy(X_perm).float()).detach().numpy().ravel()
        perm_score = metric(y, y_pred_perm)
        importances[f] = perm_score - baseline_score

    return importances, baseline_score



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.ConvLSTM.convlstm import PrecipConvLSTM

if __name__=='__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"

    train_dataset = ERA5Dataset(train_dataset_path, 8, 1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    input_vars = list(train_dataset.X.coords["channel"].values)
    C_in = len(input_vars)

    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=[32, 64],
        kernel_size=3,
    ).to(device)

    ckpt_path = "./features_importance/best_checkpoint_epoch2_batch1091.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {ckpt_path}")

    clf = TorchRegressorWrapper(model)
    
    batch = next(iter(train_loader))
    X, y = batch[0], batch[1]

    X_flat = X[0]            # supprime la dimension batch -> (8, 33, 149, 221)
    X_flat = X_flat.reshape(8*33, -1)  # (264, 149*221)
    X_flat = X_flat.T          # (149*221, 264) -> shape (n_samples, n_features)

    y_flat = y.reshape(-1)

    print(X_flat.shape) 
    print(y_flat.shape)
    breakpoint()


    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
    plot_permutation_importance(clf, X_flat, y_flat, ax1)
    ax1.set_xlabel("Increase in prediction error (ΔMSE)")
    fig.suptitle(
        "Permutation importances on multicollinear features (train set)"
    )
    _ = fig.tight_layout()
    breakpoint()

    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X_val, y_val,
                            n_repeats=30,
                            random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{diabetes.feature_names[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")
        
        
# Hierarchical clustering to handle collinearity

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
_ = fig.tight_layout()