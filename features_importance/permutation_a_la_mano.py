import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM


def reshape_for_convlstm(X_flat, B=1, T=8, C=33, H=149, W=221):
    """
    Transforme X_flat (pixels, features) en X_conv (B, T, C, H, W)
    """
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    X_tensor = X_tensor.T
    X_tensor = X_tensor.reshape(T, C, H, W)
    X_tensor = X_tensor.unsqueeze(0)
    return X_tensor


def permutation_importance_batch(model, X_flat, y_flat, metric, T=8, C=33, H=149, W=221,
                                 batch_size_features=16, n_repeats=5):
    """
    Calcul des importances par permutation en batch, avec plusieurs répétitions.
    """
    model.eval()
    n_samples, n_features = X_flat.shape
    importances = np.zeros(n_features)

    # Convertir X_flat en torch 5D
    X_5d = reshape_for_convlstm(X_flat).to(next(model.parameters()).device)

    # Baseline
    with torch.no_grad():
        y_pred = model(X_5d)
        y_pred = torch.clamp(y_pred, min=0.0)
        baseline = metric(y_flat, y_pred.cpu().numpy().ravel())
    print(f"Baseline score: {baseline}")

    for repeat in range(n_repeats):
        print(f"Repeat {repeat+1}/{n_repeats}")
        for start in range(0, n_features, batch_size_features):
            end = min(start + batch_size_features, n_features)
            batch_feats = end - start

            X_perm_batch = X_5d.repeat(batch_feats, 1, 1, 1, 1)

            for i, f in enumerate(range(start, end)):
                t = f // C
                c = f % C
                idx = torch.randperm(H * W)
                X_perm_batch[i, t, c] = X_perm_batch[i, t, c].flatten()[idx].reshape(H, W)

            with torch.no_grad():
                y_pred_perm = model(X_perm_batch)
                y_pred_perm = y_pred_perm.view(batch_feats, -1).cpu().numpy()

            for i, f in enumerate(range(start, end)):
                importances[f] += metric(y_flat, y_pred_perm[i]) - baseline

    # Moyenne sur les répétitions
    importances /= n_repeats

    return importances, baseline

import os
def save_barplot_mean_std(mean_vals, std_vals, labels, out_path, title="", top_k=15): 
    mean_vals = np.asarray(mean_vals) 
    std_vals = np.asarray(std_vals) 
    idx = np.argsort(-mean_vals)[:top_k] 
    m = mean_vals[idx][::-1] 
    s = std_vals[idx][::-1] 
    l = [labels[i] for i in idx][::-1] 
    fig, ax = plt.subplots(figsize=(9, 5)) 
    ax.barh(range(len(m)), m, xerr=s) 
    ax.set_yticks(range(len(m))) 
    ax.set_yticklabels(l) 
    ax.set_title(title) 
    ax.set_xlabel("Importance (mean ± std over samples)") 
    plt.tight_layout() 
    os.makedirs(os.path.dirname(out_path), exist_ok=True) 
    plt.savefig(out_path, dpi=200, bbox_inches="tight") 
    plt.close(fig) 
    print(f"[FIG] Saved: {out_path}") 
    
def save_lineplot_mean_std(mean_vals, std_vals, out_path, title="", xlabel="t index", ylabel="Importance"): 
    mean_vals = np.asarray(mean_vals) 
    std_vals = np.asarray(std_vals) 
    x = np.arange(len(mean_vals)) 
    fig, ax = plt.subplots(figsize=(8, 4)) 
    ax.plot(x, mean_vals, marker="o") 
    ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, alpha=0.25) 
    ax.set_title(title) 
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.grid(True, linestyle="--", linewidth=0.5) 
    plt.tight_layout() 
    os.makedirs(os.path.dirname(out_path), exist_ok=True) 
    plt.savefig(out_path, dpi=200, bbox_inches="tight") 
    plt.close(fig) 
    print(f"[FIG] Saved: {out_path}")


if __name__=="__main__":
    # ------------------------------
    # CONFIG
    # ------------------------------
    train_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
    test_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
    checkpoint_path = "epoch3_full.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # DATASET
    # ------------------------------
    train_dataset = ERA5Dataset(train_dataset_path, 8, 1)
    test_dataset = ERA5Dataset(test_dataset_path, 8, 1)
    batch = next(iter(DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)))
    X, y = batch[0], batch[1]
    batch, time_steps, n_channels, H, W = X.shape
    n_features = time_steps * n_channels
    n_samples = H * W
    X_flat = X[0].reshape(time_steps * n_channels, -1).T
    y_flat = y.reshape(-1)

    # ------------------------------
    # MODELE
    # ------------------------------
    input_vars = list(train_dataset.X.coords["channel"].values)
    model = PrecipConvLSTM(input_channels=len(input_vars), hidden_channels=[32, 64], kernel_size=3)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ------------------------------
    # PERMUTATION IMPORTANCE
    # ------------------------------
    metric = mean_squared_error
    
    importances_sum = np.zeros(n_features)
    baseline_sum = 0.0
    n_images = len(test_dataset)
    T,C = 8, 33

    loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    # for idx, batch in enumerate(loader):
    #     print(idx)
    #     if idx==100:
    #         break
    #     X, y = batch[0], batch[1]
    #     X_flat = X.reshape(T*C, -1).T
    #     y_flat = y.reshape(-1)
        
    #     imp, base = permutation_importance_batch(
    #         model, X_flat, y_flat, metric,
    #         T=T, C=C, H=H, W=W,
    #         batch_size_features=16,
    #         n_repeats=5
    #     )
    #     importances_sum += imp
    #     baseline_sum += base
    
    all_importances = []   # (N_images, T*C)
    all_baselines = []

    for idx, batch in enumerate(loader):
        print(idx)
        if idx == 100:
            break

        X, y = batch[0], batch[1]
        X_flat = X.reshape(T*C, -1).T
        y_flat = y.reshape(-1)

        imp, base = permutation_importance_batch(
            model, X_flat, y_flat, metric,
            T=T, C=C, H=H, W=W,
            batch_size_features=16,
            n_repeats=5
        )

        all_importances.append(imp)
        all_baselines.append(base)

    all_importances = np.stack(all_importances)   # shape = (N, T*C)
    all_baselines = np.array(all_baselines)
    
    np.savez(
        "permutation_importances_to_stack_time_and_var.npz",
        importances_sorted=all_importances
    )

    N = all_importances.shape[0]
    imp_tc = all_importances.reshape(N, T, C)
    imp_var = imp_tc.mean(axis=1)   # (N, C)
    
    mean_var = imp_var.mean(axis=0)
    std_var  = imp_var.std(axis=0)

    labels_var = input_vars
    
    save_barplot_mean_std(
        mean_var,
        std_var,
        labels_var,
        "figures/importance_per_variable.png",
        title="Permutation importance — aggregated per variable",
        top_k=15
    )

    imp_time = imp_tc.mean(axis=2)   # (N, T)

    mean_time = imp_time.mean(axis=0)
    std_time  = imp_time.std(axis=0)
    time_labels = [f"{(T-1-t)*6}h" for t in range(T)]

    save_lineplot_mean_std(
        mean_time,
        std_time,
        "figures/importance_per_time.png",
        title="Permutation importance — aggregated per timestep",
        xlabel="Time index (lag)",
        ylabel="ΔMSE"
    )



    # importances_avg = importances_sum / 100
    # baseline_avg = baseline_sum / 100
    
    # print("Baseline MSE:", baseline_avg)
    # print("Permutation importances shape:", importances_avg.shape)

    # ------------------------------
    # VISUALISATION TRIÉE
    # ------------------------------
    # feature_labels = [f"{input_vars[c]}_{(7-t)*6}h" for t in range(time_steps) for c in range(n_channels)]
    # sorted_idx = np.argsort(importances_avg)[::-1]  # décroissant
    # importances_sorted = importances_avg[sorted_idx]
    # feature_labels_sorted = [feature_labels[i] for i in sorted_idx]
    
    # np.savez(
    #     "permutation_importance.npz",
    #     importances_sorted=importances_sorted,
    #     feature_labels_sorted=np.array(feature_labels_sorted),
    #     sorted_idx=sorted_idx,
    # )
    
    # data = np.load("features_importance/permutation_importance.npz", allow_pickle=True)
    # importances_sorted = data["importances_sorted"]
    # feature_labels_sorted = data["feature_labels_sorted"]

    # plt.figure(figsize=(12, 25))
    # plt.barh(range(n_features), importances_sorted)
    # plt.yticks(range(n_features), feature_labels_sorted, fontsize=6)
    # plt.gca().invert_yaxis()  # mettre les plus importantes en haut
    # plt.xlabel("ΔMSE after permutation")
    # plt.ylabel("Feature")
    # plt.title("Permutation Importance per feature")
    # plt.tight_layout()
    # plt.savefig("permutation_importance_sorted.png")
    
    # top_k = 40  # nombre de features affichées
    # importances_top = importances_sorted[:top_k]
    # labels_top = feature_labels_sorted[:top_k]

    # fig_height = 0.35 * top_k + 1
    # plt.figure(figsize=(10, fig_height))

    # y_pos = np.arange(top_k)
    # plt.barh(y_pos, importances_top)

    # plt.yticks(y_pos, labels_top, fontsize=8)
    # plt.gca().invert_yaxis()

    # plt.xlabel("ΔMSE after permutation")
    # plt.title(f"Top {top_k} — Permutation Importance")

    # # grille légère pour lecture
    # plt.grid(axis="x", linestyle="--", alpha=0.4)

    # # réduction forte des marges blanches
    # plt.margins(x=0.01, y=0.01)
    # plt.subplots_adjust(left=0.42, right=0.98, top=0.97, bottom=0.03)

    # plt.savefig("permutation_importance_topk.png", dpi=200, bbox_inches="tight")
    # plt.close()

