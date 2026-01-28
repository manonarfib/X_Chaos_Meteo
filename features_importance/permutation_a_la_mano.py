# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from models.utils.ERA5_dataset_from_local import ERA5Dataset
# from models.ConvLSTM.convlstm import PrecipConvLSTM  # adapter à ton modèle


# def reshape_for_convlstm(X_flat, B=1, T=8, C=33, H=149, W=221):
#     """
#     Transforme X_flat (pixels, features) en X_conv (B, T, C, H, W)
    
#     X_flat: np.ndarray ou torch.Tensor, shape = (H*W, T*C)
#     Retour: torch.Tensor shape = (B, T, C, H, W)
#     """
#     X_tensor = torch.tensor(X_flat, dtype=torch.float32)
#     # reshape (pixels, T*C) -> (T*C, H*W)
#     X_tensor = X_tensor.T
#     X_tensor = X_tensor.reshape(T, C, H, W)
#     # ajouter batch dim
#     X_tensor = X_tensor.unsqueeze(0)  # (1, T, C, H, W)
#     return X_tensor


# if __name__=="__main__":
#     # ------------------------------
#     # CONFIG
#     # ------------------------------
#     train_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
#     checkpoint_path = "./features_importance/best_checkpoint_epoch2_batch1091.pt"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ------------------------------
#     # CHARGEMENT DU DATASET
#     # ------------------------------
#     train_dataset = ERA5Dataset(train_dataset_path, 8, 1)
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
#     batch = next(iter(train_loader))
#     X, y = batch[0], batch[1]

#     print("Device:", device)
#     print("Original X shape:", X.shape)

#     # ------------------------------
#     # FLATTEN X ET Y
#     # ------------------------------
#     # X: (1, 8, 33, H, W)
#     batch, time_steps, n_channels, H, W = X.shape
#     n_features = time_steps * n_channels
#     n_samples = H * W

#     # Flatten en (n_samples, n_features)
#     X_flat = X[0].reshape(time_steps * n_channels, -1).T  # (H*W, 8*33)
#     y_flat = y.reshape(-1)  # (H*W,)

#     print("Flattened X shape:", X_flat.shape)
#     print("Flattened y shape:", y_flat.shape)

#     # ------------------------------
#     # CHARGEMENT DU MODELE
#     # ------------------------------
#     input_vars = list(train_dataset.X.coords["channel"].values)
#     C_in = len(input_vars)
#     model = PrecipConvLSTM(
#         input_channels=C_in,
#         hidden_channels=[32, 64],
#         kernel_size=3,
#     ).to(device)
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"])  # ou ckpt si direct
#     model.to(device)
#     model.eval()

#     # Fonction helper pour prédiction
#     def predict_numpy(model, X_flat):
#         """X_input: np.ndarray shape (n_samples, n_features)"""
#         with torch.no_grad():
#             X_5d = reshape_for_convlstm(X_flat)  # remet la forme 5D
#             y_pred = model(X_5d)
#             return y_pred.cpu().numpy().ravel()

#     # ------------------------------
#     # PERMUTATION IMPORTANCE
#     # ------------------------------
#     def permutation_importance(model, X, y, metric):
#         """
#         Calcul de l'importance par permutation manuelle.
#         X: (n_samples, n_features) np.ndarray
#         y: (n_samples,) np.ndarray
#         metric: fonction(y_true, y_pred) -> float
#         """
#         X = np.array(X)
#         y = np.array(y)
#         n_samples, n_features = X.shape
#         importances = np.zeros(n_features)
#         # importances = np.zeros(5)

#         # baseline
#         y_pred = predict_numpy(model, X)
#         baseline_score = metric(y, y_pred)
#         print(f"baseline score : {baseline_score}")

#         for f in range(n_features):
#         # for f in range(5):
#             X_perm = X.copy()
#             idx = np.random.permutation(n_samples)
#             X_perm[:, f] = X_perm[idx, f]

#             y_pred_perm = predict_numpy(model, X_perm)
#             perm_score = metric(y, y_pred_perm)
#             importances[f] = perm_score - baseline_score  # ΔMSE

#             if (f+1) % 20 == 0 or f == n_features-1:
#                 print(f"Processed feature {f+1}/{n_features}")

#         return importances, baseline_score
    
#     def permutation_importance_fast(model, X_flat, y_flat, metric, T=8, C=33, H=149, W=221):
#         """
#         Calcul plus rapide des importances par permutation pour ConvLSTM.
#         """
#         model.eval()
#         n_samples, n_features = X_flat.shape
#         importances = np.zeros(n_features)

#         # Convertir X_flat en torch 5D
#         X_5d = reshape_for_convlstm(X_flat)  # (1, T, C, H, W)
#         X_5d = X_5d.to(next(model.parameters()).device)
#         y_tensor = torch.tensor(y_flat, dtype=torch.float32).to(X_5d.device)

#         # Baseline
#         with torch.no_grad():
#             y_pred = model(X_5d)
#             baseline = metric(y_flat, y_pred.cpu().numpy().ravel())
#         print(f"Baseline score: {baseline}")

#         # Pour chaque feature
#         for f in range(n_features):
#             # Indices de t et c correspondant à la feature f
#             t = f // C
#             c = f % C

#             # Permutation
#             X_perm = X_5d.clone()
#             idx = torch.randperm(H*W)
#             # flatten H*W pour remplacer la feature
#             X_perm[0, t, c] = X_perm[0, t, c].flatten()[idx].reshape(H, W)

#             # Forward
#             with torch.no_grad():
#                 y_pred_perm = model(X_perm)
#                 perm_score = metric(y_flat, y_pred_perm.cpu().numpy().ravel())
#             importances[f] = perm_score - baseline

#             if (f+1) % 20 == 0 or f == n_features-1:
#                 print(f"Processed feature {f+1}/{n_features}")

#         return importances, baseline
    
#     def permutation_importance_batch(model, X_flat, y_flat, metric, T=8, C=33, H=149, W=221, batch_size_features=16):
#         """
#         Calcul des importances par permutation avec batch de features.
#         """
#         model.eval()
#         n_samples, n_features = X_flat.shape
#         importances = np.zeros(n_features)

#         # Convertir X_flat en torch 5D
#         X_5d = reshape_for_convlstm(X_flat).to(next(model.parameters()).device)
#         y_tensor = torch.tensor(y_flat, dtype=torch.float32).to(X_5d.device)

#         # Baseline
#         with torch.no_grad():
#             y_pred = model(X_5d)
#             baseline = metric(y_flat, y_pred.cpu().numpy().ravel())
#         print(f"Baseline score: {baseline}")

#         # Boucle par batch
#         for start in range(0, n_features, batch_size_features):
#             end = min(start + batch_size_features, n_features)
#             batch_feats = end - start

#             # Créer un batch de X permutés pour ces features
#             X_perm_batch = X_5d.repeat(batch_feats, 1, 1, 1, 1)  # (batch_feats, T, C, H, W)

#             for i, f in enumerate(range(start, end)):
#                 t = f // C
#                 c = f % C
#                 idx = torch.randperm(H*W)
#                 X_perm_batch[i, t, c] = X_perm_batch[i, t, c].flatten()[idx].reshape(H, W)

#             # Forward batch
#             print(X_perm_batch.shape)
#             with torch.no_grad():
#                 y_pred_perm = model(X_perm_batch)  # shape (batch_feats, H, W)
#                 y_pred_perm = y_pred_perm.view(batch_feats, -1).cpu().numpy()  # flatten pixels
            
#             # Calcul ΔMSE
#             for i, f in enumerate(range(start, end)):
#                 importances[f] = metric(y_flat, y_pred_perm[i]) - baseline

#             print(f"Processed features {start+1}-{end}/{n_features}")

#         return importances, baseline


#     # Définir la métrique
#     metric = mean_squared_error

#     importances, baseline = permutation_importance_batch(model, X_flat, y_flat, metric,
#                                                     T=time_steps, C=n_channels, H=H, W=W)

#     # importances, baseline = permutation_importance(model, X_flat, y_flat, metric)

#     print("Baseline MSE:", baseline)
#     print("Permutation importances shape:", importances.shape)

#     # ------------------------------
#     # VISUALISATION
#     # ------------------------------
    
#     feature_labels = []
#     for t in range(time_steps):
#         time=(7-t)*6
#         for c in range(n_channels):
#             feature_labels.append(f"{input_vars[c]}_{time}h")
    
#     plt.figure(figsize=(12, 20))
#     print("figsize = 20")
#     plt.barh(range(n_features), importances)
#     # plt.barh(range(5), importances)
#     plt.xlabel("ΔMSE after permutation")
#     plt.yticks(range(n_features), feature_labels, fontsize=6)
#     plt.ylabel("Feature index (time x channel)")
#     plt.title("Permutation Importance per feature")
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig("permutation_importance.png")


#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    # X_tensor = X_flat.detach.clone()
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
    # y_tensor = torch.tensor(y_flat, dtype=torch.float32).to(X_5d.device)

    # Baseline
    with torch.no_grad():
        y_pred = model(X_5d)
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


if __name__=="__main__":
    # ------------------------------
    # CONFIG
    # ------------------------------
    train_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
    test_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
    checkpoint_path = "./features_importance/best_checkpoint_epoch2_batch1091.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # DATASET
    # ------------------------------
    train_dataset = ERA5Dataset(train_dataset_path, 8, 1)
    test_dataset = ERA5Dataset(test_dataset_path, 8, 1)
    batch = next(iter(DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)))
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
    # importances, baseline = permutation_importance_batch(
    #     model, X_flat, y_flat, metric,
    #     T=time_steps, C=n_channels, H=H, W=W,
    #     batch_size_features=16,
    #     n_repeats=5
    # )
    
    importances_sum = np.zeros(n_features)
    baseline_sum = 0.0
    n_images = len(test_dataset)
    T,C = 8, 33

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, batch in enumerate(loader):
        print(idx)
        breakpoint()
        if idx==5:
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
        importances_sum += imp
        baseline_sum += base


    importances_avg = importances_sum / 5
    baseline_avg = baseline_sum / 5
    
    print("Baseline MSE:", baseline_avg)
    print("Permutation importances shape:", importances_avg.shape)
    
    # # Charger tout le test set et empiler
    # X_list = []
    # y_list = []

    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # i=0
    # for batch in test_loader:
    #     print(i)
    #     X_batch, y_batch = batch[0], batch[1]  # (1, T, C, H, W)
    #     X_list.append(X_batch[0])              # enlever la batch dim
    #     y_list.append(y_batch[0])              # enlever la batch dim
    #     i+=1
    #     if i==100:
    #         break
    # print("Xlist done")
    # # Empiler tous les samples
    # X_all = torch.cat(X_list, dim=1)  # concat sur H*W -> (T, C, H_total, W)
    # y_all = torch.cat(y_list, dim=1)  # concat sur H*W -> (H_total*W,)

    # # Flatten pour permutation_importance_batch
    # T, C, H_total, W = X_all.shape
    # n_features = T * C
    # X_flat_all = X_all.reshape(T*C, -1).T  # (n_samples_total, n_features)
    # y_flat_all = y_all.reshape(-1)

    # print("Shape X_flat_all:", X_flat_all.shape)
    # print("Shape y_flat_all:", y_flat_all.shape)

    # # Calcul des importances sur tout le test set
    # importances, baseline = permutation_importance_batch(
    #     model, X_flat_all, y_flat_all, metric,
    #     T=T, C=C, H=H_total, W=W,
    #     batch_size_features=16,
    #     n_repeats=5
    # )
    # print("Baseline MSE:", baseline)
    # print("Permutation importances shape:", importances.shape)

    # ------------------------------
    # VISUALISATION TRIÉE
    # ------------------------------
    feature_labels = [f"{input_vars[c]}_{(7-t)*6}h" for t in range(time_steps) for c in range(n_channels)]
    sorted_idx = np.argsort(importances_avg)[::-1]  # décroissant
    importances_sorted = importances_avg[sorted_idx]
    feature_labels_sorted = [feature_labels[i] for i in sorted_idx]

    plt.figure(figsize=(12, 25))
    plt.barh(range(n_features), importances_sorted)
    plt.yticks(range(n_features), feature_labels_sorted, fontsize=6)
    plt.gca().invert_yaxis()  # mettre les plus importantes en haut
    plt.xlabel("ΔMSE after permutation")
    plt.ylabel("Feature")
    plt.title("Permutation Importance per feature")
    plt.tight_layout()
    plt.savefig("permutation_importance_sorted.png")
