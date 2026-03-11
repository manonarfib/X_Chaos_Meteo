# MODEL_TYPE = "unet" 
# LEAD=1
# T=8
# MAX_LEAD=1
# WITHOUT_PRECIP=False
# BATCH_SIZE=16
# DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
# CKPT_PATH = 

def feature_permutation_for_one_sample(MODEL_TYPE, CKPT_PATH, DATASET_PATH, LEAD, T, MAX_LEAD, WITHOUT_PRECIP=False, BATCH_SIZE=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD, without_precip=WITHOUT_PRECIP, max_lead=MAX_LEAD)
    input_vars = list(test_dataset.X.coords["channel"].values)
    C_in = len(input_vars)

    model = build_model(MODEL_TYPE, C_in, T, device, max_lead=MAX_LEAD)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metric = mean_squared_error

    baseline_sum = 0.0
    C = 8, 33

    loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    all_importances = []   # (N_images, T*C)

    X,y = loader[idx]
    _, _, _, H, W = X.shape
    X_flat = X.reshape(T*C, -1).T
    y_flat = y.reshape(-1)

    imp, base = permutation_importance_batch(
        model, X_flat, y_flat, metric,
        T=T, C=C, H=H, W=W,
        batch_size_features=16,
        n_repeats=5
    )

    all_importances.append(imp)

    all_importances = np.stack(all_importances)   # shape = (N, T*C)

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
        f"explainability/features_permutation/figures/importance_per_variable_{MODEL_TYPE}_new.png",
        title="Permutation importance — aggregated per variable",
        top_k=20
    )

    imp_time = imp_tc.mean(axis=2)   # (N, T)

    mean_time = imp_time.mean(axis=0)
    std_time  = imp_time.std(axis=0)
    time_labels = [f"{(T-1-t)*6}h" for t in range(T)]

    time_labels = ["t-42h", "t-36h", "t-30h", "t-24h", "t-18h", "t-12h", "t-6h", "t"]

    save_lineplot_mean_std(
        mean_time,
        std_time,
        f"explainability/features_permutation/figures/importance_per_time_{MODEL_TYPE}_new.png",
        title="Permutation importance — aggregated per timestep",
        xlabel="Time",
        ylabel="ΔMSE",
        xtick_labels=time_labels
    )