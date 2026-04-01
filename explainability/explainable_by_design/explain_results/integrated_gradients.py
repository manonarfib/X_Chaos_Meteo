import torch
import os
import numpy as np

from models.utils.ERA5_dataset_from_local import ERA5Dataset

import explainability.integrated_gradients.integrated_gradients_over_multi_samples as ig_utils

# ============================================================
# USER CONFIG (no argparse, no inference)
# ============================================================

MODEL_TYPE = "with_reg_on_vars" #classic or with_reg_on_vars
LOSS_NAME  = "MSE"       # purely for output folder naming
CKPT_PATH  = "checkpoints/weather_cbm/exp_reg_on_vars/best_checkpoint_epoch4_batch545.pt"
N_CONCEPTS=6

DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
SAMPLE_IDX = 250
T = 8
LEAD = 1

# Aggregate attribution settings
DO_AGG = True
N_SAMPLES_AGG = 50
SEED = 0

# IG settings
METHOD = "ig"            # "ig" or "gradxinput"
IG_STEPS = 30
BASELINE_MODE = "zeros"  # "zeros" or "mean_over_space_time"
REGION_QUANTILE = 0.90

# Viz settings
T_VIEW = 7
CONTOUR_Q = 0.95
TOP_K_VARS = 5
# ============================================================

def build_model(model_type: str, C_in: int, T: int, device: torch.device) -> torch.nn.Module:
    model_type = model_type.lower().strip()
    if model_type == "classic":
        from explainability.explainable_by_design.WeatherCBM import WeatherCBM
        return WeatherCBM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3,
            n_concepts=N_CONCEPTS
        ).to(device)
    elif model_type=="with_reg_on_vars":
        from explainability.explainable_by_design.WeatherCBM_with_reg_on_vars import WeatherCBM
        return WeatherCBM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3,
            n_concepts=N_CONCEPTS
        ).to(device)
    else :
        raise ValueError(f"Unknown model_type={model_type}")
    

def integrated_gradients(model, x, baseline, steps=30, target="region_sum", region_mask=None):
    """
    x: (B,T,C,H,W)
    baseline: (B,T,C,H,W)
    region_mask: (B,1,H,W) if target="region_sum"
    returns attr: (B,T,C,H,W)
    """
    assert x.shape == baseline.shape
    B, T, C, H, W = x.shape

    if target == "region_sum" and region_mask is None:
        region_mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)

    x = x.float()
    baseline = baseline.float()

    with torch.no_grad():
        y_hat, alpha = model(x)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)
    K = alpha.shape[1]

    total_grad = torch.zeros((B, K, T, C, H, W), device=x.device)

    for s in range(1, steps + 1):
        a = s / steps
        x_alpha = baseline + a * (x - baseline)
        x_alpha.requires_grad_(True)

        y_hat, alpha = model(x_alpha)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)
  
        grads_step = []

        for k in range(K):
            if target == "mean":
                S_k = alpha[:, k].mean()
            elif target == "region_sum":
                S_k = alpha[:, k].sum()
            else:
                raise ValueError("Unknown target")

            model.zero_grad(set_to_none=True)
            if x_alpha.grad is not None:
                x_alpha.grad.zero_()

            S_k.backward(retain_graph=True)

            grads_step.append(x_alpha.grad.detach().clone())

        grads_step = torch.stack(grads_step, dim=1)  # (B,K,T,C,H,W)
        total_grad += grads_step
        
    avg_grad = total_grad / float(steps)
    attr = (x - baseline).unsqueeze(1) * avg_grad
    
    return attr # (B,K,T,C,H,W)


def grad_x_input(model, x, target="region_sum", region_mask=None):
    """
    Faster alternative (1 backward). Useful for quick iteration.
    x: (B,T,C,H,W)
    returns attr: (B,T,C,H,W)
    """
    B, T, C, H, W = x.shape
    if target == "region_sum" and region_mask is None:
        region_mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)

    x_req = x.clone().detach().float().requires_grad_(True)
    y_hat, alpha = model(x_req)  # alpha: (B,K,H,W)
    K = alpha.shape[1]

    grads = []
    for k in range(K):
        if target == "mean":
            S_k = alpha[:, k].mean()
        elif target == "region_sum":
            S_k = alpha[:, k].sum()
        else:
            raise ValueError(f"Unknown spatial_agg: {target}")
        
        model.zero_grad(set_to_none=True)
        if x_req.grad is not None:
            x_req.grad.zero_()

        S_k.backward(retain_graph=True)

        grads.append(x_req.grad.detach().clone())

    grads = torch.stack(grads, dim=1)

    attr = grads * x_req.detach().unsqueeze(1)

    return attr


def compute_importance_over_random_samples(
    model,
    dataset,
    input_vars,
    device,
    n_samples=100,
    seed=0,
    method="ig",
    steps=30,
    baseline_mode="zeros",
    region_quantile=0.90,
):
    """
    Returns:
      var_mean, var_std  (C,)
      time_mean, time_std (T,)
      chosen_indices (n_samples,)
    """
    rng = np.random.default_rng(seed)
    N = len(dataset)
    chosen = rng.choice(N, size=min(n_samples, N), replace=False)

    C = len(input_vars)
    T = dataset.T if hasattr(dataset, "T") else None  # fallback
    # We'll infer T from first sample
    X0, *_ = dataset[int(chosen[0])]
    T = X0.shape[0]

    var_all = []
    time_all = []

    model.eval()

    for k, idx in enumerate(chosen, start=1):
        X, y, *_ = dataset[int(idx)]
        X = X.unsqueeze(0).to(device).float()  # (1,T,C,H,W)

        # predict -> define region mask from prediction
        with torch.no_grad():
            y_hat, _ = model(X)
            if y_hat.dim() == 3:
                y_hat = y_hat.unsqueeze(1)  # (1,1,H,W)

            pred_map = y_hat[0, 0]
            thresh = torch.quantile(pred_map.flatten(), region_quantile)
            region = (pred_map >= thresh).float()
            region_mask = region.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # attribution
        if method == "ig":
            baseline = ig_utils.make_baseline(X, mode=baseline_mode)
            attr = integrated_gradients(
                model=model,
                x=X,
                baseline=baseline,
                steps=steps,
                target="region_sum",
                region_mask=region_mask,
            )
        elif method == "gradxinput":
            attr = grad_x_input(
                model=model,
                x=X,
                target="region_sum",
                region_mask=region_mask,
            )
        else:
            raise ValueError("method must be 'ig' or 'gradxinput'")

        attr_abs = attr.abs()  # (1,T,C,H,W)

        # var importance: sum over T,H,W -> (C,)
        var_imp = attr_abs.sum(dim=(2, 4, 5))[0].detach().cpu().numpy()
        # time importance: sum over C,H,W -> (T,)
        time_imp = attr_abs.sum(dim=(3, 4, 5))[0].detach().cpu().numpy()

        var_all.append(var_imp)
        time_all.append(time_imp)

        if k % 10 == 0:
            print(f"[AGG] {k}/{len(chosen)} samples processed")

    var_all = np.stack(var_all, axis=0)   # (N,K,C)
    time_all = np.stack(time_all, axis=0) # (N,K,T)

    return (
        var_all.mean(axis=0), var_all.std(axis=0),
        time_all.mean(axis=0), time_all.std(axis=0),
        chosen
    )

# ----------------------------
# Main
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_type = MODEL_TYPE.lower().strip()

    # ---- data ----
    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)
    print("C_in:", C_in)

    # ---- model ----
    model = build_model(model_type, C_in=C_in, T=T, device=device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model | loss={LOSS_NAME} | ckpt={CKPT_PATH}")

    # ---- output dir ----
    lead_str = ig_utils.lead_to_str(LEAD)
    model_tag = model_type
    loss_tag = LOSS_NAME.lower()
    out_dir = f"explainability/explainable_by_design/explain_results/ig_outputs/{model_tag}"
    os.makedirs(out_dir, exist_ok=True)
    print("[OUT_DIR]", out_dir)

    # ---- aggregate importance ----
    if DO_AGG:
        agg_dir = os.path.join(out_dir, f"aggregate_{METHOD}_{N_SAMPLES_AGG}")
        os.makedirs(agg_dir, exist_ok=True)

        v_mean, v_std, t_mean, t_std, chosen_idx = compute_importance_over_random_samples(
            model=model,
            dataset=dataset,
            input_vars=input_vars,
            device=device,
            n_samples=N_SAMPLES_AGG,
            seed=SEED,
            method=METHOD,
            steps=IG_STEPS,
            baseline_mode=BASELINE_MODE,
            region_quantile=REGION_QUANTILE,
        )

        np.save(os.path.join(agg_dir, "chosen_indices.npy"), chosen_idx)

        K = v_mean.shape[0]

        for k in range(K):
            concept_dir = os.path.join(agg_dir, f"concept_{k}")
            os.makedirs(concept_dir, exist_ok=True)

            ig_utils.save_barplot(
                v_mean[k],
                labels=input_vars,
                out_path=os.path.join(concept_dir, "var_importance_mean.png"),
                title=f"{METHOD.upper()} variable importance (MEAN over {len(chosen_idx)} samples)\nconcept {k} | model={model_tag} loss={LOSS_NAME} lead={lead_str}",
                top_k=20,
            )
            ig_utils.save_barplot_mean_std(
                v_mean[k], v_std[k],
                labels=input_vars,
                out_path=os.path.join(concept_dir, "var_importance_mean_std.png"),
                title=f"{METHOD.upper()} variable importance (mean ± std over {len(chosen_idx)} samples)\nconcept {k} | model={model_tag} loss={LOSS_NAME} lead={lead_str}",
                top_k=33,
            )
            ig_utils.save_lineplot(
                t_mean[k],
                out_path=os.path.join(concept_dir, "time_importance_mean.png"),
                title=f"{METHOD.upper()} time importance (MEAN over {len(chosen_idx)} samples)\nconcept {k} | model={model_tag} loss={LOSS_NAME} lead={lead_str}",
                xlabel="t index in input window (0..T-1, past→present)",
                ylabel="Importance (mean sum abs attribution)",
            )
            ig_utils.save_lineplot_mean_std(
                t_mean[k], t_std[k],
                out_path=os.path.join(concept_dir, "time_importance_mean_std.png"),
                title=f"{METHOD.upper()} time importance (mean ± std over {len(chosen_idx)} samples)\nconcept {k} | model={model_tag} loss={LOSS_NAME} lead={lead_str}",
                xlabel="t index in input window (0..T-1, past→present)",
                ylabel="Importance (sum abs attribution)",
            )

        print("[AGG DONE] Aggregate plots written to:", agg_dir)
    else:
        print("[AGG SKIPPED] DO_AGG=False")


if __name__ == "__main__":
    main()