import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM


def save_heatmap(arr2d, out_path, title="", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(arr2d, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved heatmap: {out_path}")


def save_barplot(values, labels, out_path, title="", top_k=15):
    values = np.asarray(values)
    idx = np.argsort(-values)[:top_k]
    v = values[idx][::-1]
    l = [labels[i] for i in idx][::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(v)), v)
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels(l)
    ax.set_title(title)
    ax.set_xlabel("Importance (sum abs IG)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved barplot: {out_path}")


def save_lineplot(values, out_path, title="", xlabel="t (past steps)", ylabel="Importance (sum abs IG)"):
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(values)), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved lineplot: {out_path}")


@torch.no_grad()
def make_baseline(x, mode="zeros"):
    if mode == "zeros":
        return torch.zeros_like(x)
    elif mode == "mean_over_space_time":
        # constant per-channel baseline: mean over (T,H,W)
        b = x.mean(dim=(1, 3, 4), keepdim=True)  # (B,1,C,1,1)
        return b.expand_as(x).clone()
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")


def integrated_gradients(model, x, baseline, steps=50, target="region_sum", region_mask=None):
    """
    x: (B,T,C,H,W)
    baseline: (B,T,C,H,W)
    region_mask: (B,1,H,W) if target="region_sum"
    returns ig: (B,T,C,H,W)
    """
    assert x.shape == baseline.shape
    B, T, C, H, W = x.shape

    if target == "region_sum" and region_mask is None:
        region_mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)

    total_grad = torch.zeros_like(x, dtype=torch.float32)

    x = x.float()
    baseline = baseline.float()

    for s in range(1, steps + 1):
        alpha = s / steps
        x_alpha = baseline + alpha * (x - baseline)
        x_alpha.requires_grad_(True)

        y_hat = model(x_alpha)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)  # (B,1,H,W)

        if target == "mean":
            S = y_hat.mean()
        elif target == "region_sum":
            S = (y_hat * region_mask).sum()
        else:
            raise ValueError(f"Unknown target: {target}")

        model.zero_grad(set_to_none=True)
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        S.backward()

        total_grad += x_alpha.grad.detach()

    avg_grad = total_grad / float(steps)
    ig = (x - baseline) * avg_grad
    return ig

def find_precip_channel(input_vars):
    # adapte si tes noms diffèrent
    candidates = ["tp", "tp_6h", "total_precipitation", "precip", "precipitation"]
    for cand in candidates:
        for i, name in enumerate(input_vars):
            if cand == name or cand in name:
                return i, name
    return None, None


def overlay_two_maps(base, heat, out_path, title="", base_name="Input tp", heat_name="Attribution"):
    """
    base, heat: np.ndarray (H,W)
    """
    # normalisation pour overlay visuel (uniquement)
    base_norm = (base - np.nanmin(base)) / (np.nanmax(base) - np.nanmin(base) + 1e-8)
    heat_norm = (heat - np.nanmin(heat)) / (np.nanmax(heat) - np.nanmin(heat) + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    im0 = axes[0].imshow(base)
    axes[0].set_title(base_name)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(heat)
    axes[1].set_title(heat_name)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Overlay: base en fond, attribution en alpha
    axes[2].imshow(base_norm)
    im2 = axes[2].imshow(heat_norm, alpha=0.55)
    axes[2].set_title("Overlay (normed)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Top-5% attribution mask + contours over base
    q = np.quantile(heat_norm.flatten(), 0.95)
    mask = (heat_norm >= q).astype(np.float32)
    axes[3].imshow(base_norm)
    axes[3].contour(mask, levels=[0.5], linewidths=1.5)
    axes[3].set_title("Top-5% attr contour")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved overlay comparison: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
    T, lead = 8, 1
    dataset = ERA5Dataset(dataset_path, T=T, lead=lead)

    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)
    print("C_in:", C_in)

    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=[32, 64],
        kernel_size=3,
    ).to(device)

    ckpt_path = "checkpoints_mse/epoch3_full.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {ckpt_path}")

    sample_idx = 50
    X, y, *_ = dataset[sample_idx]
    X = X.unsqueeze(0).to(device).float()  # (1,T,C,H,W)
    y = y.unsqueeze(0).to(device).float()
    if y.dim() == 3:
        y = y.unsqueeze(1)  # (1,1,H,W)

    with torch.no_grad():
        y_hat = model(X)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)

    B, _, H, W = y_hat.shape
    print("y_hat shape:", tuple(y_hat.shape))

    # -------- define region to explain --------
    with torch.no_grad():
        pred_map = y_hat[0, 0]  # (H,W)
        thresh = torch.quantile(pred_map.flatten(), 0.90)
        region = (pred_map >= thresh).float()
        region_mask = region.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # -------- IG --------
    baseline = make_baseline(X, mode="zeros")  # try also "mean_over_space_time"
    steps = 50
    t0 = time.time()
    ig = integrated_gradients(
        model=model,
        x=X,
        baseline=baseline,
        steps=steps,
        target="region_sum",
        region_mask=region_mask,
    )
    print(f"Computed IG in {time.time()-t0:.2f}s (steps={steps})")

    ig_abs = ig.abs()  # (1,T,C,H,W)

    # ig_abs: (1,T,C,H,W)
    spatial_attr = ig_abs.sum(dim=(1, 2))[0].detach().cpu().numpy()   # sum over T,C => (H,W)


    # 1) Variable importance: sum over (T,H,W) -> keep C
    var_importance = ig_abs.sum(dim=(1, 3, 4))  # (B,C) because dims: B,T,C,H,W
    var_importance = var_importance[0].detach().cpu().numpy()  # (C,)

    # 2) Time importance: sum over (C,H,W) -> keep T
    # correct time importance:
    time_importance = ig_abs.sum(dim=(2, 3, 4))  # (B,T)
    time_importance = time_importance[0].detach().cpu().numpy()  # (T,)

    # 3) Spatial attribution: sum over (T,C) -> (H,W)
    spatial_attr = ig_abs.sum(dim=(1, 2))  # (B,H,W)
    spatial_attr = spatial_attr[0].detach().cpu().numpy()

    out_dir = "explainability/ig_outputs_mse"
    os.makedirs(out_dir, exist_ok=True)

    # Save spatial attribution heatmap
    save_heatmap(
        spatial_attr,
        out_path=os.path.join(out_dir, f"sample{sample_idx}_IG_spatial.png"),
        title=f"Integrated Gradients (abs) - spatial (sum over T,C)\nSample {sample_idx}"
    )

    # Save region mask for reference
    save_heatmap(
        region.detach().cpu().numpy(),
        out_path=os.path.join(out_dir, f"sample{sample_idx}_region_mask.png"),
        title=f"Explained region mask (top-10% pred)\nSample {sample_idx}",
        vmin=0, vmax=1
    )

    # Save variable importance barplot
    save_barplot(
        var_importance,
        labels=input_vars,
        out_path=os.path.join(out_dir, f"sample{sample_idx}_IG_var_importance.png"),
        title=f"Integrated Gradients - variable importance\n(sum abs IG over T,H,W)\nSample {sample_idx}",
        top_k=15
    )

    # Save time importance curve
    save_lineplot(
        time_importance,
        out_path=os.path.join(out_dir, f"sample{sample_idx}_IG_time_importance.png"),
        title=f"Integrated Gradients - time importance\n(sum abs IG over C,H,W)\nSample {sample_idx}",
        xlabel="t index in input sequence (0..T-1, past→present)",
        ylabel="Importance (sum abs IG)"
    )

    # Optional: top-k variable spatial maps
    top_k = 5
    top_idx = np.argsort(-var_importance)[:top_k]
    for rank, c_idx in enumerate(top_idx, start=1):
        # sum over time only -> (H,W) for this variable
        m = ig_abs[0, :, c_idx].sum(dim=0).detach().cpu().numpy()
        save_heatmap(
            m,
            out_path=os.path.join(out_dir, f"sample{sample_idx}_IG_var{rank}_{input_vars[c_idx]}.png"),
            title=f"IG spatial for var={input_vars[c_idx]} (sum over T)\nSample {sample_idx}"
        )

    t_view = T - 1  # "temps actuel" = dernier pas de la fenêtre
    tp_idx, tp_name = find_precip_channel(input_vars)

    if tp_idx is None:
        print("[WARN] Aucun channel 'tp' trouvé dans input_vars. Je ne peux pas superposer la pluie d'entrée.")
    else:
        tp_in = X[0, t_view, tp_idx].detach().cpu().numpy()  # (H,W)
        overlay_two_maps(
            base=tp_in,
            heat=spatial_attr,
            out_path=os.path.join(out_dir, f"sample{sample_idx}_IG_overlay_tp_input_t{t_view}.png"),
            title=f"Sample {sample_idx} overlay",
            base_name=f"Input {tp_name} at t={t_view}",
            heat_name="IG spatial (sum T,C)"
        )



if __name__ == "__main__":
    main()
