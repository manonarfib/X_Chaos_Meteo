import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from models.utils.ERA5_dataset_from_local import  ERA5Dataset
from explainability.explainable_by_design.WeatherCBM import WeatherCBM

@torch.no_grad()
def analyze_concepts(model, dataloader, device="cuda"):
    model.eval()

    all_alpha = []
    all_y = []

    for x, y, _ in tqdm(dataloader, desc="Processing batches"):
        x = x.to(device)

        y_hat, alpha = model(x)  # alpha: (B, K, H, W)

        alpha_mean = alpha.mean(dim=(2, 3))

        y_mean = y_hat.mean(dim=(1, 2, 3))

        all_alpha.append(alpha_mean.cpu())
        all_y.append(y_mean.cpu())

    all_alpha = torch.cat(all_alpha, dim=0)  # (N, K)
    all_y = torch.cat(all_y, dim=0)          # (N,)

    N, K = all_alpha.shape

    B = model.linear_combination.weight.data.view(-1).cpu()  # (K,)

    results = []

    for k in range(K):
        alpha_k = all_alpha[:, k]

        B_k = B[k].item()
        mean_alpha = alpha_k.mean().item()
        var_alpha = alpha_k.var().item()

        mean_contrib = (B_k * alpha_k).mean().item()

        q_low = torch.quantile(alpha_k, 0.2)
        q_high = torch.quantile(alpha_k, 0.8)

        low_mask = alpha_k <= q_low
        high_mask = alpha_k >= q_high

        y_low = all_y[low_mask].mean().item()
        y_high = all_y[high_mask].mean().item()

        conditional_effect = y_high - y_low

        results.append({
            "concept": k,
            "B_k": B_k,
            "mean_activation": mean_alpha,
            "var_activation": var_alpha,
            "mean_contribution": mean_contrib,
            "y_low (20%)": y_low,
            "y_high (80%)": y_high,
            "conditional_effect": conditional_effect,
        })

    df = pd.DataFrame(results)

    df = df.sort_values(by="mean_contribution", ascending=False)

    return df


if __name__=="__main__":

    BATCH_SIZE=16
    DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    MAX_LEAD=1
    LEAD=1
    CKPT_PATH="checkpoints/weather_cbm/exp_0/epoch6_full.pt"
    T=8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD, without_precip=False, max_lead=MAX_LEAD)    
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)

    model = WeatherCBM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3,
            output_size=MAX_LEAD,
            n_concepts=10
        ).to(device)
    
    df_concepts=analyze_concepts(model, test_loader, device)
    df_concepts.to_csv("explainability/explainable_by_design/explain_results/concept_analysis.csv", index=False)