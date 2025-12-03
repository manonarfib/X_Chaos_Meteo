#!/bin/bash
#SBATCH --job-name=spearman_corr
#SBATCH --output=logs/spearman_%j.out
#SBATCH --error=logs/spearman_%j.err
#SBATCH --time=12:00:00      
#SBATCH --nodes=1
#SBATCH --partition=cpu_prod          # ou cpu, long, bigmem…
#SBATCH --nodelist=kyle03

echo "=== Job started on $(date) ==="

module load anaconda3/2022.10/gcc-13.1.0
source activate x_chaos_env

echo "Environment activated: $(which python)"

# -------------------------------------------------------------------
# 3. Lancer le script
# -------------------------------------------------------------------
python spearman_correlation_cluster.py

echo "=== Job finished on $(date) ==="
