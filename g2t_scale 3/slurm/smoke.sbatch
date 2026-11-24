#!/bin/bash
#SBATCH --job-name=g2t-smoke
#SBATCH --partition=gpu            # <-- EDIT: Prajna GPU partition name (sinfo)
#SBATCH --gres=gpu:1               # one A100 (80 GB or 40 GB)
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# SMOKE TEST: every stage and every system on tiny data with the real 7B models. Expected ~2-3 h on one A100.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load cuda/12.1 anaconda3   # <-- EDIT for Prajna
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ${ENV_NAME:-g2t}
nvidia-smi
python run.py --config config.yaml --stage all --smoke
echo "smoke run finished -> see runs/<run_name>_smoke/report.md and logs/run.log"
