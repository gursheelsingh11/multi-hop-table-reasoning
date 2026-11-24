#!/bin/bash
#SBATCH --job-name=g2t-full
#SBATCH --partition=gpu            # <-- EDIT: Prajna GPU partition name (sinfo)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=72:00:00            # <-- EDIT to the partition's limit; every stage is resumable, just resubmit if it times out
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# FINAL RUN (smoke=false). Stages already finished are skipped, unfinished evaluation resumes from the instance logs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load cuda/12.1 anaconda3   # <-- EDIT for Prajna
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ${ENV_NAME:-g2t}
nvidia-smi
STAGE=${STAGE:-all}                # single-GPU, all stages sequentially (use slurm/submit_all.sh for multi-GPU)
python run.py --config config.yaml --stage $STAGE --full
