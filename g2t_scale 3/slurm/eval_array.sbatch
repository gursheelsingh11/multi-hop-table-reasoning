#!/bin/bash
#SBATCH --job-name=g2t-eval
#SBATCH --gres=gpu:1                # one GPU per shard: any type with >= 24 GB (A100/A40/A6000/L40/V100-32GB/RTX-3090/4090 ...)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
# Evaluation (STAGE=eval) or counterfactual (STAGE=cf) as a job ARRAY: task i evaluates shard i of N for every system.
#   sbatch --array=0-7 --export=ALL,STAGE=eval,MODE=--full slurm/eval_array.sbatch
# Each shard resumes from its own instance logs; re-submit a single failed index with --array=<i>.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load cuda/12.1 anaconda3     # <-- EDIT for Prajna
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ${ENV_NAME:-g2t}
nvidia-smi -L
N=${SLURM_ARRAY_TASK_COUNT:-1}
python run.py --config ${CONFIG:-config.yaml} --stage ${STAGE:-eval} --shard $SLURM_ARRAY_TASK_ID --num_shards $N ${MODE:-} ${EXTRA:-}
