#!/bin/bash
#SBATCH --job-name=g2t-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# CPU-only stages: data (teacher search, multiprocessing) and report.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ${ENV_NAME:-g2t}
python run.py --config ${CONFIG:-config.yaml} --stage $STAGE ${MODE:-} ${EXTRA:-}
