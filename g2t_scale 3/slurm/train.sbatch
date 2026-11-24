#!/bin/bash
#SBATCH --job-name=g2t-train
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# ONE training stage on all GPUs of this job (torchrun, data-parallel LoRA). Submitted by slurm/submit_all.sh, e.g.:
#   sbatch --partition=<a100 partition> --gres=gpu:a100:4 --export=ALL,STAGE=wtq_ours_sft,MODE=--full slurm/train.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load cuda/12.1 anaconda3     # <-- EDIT for Prajna
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ${ENV_NAME:-g2t}
NGPU=$(nvidia-smi -L | wc -l)
echo "stage $STAGE on $NGPU GPU(s):"; nvidia-smi -L
export OMP_NUM_THREADS=4 NCCL_ASYNC_ERROR_HANDLING=1
if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node=$NGPU run.py --config ${CONFIG:-config.yaml} --stage $STAGE ${MODE:-} ${EXTRA:-}
else
  python run.py --config ${CONFIG:-config.yaml} --stage $STAGE ${MODE:-} ${EXTRA:-}
fi
