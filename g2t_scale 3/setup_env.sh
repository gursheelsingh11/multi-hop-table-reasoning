#!/bin/bash
# One-time environment setup on Prajna (run on the login node, which has internet).
# EDIT the module line to whatever `module avail` shows on Prajna (CUDA 12.x + a conda/python module).
set -e
# module load cuda/12.1 anaconda3          # <-- EDIT for Prajna
ENV_NAME=${ENV_NAME:-g2t}
conda create -y -n $ENV_NAME python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate $ENV_NAME
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121      # match the cluster CUDA (cu121 / cu124 ...)
pip install -r requirements.txt
pip install flash-attn --no-build-isolation || echo "flash-attn not installed -> sdpa attention will be used (slower, same results)"
pip install vllm || echo "vLLM not installed -> HF generation (Table-R1-SFT outputs capped at eval.hf_max_new_cap)"
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__)"
