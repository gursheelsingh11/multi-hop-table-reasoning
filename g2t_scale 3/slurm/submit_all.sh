#!/bin/bash
# Submits the whole pipeline as a dependency graph so all your GPUs are used:
#   data (CPU) -> {wtq_ours_sft, wtq_direct_sft, sparks_sft, graph_sft}            training jobs on A100s (torchrun, TRAIN_GRES each)
#              -> wtq_ours_rft -> wtq_ours_grpo ; wtq_direct_grpo ; sparks_grpo + sparks_bird_grpo ; graph_grpo
#              -> eval (job array, NSHARDS x 1 GPU of ANY good type) -> cf (job array) -> report (CPU)
# SLURM queues jobs until GPUs are free, so independent stages run concurrently as far as the cluster allows.
#
# usage:  MODE=--smoke bash slurm/submit_all.sh       (smoke test first)
#         MODE=--full  bash slurm/submit_all.sh       (final run)
# EDIT these for Prajna (`sinfo -o "%P %G %N"` shows partitions and GPU types):
TRAIN_PART=${TRAIN_PART:-gpu}            # partition with the A100s
TRAIN_GRES=${TRAIN_GRES:-gpu:a100:2}     # GPUs per training job (all A100 / same type!). e.g. 4 A100s -> two jobs of 2 run side by side
EVAL_PART=${EVAL_PART:-gpu}              # partition(s) for evaluation shards (any GPU type with >= 24 GB)
EVAL_GRES=${EVAL_GRES:-gpu:1}
NSHARDS=${NSHARDS:-8}                    # evaluation shards (= GPUs that can work on evaluation in parallel)
CPU_PART=${CPU_PART:-cpu}
MODE=${MODE:---smoke}
set -e; mkdir -p slurm_logs
X="--export=ALL,MODE=$MODE"
tr()  { sbatch --parsable --partition=$TRAIN_PART --gres=$TRAIN_GRES ${2:+--dependency=afterok:$2} --job-name=g2t-$1 $X,STAGE=$1 slurm/train.sbatch; }
arr() { sbatch --parsable --partition=$EVAL_PART --gres=$EVAL_GRES --array=0-$((NSHARDS-1)) --dependency=afterok:$2 --job-name=g2t-$1 $X,STAGE=$1 slurm/eval_array.sbatch; }
cpu() { sbatch --parsable --partition=$CPU_PART ${2:+--dependency=afterok:$2} --job-name=g2t-$1 $X,STAGE=$1 slurm/cpu.sbatch; }
D=$(cpu data)
A=$(tr wtq_ours_sft $D);   A2=$(tr wtq_ours_rft $A);   A3=$(tr wtq_ours_grpo $A2)
B=$(tr wtq_direct_sft $D); B2=$(tr wtq_direct_grpo $B)
C=$(tr sparks_sft $D);     C2=$(tr sparks_grpo $C);    C3=$(tr sparks_bird_grpo $C)
G=$(tr graph_sft $D);      G2=$(tr graph_grpo $G)
E=$(arr eval "$A3:$B2:$C2:$C3:$G2"); F=$(arr cf "$E"); R=$(cpu report "$F")
echo "data $D | wtq_ours $A,$A2,$A3 | wtq_direct $B,$B2 | sparks $C,$C2,$C3 | graph $G,$G2 | eval[array] $E | cf[array] $F | report $R"
echo "watch: squeue -u $USER ; logs: slurm_logs/ and runs/<run_name>/logs/run.log"
