#!/bin/bash
# Learning-rate sweep for OUR SFT and the DIRECT control (identical grid), selected on the VALIDATION split (never test).
# Needs the main run's data stage first (re-uses runs/<run_name>/data). Compare afterwards: python tools/compare_runs.py runs/sweep_*
MAIN=${MAIN:-$(python -c "import yaml,os;c=yaml.safe_load(open('config.yaml'));print(os.path.join(c['paths']['root'],'runs',c['run_name']))")}
LRS=${LRS:-"5e-5 1e-4 2e-4"}
TRAIN_PART=${TRAIN_PART:-gpu}; TRAIN_GRES=${TRAIN_GRES:-gpu:a100:2}; EVAL_PART=${EVAL_PART:-gpu}; EVAL_GRES=${EVAL_GRES:-gpu:1}; CPU_PART=${CPU_PART:-cpu}
set -e; mkdir -p slurm_logs configs
for lr in $LRS; do
  name=sweep_lr$lr; cfgf=configs/$name.yaml
  python - "$lr" "$name" "$cfgf" <<'PY'
import sys, yaml
lr, name, out = sys.argv[1:]
c = yaml.safe_load(open("config.yaml")); c["smoke"] = False; c["run_name"] = name; c["sft"]["lr"] = float(lr)
c["wtq"]["eval_split"] = "validation"; c["benchmarks"] = ["wtq"]; c["systems"]["wtq"] = ["ours_sft", "direct_sft"]
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
  root=$(python -c "import yaml,os;c=yaml.safe_load(open('$cfgf'));print(os.path.join(c['paths']['root'],'runs','$name'))")
  mkdir -p $root/data $root/done; cp $MAIN/data/* $root/data/; cp $MAIN/done/data $root/done/
  X="--export=ALL,MODE=--full,CONFIG=$cfgf"
  a=$(sbatch --parsable --partition=$TRAIN_PART --gres=$TRAIN_GRES $X,STAGE=wtq_ours_sft slurm/train.sbatch)
  b=$(sbatch --parsable --partition=$TRAIN_PART --gres=$TRAIN_GRES $X,STAGE=wtq_direct_sft slurm/train.sbatch)
  e=$(sbatch --parsable --partition=$EVAL_PART --gres=$EVAL_GRES --array=0-1 --dependency=afterok:$a:$b $X,STAGE=eval slurm/eval_array.sbatch)
  r=$(sbatch --parsable --partition=$CPU_PART --dependency=afterok:$e $X,STAGE=report slurm/cpu.sbatch)
  echo "lr $lr: train $a $b | eval $e | report $r   (config $cfgf)"
done
echo "when done: python tools/compare_runs.py $(dirname $MAIN)/sweep_*  -> put the best lr into config.yaml (sft.lr) for the final run"
