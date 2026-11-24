# g2t_scale — executable-program table reasoning at 7B (WTQ full + graph-as-table), vs Table-R1 and Sparks

Scales the v8 (WTQ hybrid programs) and v3 (graph → table paths) Kaggle experiments to **Qwen2.5-7B-Instruct on A100**, on
**all of WikiTableQuestions** (train 11,321 / validation 2,831 / **test 4,344** = pristine-unseen-tables), with two published baselines:

| baseline | how it is run here |
|---|---|
| **Table-R1** (Yang et al., EMNLP 2025) | the **released checkpoints** `Table-R1/Table-R1-Zero-7B` and `Table-R1/Table-R1-SFT-7B`, with the paper's system prompt, TQA template, `<think>/<answer>` JSON output and inference decoding (T = 0.6, top-p 0.95). |
| **Sparks of Tabular Reasoning** (Stoisser et al., TRL @ ACL 2025) | **re-implemented** (no checkpoints were released) on the same Qwen2.5-7B-Instruct: stage 1 SFT on the authors' **released CoT traces** (`jls205/synthetic_cot_traces_clinton`), stage 2 GRPO (G = 8) on WTQ with SQL executed in SQLite. |

Systems per benchmark (`config.yaml → systems`):

* **WTQ:** `zs_direct`, `zs_cot` (Table-R1 prompt, untrained), `table_r1_zero`, `table_r1_sft`, `sparks_sft`, `sparks`, `direct_sft` / `direct_grpo` (direct-answer control, same data and budget), **`ours_sft` / `ours_grpo`** (PROG executed by the engine / ANS, with fallback).
* **Graph:** the same baselines (zero-shot transfer for Table-R1 and Sparks; Sparks may write recursive CTEs), `direct_*`, and **`ours_*`** (executed paths) plus `ours_*_mask` (action-masked decoding: invented hops impossible).

## 0. Layout
```
config.yaml          all settings; `smoke: true|false` at the top
download.py          run once on the login node (internet): models + WTQ + Sparks traces -> paths.root
run.py               the pipeline: --stage data | wtq_ours_sft | wtq_direct_sft | wtq_ours_grpo | wtq_direct_grpo | sparks_sft |
                     sparks_grpo | graph_sft | graph_grpo | eval | cf | report | all
g2t/                 engine (v8), teacher (v8), graph tasks (v3), SQL executor, prompts, model, training, systems, report
slurm/               smoke.sbatch, full.sbatch, full_split.sh (sequential chain), full_parallel.sh (parallel graph)
tests/               CPU-only tests (no GPU / no downloads): bash tests/run_cpu_tests.sh
```

## 1. One-time setup (login node)
```bash
cd g2t_scale
# edit config.yaml: paths.root  (e.g. /scratch/<user>/g2t ; needs ~70 GB for 3 x 7B models + adapters + logs)
# edit setup_env.sh and slurm/*.sbatch: the `module load ...` lines and `--partition` for Prajna (see `module avail`, `sinfo`)
bash setup_env.sh                       # conda env 'g2t' (torch for CUDA 12.x, transformers, peft, flash-attn if it builds)
conda activate g2t
export HF_TOKEN=hf_...                  # optional (rate limits)
python download.py --config config.yaml # models, WTQ, Sparks traces, CRT-QA -> prints ALL PRESENT at the end
bash tests/run_cpu_tests.sh             # 1 minute, CPU only: engine + data/eval/cf/report stages with a fake model
```

## 2. Smoke test first (`smoke: true`)
```bash
mkdir -p slurm_logs
# multi-GPU (recommended: tests torchrun training + sharded evaluation exactly as the final run will use them)
TRAIN_PART=<a100-partition> TRAIN_GRES=gpu:a100:2 EVAL_PART=<gpu-partition> NSHARDS=4 MODE=--smoke bash slurm/submit_all.sh
# or single GPU, all stages in one job:
sbatch slurm/smoke.sbatch
```
Tiny data (WTQ 400 train / 120 test, graph 60 per task, 3 GRPO steps, short generations), real 7B models, every stage and system.
About 1–2 h with several GPUs. Check that:
* `runs/g2t_7b_smoke/logs/run.log` ends every stage with `finished`, with no `FAILED`. With torchrun each rank logs `rank i/N`;
* `report.md` lists all 13 WTQ and 14 graph systems (**accuracy means nothing at this size**);
* in `instances/*/*.jsonl`, Table-R1 writes `<answer>{"answer": [...]}`, Sparks writes `<sql>`, ours writes `PROG:` / `PATH:`;
* the `sparks_sft` log prints the detected columns of the CoT CSV;
* if BIRD is not downloaded, `sparks_bird_grpo` logs a warning and `sparks_paper` is skipped (download BIRD for the paper setting).

## 3. Final run (`smoke: false`) on all your GPUs
```bash
# optional but recommended: LR sweep for ours + control on VALIDATION (after the data stage of the main run exists)
LRS="5e-5 1e-4 2e-4" bash slurm/sweep.sh ; python tools/compare_runs.py <root>/runs/sweep_*   # put the best into sft.lr
TRAIN_PART=<a100-partition> TRAIN_GRES=gpu:a100:2 EVAL_PART=<gpu-partition> NSHARDS=8 MODE=--full bash slurm/submit_all.sh
```
**How the GPUs are used**
* **Training** (`slurm/train.sbatch`): one stage per job, data-parallel over all GPUs in the job via `torchrun` (LoRA gradients averaged).
  Use **identical bf16-capable GPUs within a job** (A100, or A40/A6000/L40/H100). The four training branches (ours, control, Sparks, graph)
  are independent and run **concurrently** when GPUs are free. Example with 4 A100s: `TRAIN_GRES=gpu:a100:2`, so two branches run at a time.
  Effective batch sizes scale with the GPU count (`sft.micro x sft.accum x GPUs`; GRPO `B` is split across GPUs), so results do not depend on it.
* **Evaluation / counterfactuals** (`slurm/eval_array.sbatch`): a job array of `NSHARDS` one-GPU tasks on **any GPU type with ≥ 24 GB**
  (A100, A40, A6000, L40, V100-32GB, RTX 3090/4090 ...). fp16 is used automatically on GPUs without bf16. K40 / 12-GB cards cannot hold a 7B model.
  Each shard writes `instances/<bench>/<system>.shard<i>of<N>.jsonl`; the report merges them. Re-submit a single failed shard with `--array=<i>`.
  Tip: `--set eval.only_models=[table_r1_sft]` runs only that model (its 18k-token traces) on an extra array on large GPUs.
* Stages are resumable (`done/<stage>` markers; evaluation continues from the instance logs).
* **A100-40GB:** `sft.micro: 2`, `sft.accum: 8`, `grpo.micro: 4`, `eval.token_budget: 150000`, `eval.vllm.max_model_len: 16384`.

## 4. Outputs (`runs/<run_name>/`)
| file | content |
|---|---|
| `report.md`, `metrics.json`, `summary_wtq.csv`, `summary_graph.csv` | accuracy + 95 % CI, paired-bootstrap Δ vs our system, tokens per answer, PROG/SQL execution rates, counterfactual faithfulness, training summary |
| `instances/<bench>/<system>.jsonl` | **one line per test instance**: question, gold, raw output, parsed program / SQL / path, execution status and error, prediction, `correct` (official) and `correct_lenient`, generated tokens, rows shown / truncated, fallback used |
| `cf/<bench>/<system>.jsonl` | the same, for original and edited tables (`cf_kind`: orig / sens / far / near (WTQ), orig / sens / inv (graph)) |
| `train/<adapter>.jsonl` | per-step training logs (SFT loss / lr / grad-norm; GRPO reward, rollout accuracy, groups with signal, length, clipping, reward tags) |
| `logs/run.log`, `logs/<stage>_<time>.log` | full text logs (also in `slurm_logs/*.out`) |
| `adapters/<name>/` | LoRA adapters: wtq_ours_sft/grpo, wtq_direct_sft/grpo, sparks_sft/grpo, graph_ours_sft/grpo, graph_direct_sft/grpo |
| `data/` | teacher programs (coverage per split), SFT / RL splits, data statistics |

## 5. Head-to-head design (our method vs Table-R1 and vs Sparks)
| benchmark | vs Table-R1 | vs Sparks |
|---|---|---|
| **WTQ test, all 4,344** | **direct**: their released 7B checkpoints run here with their prompt and decoding, scored with the same metric as ours. Also printed: their reported WTQ numbers (79.8 / 81.9), which include GPT-4.1-mini re-judging | **re-implementation** `sparks` (their stage 1 on their traces, stage 2 GRPO on the same WTQ RL questions as ours). Sparks reports no WTQ numbers |
| **CRT-QA, all questions (zero-shot)** | their checkpoints run here | **against their published numbers**: Sparks reports zero-shot exact match on CRT-QA (Qwen-2.5-7B SFT-CoT-GRPO 54.0). Our model is evaluated the same way (zero-shot, EM), and the re-implementation runs too, which checks how close it is to the paper |
| **Graph tasks** | checkpoints, zero-shot transfer | re-implementation, zero-shot transfer (SQL with recursive CTEs) |
| **Edit-test faithfulness** | yes | yes |

**Faithfulness (the main claim)** is measured for every baseline with the same oracle edits: follow rate on evidence edits, stale-answer rate, and
invariance to far / near edits (WTQ); re-route, NONE and invariance (graph). It is reported twice: (a) conditional on each system's own correct answers, and
(b) **on one common question set**, the questions that all main systems (`cf.common_systems`) answer correctly. (b) is the fair comparison to put in the paper.
Sparks matters most here: SQL is also executed, so it answers "is any executed program equally faithful?"

The report prints the cited numbers under each table, with a `head-to-head` line (our main system vs each cited number), and every Δ column is `system − ours`, with paired-bootstrap 95% intervals.
Caveats that the report also prints:
- Table-R1's 79.8 / 81.9 use LLM re-judging; our numbers are the official EM. The extra column `acc (num. tol 0.01)` applies their judge's numeric rule to every system.
- Sparks does not state its CRT-QA prompt, subset or answer normalisation, and its Limitations section mentions an o3-mini judge, which conflicts with its exact-match description. Treat the CRT-QA comparison as indicative.

## 6. Baselines: what follows the papers exactly, and what cannot

**Table-R1** (released checkpoints, inference only)
| setting | here | source |
|---|---|---|
| models | `Table-R1-Zero-7B`, `Table-R1-SFT-7B` | released |
| prompt | paper system prompt + TQA template, markdown table, `<think>`/`<answer>` + JSON list | paper, Appendix C |
| decoding | vLLM, temperature 0.6, top-p 0.95 | paper (inference setup) |
| max prompt | 4,096 tokens | paper |
| max output | Zero: 2,048 ("most models"); SFT: 18,000 ("reasoning models": distilled from DeepSeek-R1, trained at 20,480 tokens) | paper; which models count as "reasoning" is not stated |
| metric | official WTQ exact match (set match) | paper's primary metric |
| **not reproduced** | GPT-4.1-mini re-judging of exact-match failures | needs a paid API; expect numbers below their reported WTQ 79.8 |

**Sparks** (re-implemented: no checkpoints released)
| setting | here | source |
|---|---|---|
| backbone | Qwen2.5-7B-Instruct | paper |
| adapters | QLoRA (4-bit NF4, bitsandbytes, PEFT) | paper (Unsloth QLoRA for 7B) |
| LoRA rank / alpha | 16 / 16 | not stated → Unsloth defaults |
| stage 1 | the authors' 3,174 verified CoT traces, lr 2e-4, batch size 1 | paper |
| stage 1 epochs | 1 | not stated |
| stage 2 (`sparks_paper`, **off by default**: add it to `systems:` and download BIRD to run it) | GRPO on BIRD, lr 1e-6, G = 8, KL to the stage-1 policy, schema-only prompts | paper |
| stage 2 rewards | string match (difflib ratio) + component F1 (SELECT / WHERE / GROUP BY) + execution match, equal weights | paper: string match, component F1; **the execution reward was an o3-mini judge (1/(x+1)) and is replaced by real SQLite execution; the LLM class judge is dropped** |
| KL β, clip ε, steps, lengths, temperature | 0.04, 0.2, 200 steps (as ours), 1,024 tokens, 1.0 | not stated |
| stage 2 (`sparks`) | WTQ adaptation: same RL questions as ours, reward = SQL result matches the gold answer | WTQ has no gold SQL |
| precision | bf16 on A100 | paper: fp16 mixed precision |

**Ours vs the direct control:** identical data, LoRA, SFT and GRPO settings (including dynamic sampling); only the target format differs.

## 7. Improvements to our method in this version (each reported as its own system next to the main one)
| improvement | why (evidence from v1–v9) | system |
|---|---|---|
| rejection-sampling fine-tuning: the SFT model samples programs for questions the teacher could not cover; programs that execute to the gold answer become new training targets | teacher coverage (~37 %) capped how often programs could be used; programs win clearly where they exist (62–65 % vs 44–55 %) | `ours_rft`, and GRPO starts from it |
| execution-guided self-consistency (8 samples, vote over executed answers) | program/answer agreement had 89.8 % precision in v8 | `ours_grpo_sc` |
| GRPO dynamic sampling (keep groups whose rewards differ) | groups with signal fell to 12 % by step 40 | all GRPO runs (ours and control) |
| goal-conditioned masked decoding: stop only at the asked destination; dead end → NONE | masking removed invented hops but they became `wrong_end`; no model ever answered NONE | `ours_*_gmask` |
| more no-route questions + edge-deletion training copies | NONE after deletion was 0.00 for every model | graph training data |

Main (pre-declared) systems: `ours_grpo` (WTQ) and `ours_grpo_mask` (graph). Every improvement is reported as its own row, so its gain is measured, not assumed.

## 8. Notes
* Primary WTQ metric: the official evaluator logic, plus thousands-separator canonicalisation (the HF copy has no CoreNLP `targetCanon`). The v1–v8 lenient match is reported as well.
* WTQ tables longer than the prompt budget are cut row-wise (logged per instance); programs and SQL still run on the full table.
* Single seed by default. Run 3 seeds (`seed: 0, 1, 2` with different `run_name`) for the final paper numbers.
* CPU tests: `bash tests/run_cpu_tests.sh` (engine, data/eval/cf/report stages, shard merge, masked decoding, BIRD reward).
