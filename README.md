# Faithful Table & Graph Reasoning with Executable Programs

**Small LLMs that answer table and graph questions by writing programs a deterministic engine executes, so every answer
is grounded in the cells it was computed from.**

CS 769 course project · IIT Bombay · Aug 2026 – present

![System architecture](docs/architecture.png)

---


## Motivation

Fine-tuned language models often answer table questions correctly without using the table: a right answer is no evidence
that the model read the evidence. We make the model **show its work as an executable program**. If the evidence cell is
edited, an executed answer must change with it; a memorised answer does not. We measure this directly
(**faithfulness under counterfactual edits**), alongside accuracy and cost.

## Method

```
question + table ──► Qwen2.5 + LoRA ──► PROG: <program>  ──► engine runs it on the full table ──► answer + evidence cells
                                     ├─► PATH: a > b > c ──► engine checks every hop is an edge row
                                     └─► ANS:  <answer>   (when no program fits, or if the program fails)
```

* **Program language (DSL).** A small recursive language over columns `C0, C1, …`: row selection (`FILTER`, `OR`),
  aggregates (`COUNT SUM AVG MAX MIN MODE COUNTD`), row pickers (`FIRST LAST ARGMAX ARGMIN NTH ARGNTH`), neighbours
  (`NEXT PREV`), comparisons (`IS_GT IS_LT IS_EQ`), arithmetic and `LIST` for multi-answer questions.
  Example: `FIRST(FILTER(C2, C0=="Will Power")) → 41`.
* **Teacher.** WikiTableQuestions has answers but no programs, so an execution-verified search (linked cells × columns ×
  operations, depth ≤ 3) keeps only programs that execute to the gold answer. These become training targets.
* **Training.** (1) LoRA SFT on teacher programs (`PROG`) and direct answers (`ANS`) for uncovered questions;
  (2) rejection-sampling fine-tuning (RFT): sample 8 programs per uncovered question, keep those that execute to gold;
  (3) GRPO, 8 rollouts per question, reward = 1 only if the **executed** answer is correct (clip 0.2 / 0.28, no KL,
  dynamic sampling).
* **Graphs as tables.** A graph is serialised as an edge table `Edge | From | To | Cost | Mode`, grouped by `From`. The
  model answers with a path; the engine validates each hop and computes the answer (hop count, total cost, mode chain or
  `NONE`). **Masked decoding** offers only real out-edges of the current node, so invented hops are impossible;
  **goal-conditioned masking** stops only at the destination.
* **Faithfulness test.** Edit the evidence cell (or delete an on-path edge) and check whether the answer follows the
  edit; edits to irrelevant cells should leave it unchanged. Comparing the same weights forced into program mode versus
  direct mode isolates the effect of the program itself.
* **Control.** Every comparison includes a direct-answer model trained on the same data with the same budget.

## Results at 7B

Qwen2.5-7B-Instruct, one seed. Full tables: [`results/g2t_7b/report.md`](results/g2t_7b/report.md).

### WikiTableQuestions test (all 4,344 questions, official evaluator)

| System | Accuracy | Tokens / answer |
|---|---|---|
| Zero-shot direct / CoT | 0.444 / 0.467 | 7 / 109 |
| Direct-answer SFT / GRPO (control) | 0.603 / 0.602 | 8 |
| Ours SFT / RFT | 0.613 / 0.614 | 12 / 15 |
| **Ours GRPO** | **0.663** | **11** |
| Ours GRPO + self-consistency | 0.665 | 86 |
| Table-R1-Zero-7B (released checkpoint) | 0.748 | 330 |
| Table-R1-SFT-7B (released checkpoint) | 0.765 | 1,272 |

* Ours − control: **+6.1 points [4.7, 7.5]**. Table-R1 leads by 8–10 points (its published 79.8 / 81.9 add a
  GPT-4.1-mini re-judge, not used here).
* On questions the model routes to programs, ours matches Table-R1-Zero (0.686 vs 0.685) at ~30× fewer tokens.
* After GRPO the model writes a program for 30% of questions (40% after SFT); the remaining gap is on direct answers.

**Faithfulness:** share of evidence edits the answer follows, on the 379 questions all six systems answered correctly
(620 edits, paired bootstrap by question; reproduce with `tools/faithfulness_common_set.py`).

| Ours, forced program | Table-R1-Zero | Ours, model chooses | Ours, forced direct | Table-R1-SFT | Direct GRPO |
|---|---|---|---|---|---|
| **0.826** | 0.760 | 0.753 | 0.721 | 0.713 | 0.700 |

Same weights, program − direct: **+10.5 [6.5, 14.7]**. Program mode − Table-R1-Zero: **+6.6 [3.4, 10.2]**.

### Graph path questions (1,800: 900 in-distribution, 900 larger out-of-distribution)

| System | Accuracy | ID | OOD | Invented hops | Tokens |
|---|---|---|---|---|---|
| Table-R1-SFT-7B (zero-shot transfer) | 0.470 | 0.61 | 0.33 | – | 3,764 |
| Direct-answer GRPO (same graphs and reward) | 0.468 | 0.56 | 0.37 | – | 6 |
| Ours GRPO, free decoding | 0.706 | 0.81 | 0.60 | 0.10 | 19 |
| Ours GRPO + mask | 0.691 | 0.80 | 0.58 | 0.00 | 13 |
| **Ours GRPO + goal-conditioned mask** | **0.742** | **0.85** | **0.64** | **0.00** | **12** |

### Pilot at 1.5B (Kaggle notebooks)

Same weights followed 81% of evidence edits in program mode vs 27% in direct mode; executed paths beat direct answers on
graphs by +0.30 / +0.26 (ID / OOD). See `notebooks/kaggle_1p5b/`.

## Code structure

```
.
├── run.py                    pipeline entry point: one --stage at a time, or --stage all
├── config.yaml               every setting; `smoke: true` runs all stages on tiny data first
├── download.py               fetch models, WikiTableQuestions, CRT-QA, Sparks traces (needs internet)
├── setup_env.sh              conda environment (torch, transformers, peft, vLLM, flash-attn)
├── requirements.txt
├── g2t/                      the library
│   ├── engine.py             DSL parser + deterministic executor
│   ├── teacher.py            execution-verified program search
│   ├── graph_tasks.py        graph generator, edge tables, solvers, path executor, deletion augmentation
│   ├── systems.py            every system: prompt → generate → parse → execute → log; masked decoding; edit test
│   ├── train.py              LoRA / QLoRA SFT, RFT, GRPO with dynamic sampling
│   ├── dist.py               multi-GPU data parallelism (torchrun)
│   ├── model.py              model + adapter loading, vLLM / HF generation backends
│   ├── prompts.py            our prompts; Table-R1 and Sparks prompts and answer parsers
│   ├── sqlexec.py            SQLite executor for the Sparks (SQL) baseline
│   ├── wtq_data.py, crtqa_data.py, sparks_data.py   dataset loaders
│   ├── wtq_eval.py           official WTQ evaluator logic, numeric tolerance, abstention
│   ├── report.py             summaries, paired bootstrap CIs, report.md, CSVs
│   ├── logs.py, config.py    run directories, completion markers, config loading and overrides
├── slurm/                    cluster jobs
│   ├── smoke.sbatch          smoke test with the real 7B models
│   ├── submit_all.sh         full run as a dependency graph (data → train in parallel → eval → edit test → report)
│   ├── train.sbatch, eval_array.sbatch, cpu.sbatch, full.sbatch
│   ├── ablations.sh, sweep.sh
│   └── rerun_graph.sh        re-run graph evaluation after the generator fix (no retraining)
├── tools/
│   ├── make_ablation.py      prepare an ablation run that reuses the main run's data and adapters
│   ├── compare_runs.py       compare two runs side by side
│   └── faithfulness_common_set.py   paired faithfulness on questions every system answered correctly
├── tests/                    CPU-only tests (engine, teacher, full pipeline with a fake model)
├── results/g2t_7b/           7B report, summary CSVs, resolved config, training logs
├── notebooks/kaggle_1p5b/    1.5B experiments: rl-v8 / rl-v9 (WTQ), graph2table_path_* (graph pilots)
└── docs/                     architecture.png, RUNNING.md (detailed cluster guide)
```

## Key components

**`g2t/engine.py`: the trust anchor.** Parses a program into an AST and evaluates it on the full table with typed
values (`Vec`, `Num`, `Str`, `ListVal`), date and number parsing, and string normalisation. It returns an answer or raises
`ExprError`; it never guesses. Execution errors trigger the `ANS` fallback and score 0 in the RL reward.

**`g2t/teacher.py`: program supervision.** `search_exprs` enumerates programs from the cells and columns the question
mentions (fuzzy linking, number words), ranks them by grounding and lexical cues, and guards against programs that only
echo the question's entity. `search_list` handles multi-answer questions; `mine_exprs` runs the search over a dataset in
parallel workers.

**`g2t/graph_tasks.py`: graphs as tables.** `random_graph` builds graphs with invented node names (nothing to memorise);
`solve` computes gold answers (BFS for hops, Dijkstra for cost, mode chains, `NONE` when unreachable); `graph_table`
writes the edge table; `deletion_augment` adds training copies with a gold-path edge removed so the model learns to
re-route or answer `NONE`.

**`g2t/systems.py`: one pipeline for every system.** Each system (ours, controls, Table-R1, Sparks, zero-shot) goes
through the same prompt → generate → parse → execute → log path, so comparisons are like for like. Also implements masked
and goal-masked path decoding and the counterfactual edit test.

**`g2t/train.py`: training.** LoRA SFT, RFT and GRPO with group-relative advantages, asymmetric clipping, token-level loss
and dynamic sampling (groups where every rollout gets the same reward are skipped and refilled).

**`g2t/report.py`: evaluation.** Writes one JSON line per test instance, then accuracy (official, lenient, 0.01
tolerance), program and fallback rates, tokens and latency, faithfulness, and paired bootstrap deltas against the
reference system.



## Baselines and references

* **Table-R1** (EMNLP 2025): released `Table-R1-Zero-7B` and `Table-R1-SFT-7B` checkpoints, run with their prompt and
  decoding.
* **Sparks of Tabular Reasoning** (TRL @ ACL 2025): re-implemented SFT on the authors' released reasoning traces, then
  GRPO with SQL executed in SQLite.
* **WikiTableQuestions** (Pasupat & Liang, 2015) and **CRT-QA**.
