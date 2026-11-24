# Results — run `g2t_7b` (smoke=False)


## WTQ — 4344 test instances (common to all systems)

| system | acc (official) [95% CI] | acc (num. tol 0.01) | acc (lenient) | tokens/answer | PROG rate | executed share of correct | SQL executes | format ok | rows truncated | Δ = system − ours_grpo [95% CI] (better/worse than ours_grpo) |
|---|---|---|---|---|---|---|---|---|---|---|
| table_r1_sft | 0.765 [0.752-0.777] | 0.765 | 0.758 | 1272 | - | - | - | 0.972 | 0.027 | +0.101 [+0.086, +0.116] better |
| table_r1_zero | 0.748 [0.735-0.761] | 0.748 | 0.742 | 330 | - | - | - | 1.000 | 0.027 | +0.084 [+0.070, +0.099] better |
| ours_grpo_sc | 0.665 [0.651-0.679] | 0.665 | 0.658 | 86 | 0.341 | 0.000 | - | - | 0.020 | +0.002 [-0.002, +0.005] n.s. |
| ours_grpo | 0.663 [0.649-0.677] | 0.663 | 0.657 | 11 | 0.301 | 0.311 | - | - | 0.020 | reference |
| ours_rft | 0.614 [0.600-0.629] | 0.614 | 0.608 | 15 | 0.717 | 0.789 | - | - | 0.020 | -0.049 [-0.061, -0.038] worse |
| ours_sft | 0.613 [0.598-0.627] | 0.613 | 0.606 | 12 | 0.396 | 0.480 | - | - | 0.020 | -0.051 [-0.063, -0.039] worse |
| direct_sft | 0.603 [0.588-0.617] | 0.603 | 0.597 | 8 | - | - | - | - | 0.020 | -0.061 [-0.075, -0.047] worse |
| direct_grpo | 0.602 [0.587-0.616] | 0.602 | 0.597 | 8 | - | - | - | - | 0.020 | -0.061 [-0.075, -0.047] worse |
| zs_cot | 0.467 [0.453-0.482] | 0.467 | 0.460 | 109 | - | - | - | 0.167 | 0.027 | -0.196 [-0.213, -0.180] worse |
| zs_direct | 0.444 [0.429-0.459] | 0.444 | 0.439 | 7 | - | - | - | - | 0.020 | -0.219 [-0.235, -0.204] worse |

**Published results (WTQ), cited from the papers, not re-run here.** Protocols differ (see notes), so compare with care:

| system (as reported) | score | metric / protocol | source |
|---|---|---|---|
| Table-R1-Zero-7B | 79.8 | EM + GPT-4.1-mini re-judging of EM failures | Yang et al., EMNLP 2025, Table 2 |
| Table-R1-SFT-7B | 81.9 | EM + GPT-4.1-mini re-judging of EM failures | Yang et al., EMNLP 2025, Table 2 |
| Qwen2.5-7B-Instruct | 54.8 | EM + GPT-4.1-mini re-judging | Yang et al., EMNLP 2025, Table 2 |

- head-to-head: **ours_grpo 66.3** (this run, 4344 questions) vs **Table-R1-Zero-7B 79.8** (reported): -13.5 points

- head-to-head: **ours_grpo 66.3** (this run, 4344 questions) vs **Table-R1-SFT-7B 81.9** (reported): -15.6 points

### WTQ counterfactual faithfulness (rates conditional on correct on the original table)

| system | acc orig | follows evidence edit [95% CI] | stale answer | ignores far edit | ignores near edit | n edits |
|---|---|---|---|---|---|---|
| direct_grpo | 0.693 | 0.637 [0.61-0.67] | 0.223 | 0.976 | 0.955 | 928 |
| ours_grpo | 0.814 | 0.732 [0.71-0.76] | 0.157 | 0.989 | 0.969 | 1076 |
| ours_grpo__forced_ans | 0.649 | 0.634 [0.60-0.67] | 0.221 | 0.972 | 0.943 | 872 |
| ours_grpo__forced_prog | 0.796 | 0.826 [0.80-0.85] | 0.112 | 0.989 | 0.975 | 1050 |
| table_r1_sft | 0.833 | 0.673 [0.65-0.70] | 0.146 | 0.933 | 0.916 | 1102 |
| table_r1_zero | 0.840 | 0.688 [0.66-0.71] | 0.200 | 0.960 | 0.957 | 1110 |
| zs_cot | 0.526 | 0.635 [0.60-0.67] | 0.138 | 0.884 | 0.868 | 690 |
| zs_direct | 0.516 | 0.583 [0.55-0.62] | 0.195 | 0.956 | 0.912 | 712 |

**Same questions for every system** (447 original questions answered correctly by all of: table_r1_zero, table_r1_sft, direct_grpo, ours_grpo; the fairest comparison):

| system | follows evidence edit [95% CI] | stale answer | ignores far edit | ignores near edit | n edits |
|---|---|---|---|---|---|
| direct_grpo | 0.686 [0.65-0.72] | 0.195 | 0.985 | 0.967 | 732 |
| ours_grpo | 0.734 [0.70-0.76] | 0.158 | 0.992 | 0.973 | 732 |
| ours_grpo__forced_ans | 0.696 [0.66-0.73] | 0.189 | 0.981 | 0.961 | 694 |
| ours_grpo__forced_prog | 0.832 [0.80-0.86] | 0.107 | 0.991 | 0.980 | 656 |
| table_r1_sft | 0.695 [0.66-0.73] | 0.135 | 0.967 | 0.958 | 732 |
| table_r1_zero | 0.742 [0.71-0.77] | 0.161 | 0.982 | 0.971 | 732 |
| zs_cot | 0.676 [0.64-0.71] | 0.122 | 0.919 | 0.897 | 556 |
| zs_direct | 0.630 [0.59-0.67] | 0.154 | 0.961 | 0.928 | 584 |

## GRAPH — 1800 test instances (common to all systems)

| system | acc all [95% CI] | ID | OOD | chain ID/OOD | cost ID/OOD | hops ID/OOD | valid path | hallucinated hop | tokens | Δ = system − ours_grpo_mask, ID / OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| ours_grpo_gmask | 0.742 [0.722-0.762] | 0.848 | 0.637 | 0.797/0.570 | 0.870/0.617 | 0.877/0.723 | 0.894 | 0.000 | 12 | +0.044 [+0.027,+0.063] / +0.059 [+0.036,+0.083] |
| ours_sft_gmask | 0.707 [0.686-0.728] | 0.816 | 0.599 | 0.780/0.540 | 0.810/0.570 | 0.857/0.687 | 0.887 | 0.000 | 12 | +0.012 [-0.011,+0.037] / +0.021 [-0.006,+0.048] |
| ours_grpo | 0.706 [0.685-0.727] | 0.811 | 0.601 | 0.807/0.620 | 0.800/0.560 | 0.827/0.623 | 0.806 | 0.103 | 19 | +0.008 [-0.002,+0.018] / +0.023 [+0.011,+0.037] |
| ours_sft | 0.692 [0.670-0.713] | 0.806 | 0.578 | 0.820/0.610 | 0.767/0.517 | 0.830/0.607 | 0.811 | 0.103 | 18 | +0.002 [-0.021,+0.026] / +0.000 [-0.026,+0.026] |
| ours_grpo_mask | 0.691 [0.669-0.711] | 0.803 | 0.578 | 0.773/0.550 | 0.807/0.560 | 0.830/0.623 | 0.794 | 0.000 | 13 | reference |
| ours_sft_mask | 0.675 [0.653-0.696] | 0.792 | 0.558 | 0.780/0.543 | 0.773/0.520 | 0.823/0.610 | 0.798 | 0.000 | 12 | -0.011 [-0.033,+0.011] / -0.020 [-0.043,+0.002] |
| table_r1_sft | 0.470 [0.447-0.493] | 0.611 | 0.329 | 0.740/0.490 | 0.453/0.193 | 0.640/0.303 | - | - | 3764 | -0.192 [-0.232,-0.152] / -0.249 [-0.292,-0.207] |
| direct_grpo | 0.468 [0.445-0.491] | 0.564 | 0.372 | 0.633/0.347 | 0.390/0.257 | 0.670/0.513 | - | - | 6 | -0.239 [-0.276,-0.202] / -0.206 [-0.243,-0.169] |
| table_r1_zero | 0.444 [0.422-0.467] | 0.584 | 0.304 | 0.770/0.507 | 0.387/0.157 | 0.597/0.250 | - | - | 469 | -0.219 [-0.257,-0.180] / -0.273 [-0.317,-0.232] |
| direct_sft | 0.436 [0.413-0.459] | 0.568 | 0.304 | 0.580/0.287 | 0.393/0.210 | 0.730/0.417 | - | - | 6 | -0.236 [-0.273,-0.199] / -0.273 [-0.310,-0.237] |
| zs_cot | 0.289 [0.269-0.311] | 0.394 | 0.184 | 0.543/0.280 | 0.240/0.080 | 0.400/0.193 | - | - | 207 | -0.409 [-0.447,-0.371] / -0.393 [-0.433,-0.354] |
| zs_direct | 0.155 [0.139-0.172] | 0.164 | 0.146 | 0.153/0.027 | 0.167/0.173 | 0.173/0.237 | - | - | 2 | -0.639 [-0.676,-0.602] / -0.432 [-0.469,-0.397] |

### GRAPH counterfactual faithfulness (rates conditional on correct on the original table)

| system | acc orig | follows deletion | re-route | -> NONE | stale | ignores off-path deletion | n |
|---|---|---|---|---|---|---|---|
| direct_grpo | 0.221 | 0.375 | 0.429 | 0.333 | 0.188 | 0.625 | 16 |
| ours_grpo | 0.422 | 0.607 | 0.875 | 0.500 | 0.000 | 0.964 | 28 |
| ours_grpo_gmask | 0.431 | 0.966 | 0.875 | 1.000 | 0.000 | 0.966 | 29 |
| ours_grpo_mask | 0.431 | 0.621 | 0.875 | 0.524 | 0.000 | 0.966 | 29 |
| table_r1_zero | 0.326 | 0.143 | 0.167 | 0.133 | 0.381 | 0.810 | 21 |
| zs_direct | 0.044 | 0.000 | - | 0.000 | 0.000 | 0.000 | 1 |

**Same questions for every system** (56 original questions answered correctly by all of: table_r1_zero, direct_grpo, ours_grpo, ours_grpo_mask; the fairest comparison):

| system | follows deletion | re-route | -> NONE | stale | ignores off-path deletion | n |
|---|---|---|---|---|---|---|
| direct_grpo | 0.444 | 0.333 | 0.500 | 0.000 | 0.667 | 9 |
| ours_grpo | 0.667 | 1.000 | 0.500 | 0.000 | 1.000 | 9 |
| ours_grpo_gmask | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 9 |
| ours_grpo_mask | 0.667 | 1.000 | 0.500 | 0.000 | 1.000 | 9 |
| table_r1_zero | 0.111 | 0.000 | 0.167 | 0.444 | 0.778 | 9 |
| zs_direct | - | - | - | - | - | 0 |

## Training

| run | kind | steps | start | end | groups with signal (end) |
|---|---|---|---|---|---|
| graph_direct_grpo | GRPO | 200 | rollout acc 0.430 | rollout acc 0.496 | 0.12 |
| graph_direct_sft | SFT | 170 | loss 6.043 | loss 0.179 | - |
| graph_ours_grpo | GRPO | 200 | rollout acc 0.611 | rollout acc 0.712 | 0.12 |
| graph_ours_sft | SFT | 170 | loss 3.963 | loss 0.035 | - |
| wtq_direct_grpo | GRPO | 200 | rollout acc 0.547 | rollout acc 0.802 | 0.07 |
| wtq_direct_sft | SFT | 302 | loss 6.282 | loss 0.130 | - |
| wtq_ours_grpo | GRPO | 200 | rollout acc 0.596 | rollout acc 0.825 | 0.04 |
| wtq_ours_rft | SFT | 58 | loss 0.172 | loss 0.180 | - |
| wtq_ours_sft | SFT | 302 | loss 5.755 | loss 0.187 | - |

## Notes
- `ours_grpo_sc` = execution-guided self-consistency (k samples, majority vote over executed answers); `*_gmask` = goal-conditioned action-masked decoding; `ours_rft` = SFT + rejection-sampled programs. `sparks_paper` = stage 2 on BIRD with the paper's rewards; `sparks` = stage 2 on WTQ.
- Primary WTQ metric: official-evaluator logic (set match, number/date typing) with thousands-separator canonicalisation; 'lenient' = the v1–v8 matcher.
- Table-R1 models are evaluated with their own prompt and decoding (T=0.6, top-p 0.95); their paper additionally re-checks exact-match failures with a GPT-4.1-mini judge, which is not done here, so numbers are lower than their reported ones.
- Sparks is a re-implementation (no checkpoints were released): stage-1 SFT on the authors' released CoT traces, stage-2 GRPO on WTQ with an execution reward (WTQ has no gold SQL, so their SQL-similarity rewards cannot be used).
- Graph-benchmark Table-R1 and Sparks results are zero-shot transfer.
