#!/usr/bin/env python
"""Executable-program table reasoning at 7B (WTQ full + graph-as-table): training, Table-R1 / Sparks baselines, evaluation, logs.

  single GPU      : python run.py --config config.yaml --stage all
  multi-GPU train : torchrun --nproc_per_node=4 run.py --config config.yaml --stage wtq_ours_sft          (training stages only)
  sharded eval    : python run.py --stage eval --shard 3 --num_shards 8   (one process per GPU; any GPU type with >= 24 GB)
  overrides       : python run.py --stage wtq_ours_sft --set sft.lr=5e-5 --set run_name=sweep_lr5e-5
  smoke / full    : --smoke / --full (or `smoke:` in config.yaml)      redo a finished stage: --force

Stages: data | wtq_ours_sft | wtq_direct_sft | wtq_ours_rft | wtq_ours_grpo | wtq_direct_grpo | sparks_sft | sparks_grpo | sparks_bird_grpo |
        graph_sft | graph_grpo | eval | cf | report
"""
import argparse, json, multiprocessing as mp, os, random, sys, time, traceback
import numpy as np
import yaml

from g2t.config import load_config, model_path, dump
from g2t.logs import get_logger, run_paths, JsonlWriter, read_jsonl, save_json, load_json, is_done, mark_done
from g2t import dist

STAGES = ["data", "wtq_ours_sft", "wtq_direct_sft", "wtq_ours_rft", "wtq_ours_grpo", "wtq_direct_grpo", "sparks_sft", "sparks_grpo",
          "sparks_bird_grpo", "graph_sft", "graph_grpo", "eval", "cf", "report"]
TRAIN_STAGES = {"wtq_ours_sft", "wtq_direct_sft", "wtq_ours_rft", "wtq_ours_grpo", "wtq_direct_grpo", "sparks_sft", "sparks_grpo",
                "sparks_bird_grpo", "graph_sft", "graph_grpo"}
SHARDED_STAGES = {"eval", "cf"}


def seed_all(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception: pass


# ======================================================================== shared data (deterministic; cached in the run dir)
def _mine_one(a):
    from g2t.teacher import search_exprs
    q, h, r, ans, depth = a
    try: return search_exprs(q, h, r, ans, depth)
    except Exception: return None


def wtq_data(cfg):
    from g2t.wtq_data import load_wtq
    P = run_paths(cfg); cache = os.path.join(P["data"], "wtq_meta.json")
    if not os.path.exists(cache): return None
    train, dev, test = load_wtq(cfg); meta = load_json(cache)
    teacher = meta["teacher"]
    rft = os.path.join(P["data"], "wtq_rft_programs.json")
    if os.path.exists(rft): teacher["train_rft"] = load_json(rft)
    return dict(train=train, dev=dev, test=test, teacher=teacher, sft_ids=set(meta["sft_ids"]), rl_ids=set(meta["rl_ids"]), stats=meta["stats"])


def stage_data(cfg, log):
    from g2t.wtq_data import load_wtq
    from g2t.teacher import ekey
    from g2t.prompts import ours_prompt
    from g2t.model import load_tokenizer
    P = run_paths(cfg); stats = {}
    if "wtq" in cfg["benchmarks"] or "crtqa" in cfg["benchmarks"]:          # CRT-QA systems are trained on WTQ
        train, dev, test = load_wtq(cfg); W = cfg["wtq"]
        log.info(f"WTQ loaded: train {len(train)} | validation {len(dev)} | test {len(test)}")
        ok = lambda e: len(e.answers) == 1 and len(e.rows) <= W["teacher_max_rows"] and len(e.headers) <= W["teacher_max_cols"]
        teacher = {}
        for split, exs in (("train", train), ("validation", dev), ("test", test)):
            t0 = time.time(); sel = [e for e in exs if ok(e)]
            with mp.Pool(W["teacher_workers"]) as pool:
                res = pool.map(_mine_one, [(e.question, e.headers, e.rows, e.answer, W["teacher_depth"]) for e in sel], chunksize=8)
            teacher[split] = {ekey(e): dict(expr=r["expr"], depth=r["depth"]) for e, r in zip(sel, res) if r}
            stats[f"teacher_cov_{split}"] = len(teacher[split]) / max(1, len(exs))
            log.info(f"teacher search {split}: {len(teacher[split])}/{len(exs)} covered ({stats[f'teacher_cov_{split}']:.1%}) in {time.time()-t0:.0f}s")
        tok = load_tokenizer(model_path(cfg, "base"))
        fits = [e for e in train if len(tok(ours_prompt(e), add_special_tokens=False).input_ids) <= W["max_prompt_tokens_train"]]
        ids = [e.id for e in fits]; random.Random(cfg["seed"]).shuffle(ids)
        n_rl = int(round(W["rl_holdout_frac"] * len(ids))); rl_ids, sft_ids = ids[:n_rl], ids[n_rl:]
        stats.update(train_fit=len(fits), train_total=len(train), sft_n=len(sft_ids), rl_n=len(rl_ids),
                     test_multi_answer=sum(len(e.answers) > 1 for e in test) / max(1, len(test)))
        log.info(f"train questions within {W['max_prompt_tokens_train']} prompt tokens: {len(fits)}/{len(train)} -> SFT {len(sft_ids)} | RL (disjoint) {len(rl_ids)}")
        save_json(dict(teacher=teacher, sft_ids=sft_ids, rl_ids=rl_ids, stats=stats), os.path.join(P["data"], "wtq_meta.json"))
        for e in [e for e in train if ekey(e) in teacher["train"]][:8]:
            log.info(f"  teacher sample: {e.question[:80]!r} gold={e.answers} PROG {teacher['train'][ekey(e)]['expr']}")
    if "crtqa" in cfg["benchmarks"]:
        from g2t.crtqa_data import load_crtqa
        cq = load_crtqa(cfg, log); stats["crtqa_n"] = len(cq)
        for e in cq[:3]: log.info(f"  crtqa sample: {e.question[:90]!r} gold={e.answers} | table {len(e.rows)}x{len(e.headers)}")
    if "graph" in cfg["benchmarks"]:
        gs = graph_splits(cfg)
        for k, v in gs.items(): log.info(f"graph {k}: {len(v)} questions" + (f" (NONE {np.mean([q.gold == 'NONE' for q in v]):.0%})" if v else ""))
        stats["graph_sizes"] = {k: len(v) for k, v in gs.items()}
    if any(s.startswith("sparks") for b in cfg["benchmarks"] for s in cfg["systems"].get(b, [])):
        p = os.path.join(cfg["paths"]["data"], "sparks", "cot.csv"); stats["sparks_cot_csv"] = os.path.exists(p)
        if not os.path.exists(p): raise FileNotFoundError(f"{p} missing -> run download.py")
    save_json(stats, os.path.join(P["data"], "data_stats.json"))


def graph_splits(cfg):
    from g2t import graph_tasks as G
    G.configure(cfg["graph"]); g = cfg["graph"]; s = g["data_seed"]
    train = G.make_split(s + 1, g["n_train_per_task"], "train")
    if g.get("deletion_aug", 0) > 0: train = train + G.deletion_augment(train, g["deletion_aug"], s + 9)
    return dict(train=train, test_id=G.make_split(s + 2, g["n_test_per_task"], "id"),
                test_ood=G.make_split(s + 3, g["n_test_per_task"], "ood"), val=G.make_split(s + 4, g["n_val_per_task"], "id"),
                pool_id=G.make_split(s + 20, g["n_pool_per_task"], "train"), pool_mid=G.make_split(s + 22, g["n_pool_per_task"], "mid"))


# ======================================================================== training helpers
def _train_model(cfg, log, quant=None):
    from g2t.model import load_hf, load_tokenizer
    path = model_path(cfg, "base"); tok = load_tokenizer(path); model = load_hf(path, cfg, log, trainable=True, quant=quant)
    return model, tok


def adapter_dir(cfg, name): return os.path.join(run_paths(cfg)["adapters"], name)


def _save_adapter(model, cfg, name, log):
    if dist.is_main():
        P = run_paths(cfg); model.save_pretrained(P["adapters"], selected_adapters=[name])
        if not os.path.isdir(adapter_dir(cfg, name)): raise RuntimeError(f"adapter {name} not found at {adapter_dir(cfg, name)}")
        log.info(f"saved adapter -> {adapter_dir(cfg, name)}")
    dist.barrier()


def _hist(cfg, name): return os.path.join(run_paths(cfg)["train"], f"{name}.jsonl")


def _dtype(cfg):
    from g2t.model import resolve_dtype
    return resolve_dtype(cfg)


# ======================================================================== WTQ: ours / direct control
def stage_wtq_sft(cfg, log, which):
    from g2t.model import add_trainable_adapter
    from g2t.prompts import ours_prompt, ours_target, direct_target
    from g2t.train import tokenize_pairs, lora_sft
    D = wtq_data(cfg); assert D, "run the data stage first"
    teacher = D["teacher"]["train"]; exs = [e for e in D["train"] if e.id in D["sft_ids"]]
    pairs, kinds = [], {"PROG": 0, "ANS": 0}
    for e in exs:
        t, kind = ours_target(e, teacher) if which == "ours" else direct_target(e)
        if t: pairs.append((ours_prompt(e), t)); kinds[kind] += 1
    log.info(f"[wtq {which} SFT] {len(pairs)} pairs | {kinds}")
    model, tok = _train_model(cfg, log); name = f"wtq_{which}_sft"
    model = add_trainable_adapter(model, cfg, name)
    data, skipped = tokenize_pairs(tok, pairs, cfg["sft"]["max_tokens"]); log.info(f"tokenized {len(data)} (skipped {skipped})")
    lora_sft(model, tok, data, dict(cfg["sft"], seed=cfg["seed"]), _dtype(cfg), log, _hist(cfg, name), name)
    _save_adapter(model, cfg, name, log)


def stage_wtq_rft(cfg, log):
    """Improvement: rejection-sampling fine-tuning. The SFT model samples k programs for SFT-split questions the teacher did NOT cover;
    programs whose execution matches the gold answer become new PROG targets (coverage was the main limit in v8: ~37% of questions).
    Then 1 epoch of SFT on {new PROG pairs} + an equal-size replay of the original pairs."""
    from g2t.model import add_trainable_adapter, activate_trainable, HFGen
    from g2t.prompts import ours_prompt, ours_target
    from g2t.teacher import ekey
    from g2t.engine import run as run_expr
    from g2t.wtq_eval import official_match
    from g2t.train import tokenize_pairs, lora_sft
    D = wtq_data(cfg); T = D["teacher"]["train"]; Rf = cfg["improvements"]["rft"]
    unc = [e for e in D["train"] if e.id in D["sft_ids"] and ekey(e) not in T and len(e.answers) == 1]
    random.Random(cfg["seed"]).shuffle(unc); unc = unc[:Rf.get("max_questions") or len(unc)]
    mine = unc[dist.rank()::dist.world()]
    model, tok = _train_model(cfg, log); name = "wtq_ours_rft"
    model = add_trainable_adapter(model, cfg, name, init_from=adapter_dir(cfg, "wtq_ours_sft"))
    gen = HFGen(model, tok, cfg); found = {}
    log.info(f"[RFT] sampling {Rf['k']} programs for {len(mine)} uncovered questions on this GPU ({len(unc)} total)")
    for i in range(0, len(mine), 256):
        chunk = mine[i:i + 256]
        outs = gen.generate([ours_prompt(e) + " PROG:" for e in chunk], cfg["eval"]["prog_max_new"], sample=True, temp=Rf["temp"], n=Rf["k"],
                            adapter=name, log=log, tag="rft")
        for e, samples in zip(chunk, outs):
            for raw, _ in samples:
                expr = raw.strip().split("\n")[0].strip(); ans, err = run_expr(expr, e.headers, e.rows)
                if ans is not None and official_match([ans], e.answers): found[ekey(e)] = dict(expr=expr, depth=-1, source="rft"); break
    activate_trainable(model, name)
    P = run_paths(cfg); save_json(found, os.path.join(P["data"], f"wtq_rft_programs.rank{dist.rank()}.json")); dist.barrier()
    allf = {}
    for r in range(dist.world()): allf.update(load_json(os.path.join(P["data"], f"wtq_rft_programs.rank{r}.json")))
    if dist.is_main(): save_json(allf, os.path.join(P["data"], "wtq_rft_programs.json"))
    log.info(f"[RFT] found executable gold-matching programs for {len(allf)}/{len(unc)} previously uncovered questions ({len(allf)/max(1,len(unc)):.1%})")
    by = {ekey(e): e for e in unc}
    new = [(ours_prompt(by[k]), " PROG: " + v["expr"]) for k, v in allf.items() if k in by]
    old = []
    for e in D["train"]:
        if e.id in D["sft_ids"]:
            t, _ = ours_target(e, T)
            if t: old.append((ours_prompt(e), t))
    random.Random(cfg["seed"]).shuffle(old); pairs = new + old[:max(len(new), Rf.get("min_replay", 2000))]
    data, _ = tokenize_pairs(tok, pairs, cfg["sft"]["max_tokens"])
    S = dict(cfg["sft"], seed=cfg["seed"]); S.update(epochs=Rf["epochs"], lr=cfg["sft"]["lr"] * Rf["lr_mult"])
    lora_sft(model, tok, data, S, _dtype(cfg), log, _hist(cfg, name), name)
    _save_adapter(model, cfg, name, log)


def stage_wtq_grpo(cfg, log, which):
    from g2t.model import add_trainable_adapter, eos_ids
    from g2t.prompts import ours_prompt
    from g2t.systems import reward_wtq_hybrid, reward_wtq_direct
    from g2t.train import grpo
    from g2t.teacher import ekey
    D = wtq_data(cfg); teacher = D["teacher"]["train"]
    pool = [e for e in D["train"] if e.id in D["rl_ids"]]
    cov = [e for e in pool if ekey(e) in teacher] or pool; unc = [e for e in pool if ekey(e) not in teacher] or pool
    R = dict(cfg["grpo"]); frac = R["frac_covered"]
    def pool_fn(rng, n): return [rng.choice(cov if rng.random() < frac else unc) for _ in range(n)]
    init = f"wtq_{which}_sft"
    if which == "ours" and cfg["improvements"]["rft"]["enabled"] and os.path.isdir(adapter_dir(cfg, "wtq_ours_rft")): init = "wtq_ours_rft"
    model, tok = _train_model(cfg, log); name = f"wtq_{which}_grpo"
    model = add_trainable_adapter(model, cfg, name, init_from=adapter_dir(cfg, init))
    log.info(f"[wtq {which} GRPO] init {init} | RL pool {len(pool)} disjoint questions ({len(cov)} covered / {len(unc)} uncovered)")
    grpo(model, tok, eos_ids(model, tok), ours_prompt, reward_wtq_hybrid if which == "ours" else reward_wtq_direct, pool_fn, R,
         _dtype(cfg), log, _hist(cfg, name), name, seed=cfg["seed"], train_name=name)
    _save_adapter(model, cfg, name, log)


# ======================================================================== Sparks (Stoisser et al., TRL @ ACL 2025)
def _sparks_quant(cfg): return "4bit" if cfg["sparks"]["quant"] == "4bit" else None


def stage_sparks_sft(cfg, log):
    """Stage 1 as in the paper: QLoRA adapters, lr 2e-4, batch size 1, on the authors' 3,174 verified CoT traces."""
    from g2t.model import add_trainable_adapter, chat
    from g2t.sparks_data import load_sparks_pairs
    from g2t.train import tokenize_pairs, lora_sft
    model, tok = _train_model(cfg, log, quant=_sparks_quant(cfg))
    pairs = load_sparks_pairs(cfg, tok, chat, log)
    if cfg["sparks"].get("sft_n"): pairs = pairs[:cfg["sparks"]["sft_n"]]
    model = add_trainable_adapter(model, cfg, "sparks_sft", lora=cfg["sparks"]["lora"])
    data, skipped = tokenize_pairs(tok, pairs, cfg["sparks"]["max_tokens"]); log.info(f"tokenized {len(data)} (skipped {skipped})")
    S = dict(cfg["sft"], seed=cfg["seed"]); S.update(cfg["sparks"]["sft"])
    lora_sft(model, tok, data, S, _dtype(cfg), log, _hist(cfg, "sparks_sft"), "sparks_sft")
    _save_adapter(model, cfg, "sparks_sft", log)


def _sparks_grpo_common(cfg, log, name, prompt_fn, reward_fn, pool_fn, note):
    from g2t.model import add_trainable_adapter, eos_ids, load_eval_adapters, activate_trainable
    from g2t.train import grpo
    model, tok = _train_model(cfg, log, quant=_sparks_quant(cfg))
    model = add_trainable_adapter(model, cfg, name, init_from=adapter_dir(cfg, "sparks_sft"))
    R = dict(cfg["grpo"]); R.update(cfg["sparks"]["grpo"]); ref = None
    if R.get("kl_beta", 0) > 0:
        model.load_adapter(adapter_dir(cfg, "sparks_sft"), adapter_name="sparks_ref", is_trainable=False); ref = "sparks_ref"
        activate_trainable(model, name)
    log.info(f"[{name}] {note} | KL reference: {ref}")
    grpo(model, tok, eos_ids(model, tok), prompt_fn(tok), reward_fn, pool_fn, R, _dtype(cfg), log, _hist(cfg, name), name,
         seed=cfg["seed"], train_name=name, ref_adapter=ref)
    _save_adapter(model, cfg, name, log)


def stage_sparks_grpo(cfg, log):
    """Stage 2 adapted to WTQ (WTQ has no gold SQL): reward = the SQL result matches the gold answer (official evaluator). Same RL pool as ours."""
    from g2t.model import chat
    from g2t.prompts import sparks_wtq_messages, fit_rows
    from g2t.systems import reward_wtq_sql
    D = wtq_data(cfg); pool = [e for e in D["train"] if e.id in D["rl_ids"]]; mx = cfg["wtq"]["max_prompt_tokens_train"]
    def prompt_fn(tok):
        n_tok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
        return lambda e: fit_rows(lambda k: chat(tok, sparks_wtq_messages(e, cfg["sparks"]["show_rows"], k)), len(e.rows), n_tok, mx)[0]
    _sparks_grpo_common(cfg, log, "sparks_grpo", prompt_fn, reward_wtq_sql, lambda rng, n: [rng.choice(pool) for _ in range(n)],
                        f"WTQ adaptation: {len(pool)} RL questions (same pool as ours), execution reward")


def stage_sparks_bird_grpo(cfg, log):
    """Stage 2 as in the paper: GRPO on BIRD with string-match + component-F1 + execution rewards (LLM-judge rewards replaced/dropped)."""
    from g2t.model import chat, load_tokenizer
    from g2t.sparks_data import load_bird, bird_messages, bird_reward
    try: bird = load_bird(cfg, log)
    except FileNotFoundError as e:
        if cfg["sparks"]["bird_required"]: raise
        log.warning(f"{e}\n-> sparks_paper (BIRD stage 2) skipped; download BIRD to enable it (sparks.bird_required: true makes this an error)"); return
    if cfg["sparks"].get("bird_n"): bird = bird[:cfg["sparks"]["bird_n"]]
    tok = load_tokenizer(model_path(cfg, "base")); mx = cfg["wtq"]["max_prompt_tokens_train"]
    bird = [e for e in bird if len(tok(chat(tok, bird_messages(e)), add_special_tokens=False).input_ids) <= mx]
    _sparks_grpo_common(cfg, log, "sparks_bird_grpo", lambda tok: (lambda e: chat(tok, bird_messages(e))), bird_reward,
                        lambda rng, n: [rng.choice(bird) for _ in range(n)], f"BIRD (paper setting): {len(bird)} questions")


# ======================================================================== graph
def stage_graph_sft(cfg, log):
    from g2t import graph_tasks as G
    from g2t.model import add_trainable_adapter
    from g2t.train import tokenize_pairs, lora_sft
    gs = graph_splits(cfg); rep = cfg["graph"]["rep"]
    log.info(f"[graph SFT] {len(gs['train'])} training questions incl. {sum(q.split == 'train_aug' for q in gs['train'])} deletion-augmented | "
             f"NONE share {np.mean([q.gold == 'NONE' for q in gs['train']]):.0%}")
    model, tok = _train_model(cfg, log)
    for which, out in (("ours", "nodes"), ("direct", "ans")):
        name = f"graph_{which}_sft"
        if os.path.isdir(adapter_dir(cfg, name)): log.info(f"{name} exists, skipping"); continue
        if which not in cfg["graph"]["train_arms"]: continue
        pairs = [(G.make_prompt(q, rep), G.target(G.view(q, rep), out)) for q in gs["train"]]
        model = add_trainable_adapter(model, cfg, name)
        data, _ = tokenize_pairs(tok, pairs, cfg["sft"]["max_tokens"])
        S = dict(cfg["sft"], seed=cfg["seed"]); S.update(cfg["graph"]["sft"])
        lora_sft(model, tok, data, S, _dtype(cfg), log, _hist(cfg, name), name)
        _save_adapter(model, cfg, name, log)


def stage_graph_grpo(cfg, log):
    from g2t import graph_tasks as G
    from g2t.model import add_trainable_adapter, eos_ids
    from g2t.systems import reward_graph
    from g2t.train import grpo
    gs = graph_splits(cfg); rep = cfg["graph"]["rep"]; R = dict(cfg["grpo"]); R.update(cfg["graph"]["grpo"])
    def pool_fn(rng, n):
        k = int(round(n * cfg["graph"]["grpo_mid_frac"])); return rng.sample(gs["pool_mid"], k) + rng.sample(gs["pool_id"], n - k)
    model, tok = _train_model(cfg, log)
    for which, out in (("ours", "nodes"), ("direct", "ans")):
        name = f"graph_{which}_grpo"
        if os.path.isdir(adapter_dir(cfg, name)): log.info(f"{name} exists, skipping"); continue
        if which not in cfg["graph"]["train_arms"]: continue
        model = add_trainable_adapter(model, cfg, name, init_from=adapter_dir(cfg, f"graph_{which}_sft"))
        Rw = dict(R); Rw["max_new"] = cfg["eval"]["prog_max_new"] if out == "nodes" else 16
        grpo(model, tok, eos_ids(model, tok), lambda q: G.make_prompt(q, rep), reward_graph(out, rep), pool_fn, Rw, _dtype(cfg), log,
             _hist(cfg, name), name, seed=cfg["seed"], train_name=name)
        _save_adapter(model, cfg, name, log)


# ======================================================================== evaluation (sharded)
def _make_gen(cfg, model_key, adapters, log):
    from g2t.model import load_tokenizer, load_hf, load_eval_adapters, HFGen, VLLMGen
    path = model_path(cfg, model_key); tok = load_tokenizer(path)
    if cfg["eval"]["backend"] == "vllm":
        try:
            g = VLLMGen(path, tok, cfg, adapters); log.info(f"vLLM backend for {path} with adapters {list(adapters)}"); return g, tok
        except Exception as e:
            log.warning(f"vLLM unavailable ({type(e).__name__}: {str(e)[:200]}) -> falling back to HF generate")
    model = load_hf(path, cfg, log); model = load_eval_adapters(model, adapters) if adapters else model
    return HFGen(model, tok, cfg), tok


def _close(gen):
    import gc, torch
    if hasattr(gen, "close"): gen.close()
    else: del gen.model
    gc.collect(); torch.cuda.empty_cache()


def eval_sets(cfg):
    out = {}
    if "wtq" in cfg["benchmarks"]:
        D = wtq_data(cfg); out["wtq"] = D["test"] if cfg["wtq"]["eval_split"] == "test" else D["dev"]
    if "crtqa" in cfg["benchmarks"]:
        from g2t.crtqa_data import load_crtqa
        out["crtqa"] = load_crtqa(cfg)
    if "graph" in cfg["benchmarks"]:
        gs = graph_splits(cfg); out["graph"] = gs["test_id"] + gs["test_ood"] if cfg["graph"].get("eval_split", "test") == "test" else gs["val"]
    return out


def _uid(bench, x): return x.uid if bench == "graph" else x.id


def run_systems_on(cfg, log, items, out_dir, label, shard, nshards):
    """items: (sysd, examples, variant). Shard i of n gets every n-th instance (sorted by key). One JSONL per system and shard (resumable)."""
    from g2t.systems import predict_wtq, predict_graph
    groups = {}
    for sysd, exs, variant in items: groups.setdefault(sysd["model"], []).append((sysd, exs, variant))
    sfx = f".shard{shard}of{nshards}" if nshards > 1 else ""
    for model_key, lst in groups.items():
        todo = []
        for sysd, exs, variant in lst:
            mine = sorted(exs, key=lambda x: variant["key"](x))[shard::nshards]
            path = os.path.join(out_dir, sysd["bench"], f"{sysd['name']}{variant['suffix']}{sfx}.jsonl")
            have = {r.get("key") for r in read_jsonl(path)}
            rest = [x for x in mine if variant["key"](x) not in have]
            if rest: todo.append((sysd, rest, variant, path, len(mine)))
            else: log.info(f"[{label}] {sysd['bench']}/{sysd['name']}{variant['suffix']}{sfx}: complete ({len(mine)})")
        if not todo: continue
        adapters = {s["adapter"]: adapter_dir(cfg, s["adapter"]) for s, *_ in todo if s["adapter"]}
        missing = [a for a, p in adapters.items() if not os.path.isdir(p)]
        if missing:
            log.error(f"missing adapters {missing} -> run their training stages first; skipping the systems that need them")
            todo = [t for t in todo if t[0]["adapter"] not in missing]; adapters = {a: p for a, p in adapters.items() if a not in missing}
            if not todo: continue
        gen, tok = _make_gen(cfg, model_key, adapters, log)
        try:
            for sysd, rest, variant, path, n_all in todo:
                t0 = time.time(); log.info(f"[{label}] {sysd['bench']}/{sysd['name']}{variant['suffix']}{sfx}: {len(rest)} of {n_all} to do (model {model_key}, adapter {sysd['adapter']})")
                CH = cfg["eval"]["chunk"]
                with JsonlWriter(path, "a") as W:
                    for i in range(0, len(rest), CH):
                        chunk = rest[i:i + CH]
                        recs = (predict_wtq(gen, tok, cfg, sysd, chunk, log, forced=variant.get("forced")) if sysd["bench"] in ("wtq", "crtqa")
                                else predict_graph(gen, tok, cfg, sysd, chunk, log))
                        for x, r in zip(chunk, recs): r["key"] = variant["key"](x); r.update(variant.get("extra", lambda x: {})(x)); W.write(r)
                done = read_jsonl(path); acc = np.mean([r["correct"] for r in done]) if done else 0
                log.info(f"[{label}] {sysd['bench']}/{sysd['name']}{variant['suffix']}{sfx}: accuracy {acc:.4f} on {len(done)} ({(time.time()-t0)/60:.1f} min)")
        finally:
            _close(gen)


def stage_eval(cfg, log, shard=0, nshards=1):
    from g2t.systems import registry
    reg = registry(cfg); sets = eval_sets(cfg); P = run_paths(cfg)
    only = cfg["eval"].get("only_models")
    items = [(s, sets[s["bench"]], dict(suffix="", key=lambda x, b=s["bench"]: _uid(b, x))) for s in reg.values()
             if s["bench"] in sets and (not only or s["model"] in only)]
    run_systems_on(cfg, log, items, P["instances"], "eval", shard, nshards)


def stage_cf(cfg, log, shard=0, nshards=1):
    from g2t.systems import registry, make_cf_ast
    from g2t.wtq_data import TableExample
    from g2t.teacher import ekey
    from g2t import graph_tasks as G
    reg = registry(cfg); P = run_paths(cfg); C = cfg["cf"]; items = []
    if "wtq" in cfg["benchmarks"]:
        D = wtq_data(cfg); split = "test" if cfg["wtq"]["eval_split"] == "test" else "validation"; T = D["teacher"][split]
        base = D["test"] if split == "test" else D["dev"]
        sel = [e for e in base if ekey(e) in T]; random.Random(0).shuffle(sel); sel = sel[:C["wtq_n"]]
        exs, wmeta = [], {}
        for i, e in enumerate(sel):
            exs.append(e); wmeta[e.id] = dict(cf_kind="orig", orig=e.id, expected=e.answers)
            for s in range(C["wtq_seeds"]):
                cf = make_cf_ast(e, T[ekey(e)]["expr"], random.Random(1000 * s + i))
                if cf["sens"] is None: continue
                for kind in ("sens", "far", "near"):
                    if cf[kind] is None: continue
                    rows = cf[kind][0] if kind == "sens" else cf[kind]; exp = [cf["sens"][1]] if kind == "sens" else e.answers
                    x = TableExample(f"{e.id}:{kind}{s}", e.question, e.headers, rows, exp); exs.append(x)
                    wmeta[x.id] = dict(cf_kind=kind, orig=e.id, expected=exp)
        log.info(f"[cf wtq] {len(sel)} covered questions -> {len(exs)} tables (orig + sens/far/near edits)")
        if dist.is_main() or nshards == 1: save_json(dict(meta=wmeta), os.path.join(P["cf"], "wtq_meta.json"))
        extra = lambda x, m=wmeta: m[x.id]
        for name in C["wtq_systems"]:
            if ("wtq", name) not in reg: continue
            s = reg[("wtq", name)]; items.append((s, exs, dict(suffix="", key=lambda x: x.id, extra=extra)))
            if s["kind"] == "wtq_hybrid":
                for f in ("prog", "ans"): items.append((s, exs, dict(suffix=f"__forced_{f}", key=lambda x: x.id, extra=extra, forced=f)))
    if "graph" in cfg["benchmarks"]:
        gs = graph_splits(cfg); trips = G.build_cf(gs["test_id"], C["graph_n"], cfg["graph"]["data_seed"] + 7)
        exs = [x for t in trips for x in t]; gmeta = {}
        for q, s_, i_ in trips:
            gmeta[q.uid] = dict(cf_kind="orig", orig=q.uid); gmeta[s_.uid] = dict(cf_kind="sens", orig=q.uid, disconnect=s_.gold == "NONE")
            gmeta[i_.uid] = dict(cf_kind="inv", orig=q.uid)
        log.info(f"[cf graph] {len(trips)} triples (orig, path-edge deletion, off-path deletion)")
        for name in C["graph_systems"]:
            if ("graph", name) in reg: items.append((reg[("graph", name)], exs, dict(suffix="", key=lambda x: x.uid, extra=lambda x, m=gmeta: m[x.uid])))
    run_systems_on(cfg, log, items, P["cf"], "cf", shard, nshards)


# ======================================================================== main
def parse_sets(sets):
    out = {}
    for s in sets or []:
        k, v = s.split("=", 1); d = out; ks = k.split(".")
        for kk in ks[:-1]: d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", action="append", help="override a config value, e.g. --set sft.lr=5e-5 (repeatable)")
    ap.add_argument("--shard", type=int, default=None); ap.add_argument("--num_shards", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, smoke=True if a.smoke else (False if a.full else None), overrides=parse_sets(a.set))
    dist.init()
    os.environ.setdefault("HF_HOME", cfg["paths"]["hf_cache"]); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if cfg.get("offline", True): os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    P = run_paths(cfg)
    if dist.is_main(): dump(cfg, os.path.join(P["root"], "config.resolved.yaml"))
    shard = a.shard if a.shard is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) if a.num_shards else 0
    nshards = a.num_shards or 1
    if dist.world() > 1 and a.stage in SHARDED_STAGES:          # launched with torchrun for eval: rank = shard, HF backend (vLLM picks GPU 0)
        shard, nshards = dist.rank(), dist.world(); cfg["eval"]["backend"] = "hf"
    from g2t.report import stage_report
    fns = {"data": stage_data, "wtq_ours_sft": lambda c, l: stage_wtq_sft(c, l, "ours"), "wtq_direct_sft": lambda c, l: stage_wtq_sft(c, l, "direct"),
           "wtq_ours_rft": stage_wtq_rft, "wtq_ours_grpo": lambda c, l: stage_wtq_grpo(c, l, "ours"),
           "wtq_direct_grpo": lambda c, l: stage_wtq_grpo(c, l, "direct"), "sparks_sft": stage_sparks_sft, "sparks_grpo": stage_sparks_grpo,
           "sparks_bird_grpo": stage_sparks_bird_grpo, "graph_sft": stage_graph_sft, "graph_grpo": stage_graph_grpo,
           "eval": lambda c, l: stage_eval(c, l, shard, nshards), "cf": lambda c, l: stage_cf(c, l, shard, nshards), "report": stage_report}
    enabled = {s for b in cfg["benchmarks"] for s in cfg["systems"].get(b, [])}
    needs = {"wtq_ours_sft": {"ours_sft", "ours_rft", "ours_grpo", "ours_grpo_sc"}, "wtq_direct_sft": {"direct_sft", "direct_grpo"},
             "wtq_ours_rft": {"ours_rft"} | ({"ours_grpo", "ours_grpo_sc"} if cfg["improvements"]["rft"]["enabled"] else set()),
             "wtq_ours_grpo": {"ours_grpo", "ours_grpo_sc"}, "wtq_direct_grpo": {"direct_grpo"},
             "sparks_sft": {"sparks", "sparks_sft", "sparks_paper"}, "sparks_grpo": {"sparks"}, "sparks_bird_grpo": {"sparks_paper"},
             "graph_sft": {"ours_sft", "direct_sft"}, "graph_grpo": {"ours_grpo", "direct_grpo"}}
    bench_of = {"graph_sft": "graph", "graph_grpo": "graph"}
    stages = STAGES if a.stage == "all" else [a.stage]
    if dist.world() > 1 and a.stage == "all": raise SystemExit("with torchrun, run one training stage at a time (see slurm/submit_all.sh)")
    for st in stages:
        log = get_logger(cfg, st + (f"_shard{shard}" if nshards > 1 else ""))
        if dist.world() > 1 and st not in TRAIN_STAGES and st not in SHARDED_STAGES and not dist.is_main(): continue
        if st in needs:
            b = bench_of.get(st)
            on = lambda k: set(cfg["systems"].get(k, [])) if k in cfg["benchmarks"] else set()
            # CRT-QA (zero-shot) re-uses the WTQ-trained adapters; graph Sparks re-uses the WTQ/BIRD-trained Sparks adapters
            pool = on(b) if b else (on("wtq") | on("crtqa") | ({"sparks", "sparks_paper"} & on("graph")))
            if (b and b not in cfg["benchmarks"]) or not (needs[st] & pool):
                log.info(f"stage {st} not needed by the enabled benchmarks/systems -> skipped"); continue
        mark = st + (f".shard{shard}of{nshards}" if (st in SHARDED_STAGES and nshards > 1) else "")
        if is_done(cfg, mark) and not a.force: log.info(f"stage {mark} already done (use --force to redo)"); continue
        if dist.is_main() or st in SHARDED_STAGES:
            log.info(f"===== stage {mark} | smoke={cfg['smoke']} | GPUs in this job {dist.world()} | run dir {P['root']} =====")
        seed_all(cfg["seed"]); t0 = time.time()
        try:
            fns[st](cfg, log)
        except Exception:
            log.error(f"stage {mark} FAILED (rank {dist.rank()}):\n{traceback.format_exc()}"); sys.exit(1)
        if dist.is_main() or st in SHARDED_STAGES: mark_done(cfg, mark, dict(minutes=round((time.time() - t0) / 60, 1), smoke=cfg["smoke"], world=dist.world()))
        if dist.is_main() or st in SHARDED_STAGES: log.info(f"===== stage {mark} finished in {(time.time()-t0)/60:.1f} min =====")
        dist.barrier()
    dist.finish()


if __name__ == "__main__":
    main()
