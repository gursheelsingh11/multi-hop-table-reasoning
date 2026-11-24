"""LoRA / QLoRA SFT and GRPO, single- or multi-GPU (torchrun).

GRPO = group-relative advantages + clipped token-level loss (DAPO-style decoupled clipping), no critic. Optional:
  * KL to a frozen reference adapter (k3 estimator, coefficient kl_beta) — used for the Sparks baseline (the paper's objective has a KL term);
  * dynamic sampling (DAPO): oversample questions and keep groups whose rewards differ, so updates are not wasted on all-0 / all-1 groups.
Multi-GPU: each rank trains on its own data shard / its own questions; LoRA gradients are averaged across ranks before every step.
"""
import math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from .logs import JsonlWriter
from . import dist

AMP = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def tokenize_pairs(tok, pairs, maxtok):
    data, skipped = [], 0
    for pr, tg in pairs:
        a = tok(pr, add_special_tokens=False).input_ids; b = tok(tg + tok.eos_token, add_special_tokens=False).input_ids
        if len(a) + len(b) <= maxtok: data.append((a + b, len(b)))
        else: skipped += 1
    return data, skipped


def _batch(tok, ids_, data, dev):
    seqs = [data[i] for i in ids_]; L = max(len(s) for s, _ in seqs); T = max(nt for _, nt in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
    tm = torch.zeros((len(seqs), L), dtype=torch.bool)
    for k, (x, nt) in enumerate(seqs):
        ids[k, L - len(x):] = torch.tensor(x); att[k, L - len(x):] = 1; tm[k, L - nt:] = True
    return ids.to(dev), att.to(dev), tm.to(dev), T


def _ensure_grads(params):
    for p in params:
        if p.grad is None: p.grad = torch.zeros_like(p)


def _check_dtype_for_dist(dtype):
    if dist.world() > 1 and dtype == "float16":
        raise RuntimeError("multi-GPU training needs bf16-capable GPUs (A100/A40/A6000/L40/H100/RTX30xx+). Use fp16 GPUs (V100/T4) for evaluation shards only.")


def lora_sft(model, tok, data, S, dtype, log, hist_path, tag):
    """S: dict(epochs, lr, micro, accum, warmup_frac, max_tokens_per_micro). Length-bucketed batches; loss on target tokens only.
    Effective batch = micro x accum x world_size (sequences)."""
    _check_dtype_for_dist(dtype)
    params = [p for p in model.parameters() if p.requires_grad]; dev = model.device
    opt = torch.optim.AdamW(params, lr=S["lr"], weight_decay=S.get("weight_decay", 0.0))
    fp16 = dtype == "float16"; scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    micro, accum = S["micro"], S["accum"]; budget = S.get("max_tokens_per_micro", 10 ** 9); W, Rk = dist.world(), dist.rank()
    def chunks_of(order):
        out, cur = [], []
        for i in order:
            L = len(data[i][0])
            if cur and (len(cur) + 1 > micro or (len(cur) + 1) * L > budget): out.append(cur); cur = []
            cur.append(i)
        if cur: out.append(cur)
        return out
    n_chunks = len(chunks_of(sorted(range(len(data)), key=lambda i: len(data[i][0])))) // W
    total = max(1, n_chunks * S["epochs"] // accum); warm = max(1, int(S.get("warmup_frac", 0.03) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / total))
    model.train(); model.config.use_cache = False; step, run, cnt, bad, t0 = 0, 0.0, 0, 0, time.time()
    if dist.is_main():
        log.info(f"[{tag}] SFT: {len(data)} examples on {W} GPU(s), {S['epochs']} epoch(s), {total} optimizer steps, lr {S['lr']}, "
                 f"effective batch {micro}x{accum}x{W} sequences")
    H = JsonlWriter(hist_path) if dist.is_main() else None
    rng = random.Random(S.get("seed", 0))
    for ep in range(S["epochs"]):
        chunks = chunks_of(sorted(range(len(data)), key=lambda i: len(data[i][0]))); rng.shuffle(chunks)   # identical order on every rank
        chunks = chunks[:len(chunks) // W * W][Rk::W]                                                      # this rank's equal-sized share
        for ci, ch in enumerate(chunks):
            ids, att, tm, T = _batch(tok, ch, data, dev); K = T + 1
            with torch.autocast("cuda", dtype=AMP.get(dtype, torch.bfloat16)):
                out = model(input_ids=ids, attention_mask=att, logits_to_keep=K)
            lg = out.logits[:, -K:, :][:, :-1, :].float(); tgt = ids[:, -T:]; m = tm[:, -T:]
            loss = (F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1), reduction="none").view_as(tgt) * m).sum() / m.sum()
            if torch.isfinite(loss):
                scaler.scale(loss / accum).backward(); run += loss.item(); cnt += 1
            else:
                bad += 1
                if bad > 50: raise RuntimeError("loss NaN/inf repeatedly")
            if (ci + 1) % accum == 0 or ci == len(chunks) - 1:
                _ensure_grads(params); dist.all_reduce_mean_grads(params)
                scaler.unscale_(opt); gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                L_ = dist.scalar_sum(run / max(1, cnt)) / W
                if H: H.write(dict(step=step, epoch=ep, loss=L_, lr=sched.get_last_lr()[0], grad_norm=gn, sec=time.time() - t0))
                if dist.is_main() and (step % S.get("log_every", 20) == 0 or step == total):
                    log.info(f"[{tag}] step {step}/{total} loss {L_:.4f} lr {sched.get_last_lr()[0]:.2e} ({(time.time()-t0)/60:.1f} min)")
                run, cnt = 0.0, 0
            del ids, att, tm, out, lg
    if H: H.close()
    model.eval(); model.config.use_cache = True
    if dist.is_main(): log.info(f"[{tag}] SFT done in {(time.time()-t0)/60:.1f} min (non-finite micro-batches skipped: {bad})")


# ------------------------------------------------------------------ GRPO
def _eos_first(comp, EOS_T):
    is_eos = torch.isin(comp, EOS_T); has = is_eos.any(1)
    first = torch.where(has, is_eos.float().argmax(1), torch.full((comp.shape[0],), comp.shape[1] - 1, dtype=torch.long, device=comp.device))
    return first, has


def group_advantages(R, G):
    R = np.asarray(R, float).reshape(-1, G); mu, sd = R.mean(1, keepdims=True), R.std(1, keepdims=True)
    return np.where(sd > 1e-8, (R - mu) / np.maximum(sd, 1e-8), 0.0).reshape(-1), (sd[:, 0] > 1e-8)


def completion_logp(model, ids, att, P, C, dtype):
    pos = (att.cumsum(-1) - 1).clamp(min=0); K = C + 1
    with torch.autocast("cuda", dtype=AMP.get(dtype, torch.bfloat16)):
        out = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K, use_cache=False)
    lg = out.logits[:, -K:, :][:, :-1, :].float()
    return lg.gather(-1, ids[:, P:P + C].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1)


@torch.no_grad()
def _rollouts(model, tok, prompts, G, R, EOS_T):
    model.eval()
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=R["temp"], top_p=R.get("top_p", 1.0), top_k=0, repetition_penalty=1.0,
                         max_new_tokens=R["max_new"], num_return_sequences=G, pad_token_id=tok.pad_token_id, use_cache=True)
    P = enc["input_ids"].shape[1]; comp = out[:, P:]; first, has = _eos_first(comp, EOS_T)
    cmask = torch.arange(comp.shape[1], device=comp.device)[None] <= first[:, None]
    texts = [tok.decode(comp[i, :first[i] + 1], skip_special_tokens=True) for i in range(comp.shape[0])]
    return enc, comp, cmask, has, texts


def grpo(model, tok, eos, prompt_fn, reward_fn, pool_fn, R, dtype, log, hist_path, tag, seed=0, train_name=None, ref_adapter=None):
    """R: dict(steps, B (global questions/step), G, micro, minibatches, lr, clip_low, clip_high, temp, max_new,
               kl_beta=0, dynamic_sampling=False, oversample=2).
    reward_fn(example, text) -> (reward, tag). pool_fn(rng, n) -> list of n examples.
    train_name: name of the trainable adapter (needed when ref_adapter is used: reference log-probs are computed with ref_adapter)."""
    from .model import activate_trainable
    _check_dtype_for_dist(dtype)
    EOS_T = torch.tensor(eos, device=model.device); G, MIC, NMB = R["G"], R["micro"], R["minibatches"]; W = dist.world()
    B = max(1, math.ceil(R["B"] / W)); beta = float(R.get("kl_beta", 0.0)); dyn = bool(R.get("dynamic_sampling", False)); os_ = R.get("oversample", 2)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=R["lr"], weight_decay=0.0)
    fp16 = dtype == "float16"; scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    rng = random.Random(seed * 1000 + dist.rank()); t_start = time.time(); model.config.use_cache = False
    if dist.is_main():
        log.info(f"[{tag}] GRPO: {R['steps']} steps x {B*W} questions ({B}/GPU on {W} GPUs) x {G} rollouts | lr {R['lr']} | clip {R['clip_low']}/{R['clip_high']} "
                 f"| KL beta {beta}{' vs ' + ref_adapter if beta > 0 else ''} | dynamic sampling {dyn} | max_new {R['max_new']} | temp {R['temp']}")
    hist = []; H = JsonlWriter(hist_path) if dist.is_main() else None
    for step in range(1, R["steps"] + 1):
        t0 = time.time()
        cand = pool_fn(rng, B * os_ if dyn else B)
        enc, comp, cmask, has, texts = _rollouts(model, tok, [prompt_fn(e) for e in cand], G, R, EOS_T)
        RT = [reward_fn(cand[k // G], t) for k, t in enumerate(texts)]; Rw = [r for r, _ in RT]
        A_all, nz_all = group_advantages(Rw, G)
        # ---- dynamic sampling: keep groups with signal first, exactly B groups per rank (fixed shapes -> ranks stay in step)
        order = sorted(range(len(cand)), key=lambda g: (not nz_all[g], g))[:B] if dyn else list(range(len(cand)))
        rows = torch.tensor([g * G + j for g in order for j in range(G)], device=model.device)
        tags = Counter(RT[int(i)][1] for i in rows.tolist()); Rk = [Rw[int(i)] for i in rows.tolist()]
        A = A_all[rows.cpu().numpy()]; nz = nz_all[order]
        P = enc["input_ids"].shape[1]
        comp_k, cmask_k = comp[rows], cmask[rows]; N, C = comp_k.shape; cm = cmask_k.float()
        q_idx = torch.tensor([g for g in order for _ in range(G)], device=model.device)
        ids = torch.cat([enc["input_ids"][q_idx], comp_k], 1); att = torch.cat([enc["attention_mask"][q_idx], cmask_k.long()], 1)
        A_t = torch.tensor(A, dtype=torch.float32, device=model.device)
        with torch.no_grad():
            old = torch.cat([completion_logp(model, ids[i:i + MIC], att[i:i + MIC], P, C, dtype) for i in range(0, N, MIC)])
            ref = None
            if beta > 0 and ref_adapter:
                model.set_adapter(ref_adapter)
                ref = torch.cat([completion_logp(model, ids[i:i + MIC], att[i:i + MIC], P, C, dtype) for i in range(0, N, MIC)])
                activate_trainable(model, train_name)
        model.train(); gns, clipfrac, kls = [], [], []
        for mb in torch.randperm(N, device=model.device).chunk(NMB):
            keep = mb[A_t[mb] != 0] if beta == 0 else mb
            denom = torch.tensor([cm[mb].sum().item()], device=model.device); dist.all_reduce_sum(denom); denom = denom.clamp(min=1.0) / W
            for ch in keep.split(MIC):
                lp = completion_logp(model, ids[ch], att[ch], P, C, dtype); ratio = torch.exp(lp - old[ch]); a_ = A_t[ch].unsqueeze(1)
                obj = torch.minimum(ratio * a_, torch.clamp(ratio, 1 - R["clip_low"], 1 + R["clip_high"]) * a_)
                if ref is not None:
                    d = ref[ch] - lp; kl = torch.exp(d) - d - 1; obj = obj - beta * kl
                    kls.append(float((kl * cm[ch]).sum().item() / max(1.0, cm[ch].sum().item())))
                loss = -(obj * cm[ch]).sum() / denom
                if torch.isfinite(loss): scaler.scale(loss).backward()
                clipfrac.append((((ratio > 1 + R["clip_high"]) | (ratio < 1 - R["clip_low"])).float() * cm[ch]).sum().item() / max(1.0, cm[ch].sum().item()))
            _ensure_grads(params); dist.all_reduce_mean_grads(params)               # every rank reaches this, even with nothing kept
            scaler.unscale_(opt); gns.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0))); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        model.eval(); n_ = max(1, len(Rk))
        stats = dict(reward=dist.scalar_sum(float(np.sum(Rk))) / (n_ * W), acc=dist.scalar_sum(tags.get("correct", 0)) / (n_ * W),
                     signal=dist.scalar_sum(float(nz.sum())) / (len(order) * W), signal_before_filter=dist.scalar_sum(float(nz_all.sum())) / (len(cand) * W))
        h = dict(step=step, **stats, tags=dict(tags), length=float(cm.sum(1).mean().item()), truncated=float((~has[rows]).float().mean().item()),
                 grad_norm=float(np.mean(gns)) if gns else None, clipped=float(np.mean(clipfrac)) if clipfrac else 0.0,
                 kl=float(np.mean(kls)) if kls else None, sec=time.time() - t0)
        hist.append(h)
        if H: H.write(h)
        if dist.is_main() and (step == 1 or step % R.get("log_every", 5) == 0 or step == R["steps"]):
            kl_s = "" if h["kl"] is None else f" | kl {h['kl']:.4f}"
            log.info(f"[{tag}] step {step}/{R['steps']} reward {h['reward']:.3f} acc {h['acc']:.3f} | signal {h['signal']:.2f} (before filter "
                     f"{h['signal_before_filter']:.2f}) | len {h['length']:.0f} | trunc {h['truncated']:.2f} | grad {h['grad_norm'] or 0:.2f} | "
                     f"clip {h['clipped']:.3f}{kl_s} | {h['sec']:.0f}s | {dict(tags.most_common(4))}")
        del enc, comp, comp_k, ids, att, old, ref
        if step % 5 == 0: torch.cuda.empty_cache()
    if H: H.close()
    model.config.use_cache = True
    if dist.is_main():
        k = max(1, min(10, len(hist) // 4))
        log.info(f"[{tag}] GRPO done in {(time.time()-t_start)/60:.1f} min | rollout acc first {k} {np.mean([h['acc'] for h in hist[:k]]):.3f} -> last {k} {np.mean([h['acc'] for h in hist[-k:]]):.3f}")
    return hist
