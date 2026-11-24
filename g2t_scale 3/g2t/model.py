"""Model loading, LoRA adapters and batched generation (HF transformers; optional vLLM backend for evaluation)."""
import gc, os, time
import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _attn_impl(pref):
    if pref == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception: return "sdpa"
    return pref or "sdpa"


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok


def resolve_dtype(cfg):
    """'auto' -> bfloat16 on GPUs that support it (A100/A40/A6000/L40/H100/RTX30xx+), else float16 (V100, T4, P100)."""
    d = cfg["hardware"]["dtype"]
    if d == "auto": d = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    return d


def load_hf(path, cfg, log=None, trainable=False, quant=None):
    """quant='4bit' -> QLoRA-style NF4 base (bitsandbytes), used for the Sparks baseline as in the paper (Unsloth QLoRA)."""
    from transformers import AutoModelForCausalLM
    from .dist import local_rank
    t0 = time.time(); H = cfg["hardware"]; dt = DTYPES[resolve_dtype(cfg)]; kw = {}
    attn = _attn_impl(H.get("attn_implementation"))
    if attn == "flash_attention_2" and dt != torch.bfloat16 and dt != torch.float16: attn = "sdpa"
    if attn == "flash_attention_2" and torch.cuda.get_device_capability()[0] < 8: attn = "sdpa"      # flash-attn 2 needs Ampere+
    if quant == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
        except Exception as e:
            if log: log.warning(f"bitsandbytes unavailable ({e}); loading {path} un-quantised (LoRA instead of QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dt, attn_implementation=attn, device_map={"": local_rank()}, **kw)
    model.eval()
    if trainable:
        if "quantization_config" in kw:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    if log: log.info(f"loaded {path} in {time.time()-t0:.0f}s | {dt} | {'4-bit ' if 'quantization_config' in kw else ''}| GPU {local_rank()} "
                     f"{torch.cuda.get_device_name(local_rank())} {torch.cuda.memory_allocated()/1e9:.1f} GB | attn {model.config._attn_implementation}")
    return model


def eos_ids(model, tok):
    g = model.generation_config.eos_token_id
    return sorted({tok.eos_token_id} | set(g if isinstance(g, (list, tuple)) else [g]) - {None})


def lora_config(cfg, L=None):
    from peft import LoraConfig
    L = L or cfg["lora"]
    return LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L.get("dropout", 0.0), task_type="CAUSAL_LM", target_modules=LORA_TARGETS)


def add_trainable_adapter(model, cfg, name, init_from=None, lora=None):
    """Attach a new LoRA adapter `name` (optionally initialised from a saved adapter folder) and make ONLY it trainable (fp32).
    lora: optional {r, alpha, dropout} overriding cfg.lora (Sparks uses the Unsloth defaults)."""
    from peft import get_peft_model, PeftModel
    if init_from:
        if hasattr(model, "peft_config"): model.load_adapter(init_from, adapter_name=name, is_trainable=True)
        else: model = PeftModel.from_pretrained(model, init_from, adapter_name=name, is_trainable=True)
    else:
        if hasattr(model, "peft_config"): model.add_adapter(name, lora_config(cfg, lora))
        else: model = get_peft_model(model, lora_config(cfg, lora), adapter_name=name)
    activate_trainable(model, name)
    return model


def activate_trainable(model, name):
    """Make adapter `name` active and the only trainable parameters (also used after temporarily switching to a reference adapter)."""
    model.set_adapter(name)
    for n, p in model.named_parameters():
        mine = "lora_" in n and f".{name}." in n
        p.requires_grad_(mine)
        if mine and p.dtype != torch.float32: p.data = p.data.float()


def load_eval_adapters(model, adapters):
    """adapters: {name: folder}. Loads them all (inference only)."""
    from peft import PeftModel
    for name, path in adapters.items():
        if hasattr(model, "peft_config"): model.load_adapter(path, adapter_name=name, is_trainable=False)
        else: model = PeftModel.from_pretrained(model, path, adapter_name=name, is_trainable=False)
    model.eval(); return model


def free(*objs):
    for o in objs: del o
    gc.collect(); torch.cuda.empty_cache()


# ------------------------------------------------------------------ HF generator
class HFGen:
    """Batched generation with length-sorted batches under a token budget. adapter=None -> base weights (adapters disabled)."""
    def __init__(self, model, tok, cfg):
        self.model, self.tok, self.cfg = model, tok, cfg
        self.budget = cfg["eval"]["token_budget"]; self.max_bs = cfg["eval"]["max_batch"]
    def n_tok(self, s): return len(self.tok(s, add_special_tokens=False).input_ids)
    def _ctx(self, adapter):
        import contextlib
        if adapter is None: return self.model.disable_adapter() if hasattr(self.model, "peft_config") else contextlib.nullcontext()
        self.model.set_adapter(adapter); return contextlib.nullcontext()
    @torch.no_grad()
    def generate(self, prompts, max_new, sample=False, temp=1.0, top_p=1.0, n=1, adapter=None, log=None, tag=""):
        """-> list (per prompt) of list (n) of (text, n_generated_tokens)."""
        self.model.eval(); lens = [self.n_tok(p) for p in prompts]; res = [None] * len(prompts)
        dev = self.model.device
        order = sorted(range(len(prompts)), key=lambda i: lens[i]); batches, cur = [], []
        for i in order:
            if cur and (len(cur) + 1 > max(1, self.max_bs // n) or (len(cur) + 1) * n * (lens[i] + max_new) > self.budget): batches.append(cur); cur = []
            cur.append(i)
        if cur: batches.append(cur)
        t0 = time.time()
        with self._ctx(adapter):
            for bi, idx in enumerate(batches):
                enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id, use_cache=True, repetition_penalty=1.0)
                kw.update(dict(do_sample=True, temperature=temp, top_p=top_p, top_k=0, num_return_sequences=n) if sample
                          else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
                out = self.model.generate(**enc, **kw); comp = out[:, enc["input_ids"].shape[1]:]
                txt = self.tok.batch_decode(comp, skip_special_tokens=True)
                ntoks = (comp != self.tok.pad_token_id).sum(1).tolist()
                for j, i in enumerate(idx): res[i] = [(txt[j * n + s], ntoks[j * n + s]) for s in range(n)]
                if log and (bi % 20 == 0 or bi == len(batches) - 1):
                    log.info(f"  [{tag}] generated batch {bi+1}/{len(batches)} ({sum(len(b) for b in batches[:bi+1])}/{len(prompts)} prompts, {time.time()-t0:.0f}s)")
                del enc, out, comp
        return res

    @torch.no_grad()
    def score(self, seqs, cand_lens, adapter=None, bs=48):
        """Sum log p(last cand_lens[i] tokens | prefix) for token-id sequences (used by action-masked decoding)."""
        out = [0.0] * len(seqs); order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        with self._ctx(adapter):
            for k in range(0, len(order), bs):
                idx = order[k:k + bs]; Lm = max(len(seqs[i]) for i in idx); C = max(cand_lens[i] for i in idx); K = C + 1
                ids = torch.full((len(idx), Lm), self.tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(idx), Lm), dtype=torch.long)
                for j, i in enumerate(idx): ids[j, Lm - len(seqs[i]):] = torch.tensor(seqs[i]); att[j, Lm - len(seqs[i]):] = 1
                ids, att = ids.to(self.model.device), att.to(self.model.device); pos = (att.cumsum(-1) - 1).clamp(min=0)
                lg = self.model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K, use_cache=False).logits[:, -K:, :]
                lp = torch.log_softmax(lg[:, :-1, :].float(), -1).gather(-1, ids[:, -C:].unsqueeze(-1)).squeeze(-1)
                for j, i in enumerate(idx): out[i] = float(lp[j, C - cand_lens[i]:].sum())
                del ids, att, pos, lg, lp
        return out


# ------------------------------------------------------------------ optional vLLM generator (evaluation only)
class VLLMGen:
    """Same interface as HFGen, backed by vLLM (much faster for long chain-of-thought outputs). LoRA adapters via LoRARequest."""
    def __init__(self, path, tok, cfg, adapters=None):
        from vllm import LLM
        V = cfg["eval"]["vllm"]; self.tok, self.cfg = tok, cfg; self.adapters = adapters or {}
        self.llm = LLM(model=path, tokenizer=path, dtype=resolve_dtype(cfg), enable_lora=bool(self.adapters),
                       max_lora_rank=max(cfg["lora"]["r"], cfg["sparks"]["lora"]["r"]), max_loras=max(1, len(self.adapters)), gpu_memory_utilization=V["gpu_memory_utilization"],
                       max_model_len=V["max_model_len"], enable_prefix_caching=True, seed=cfg["seed"])
        self._ids = {n: i + 1 for i, n in enumerate(self.adapters)}
    def n_tok(self, s): return len(self.tok(s, add_special_tokens=False).input_ids)
    def _lora(self, adapter):
        if adapter is None: return None
        from vllm.lora.request import LoRARequest
        return LoRARequest(adapter, self._ids[adapter], self.adapters[adapter])
    def generate(self, prompts, max_new, sample=False, temp=1.0, top_p=1.0, n=1, adapter=None, log=None, tag=""):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, max_tokens=max_new, temperature=temp if sample else 0.0, top_p=top_p if sample else 1.0, seed=self.cfg["seed"])
        t0 = time.time(); outs = self.llm.generate(prompts, sp, lora_request=self._lora(adapter), use_tqdm=False)
        if log: log.info(f"  [{tag}] vLLM generated {len(prompts)} prompts x {n} in {time.time()-t0:.0f}s")
        return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in outs]
    def score(self, seqs, cand_lens, adapter=None, bs=None):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        outs = self.llm.generate([{"prompt_token_ids": s} for s in seqs], sp, lora_request=self._lora(adapter), use_tqdm=False)
        res = []
        for s, c, o in zip(seqs, cand_lens, outs):
            pl = o.prompt_logprobs; tot = 0.0
            for pos in range(len(s) - c, len(s)):
                d = pl[pos]; tot += d[s[pos]].logprob if (d and s[pos] in d) else -1e4
            res.append(tot)
        return res
    def close(self):
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception: pass
        del self.llm; gc.collect(); torch.cuda.empty_cache()


def chat(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
