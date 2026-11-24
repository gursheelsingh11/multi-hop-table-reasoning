#!/usr/bin/env python
"""Run ONCE on a node with internet (e.g. the Prajna login node):  python download.py --config config.yaml
Downloads: Qwen2.5-7B-Instruct, Table-R1-Zero-7B, Table-R1-SFT-7B -> paths.models/Org__Name
           WikiTableQuestions train/validation/test                -> paths.data/wtq/<split>.jsonl
           Sparks synthetic CoT traces (cot.csv)                    -> paths.data/sparks/cot.csv
Then compute jobs run fully offline. Set HF_TOKEN in the environment if you hit rate limits."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2t.config import load_config


def dl_models(cfg, keys):
    from huggingface_hub import snapshot_download
    for k in keys:
        repo = cfg["models"][k]; out = os.path.join(cfg["paths"]["models"], repo.replace("/", "__"))
        if os.path.isdir(out) and any(f.endswith(".safetensors") for f in os.listdir(out)): print(f"[models] {repo}: present"); continue
        print(f"[models] downloading {repo} -> {out}")
        snapshot_download(repo_id=repo, local_dir=out, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "tokenizer*", "*.jinja"],
                          token=os.environ.get("HF_TOKEN"))


def dl_wtq(cfg):
    from datasets import load_dataset
    out_dir = os.path.join(cfg["paths"]["data"], "wtq"); os.makedirs(out_dir, exist_ok=True)
    attempts = [dict(path="Stanford/wikitablequestions", revision="refs/convert/parquet"),
                dict(path="Stanford/wikitablequestions", name="random-split-1", revision="refs/convert/parquet"),
                dict(path="Stanford/wikitablequestions", trust_remote_code=True),
                dict(path="wikitablequestions", trust_remote_code=True)]
    for split in ("train", "validation", "test"):
        path = os.path.join(out_dir, f"{split}.jsonl")
        if os.path.exists(path): print(f"[wtq] {split}: present"); continue
        ds, errs = None, []
        for kw in attempts:
            try: ds = load_dataset(split=split, **kw); break
            except Exception as e: errs.append(f"{kw}: {type(e).__name__}: {str(e)[:150]}")
        if ds is None: raise RuntimeError("could not load WTQ:\n" + "\n".join(errs))
        with open(path, "w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps(dict(id=ex["id"], question=ex["question"], headers=ex["table"]["header"], rows=ex["table"]["rows"],
                                        answers=ex["answers"], table_name=ex["table"].get("name", "")), ensure_ascii=False) + "\n")
        print(f"[wtq] {split}: {len(ds)} questions -> {path}")


def dl_sparks(cfg):
    from huggingface_hub import hf_hub_download
    out = os.path.join(cfg["paths"]["data"], "sparks"); os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "cot.csv")): print("[sparks] cot.csv present"); return
    p = hf_hub_download("jls205/synthetic_cot_traces_clinton", "cot.csv", repo_type="dataset", local_dir=out, token=os.environ.get("HF_TOKEN"))
    import csv; csv.field_size_limit(sys.maxsize); r = list(csv.DictReader(open(p, encoding="utf-8")))
    print(f"[sparks] {len(r)} rows, columns {list(r[0].keys())} -> {p}")


def dl_crtqa(cfg):
    """CRT-QA (Zhang et al., EMNLP 2023) from the Hugging Face mirror zzh12138/CRT-QA (JSON questions + table CSVs, < 1K questions)."""
    from huggingface_hub import snapshot_download
    out = os.path.join(cfg["paths"]["data"], "crtqa", "raw")
    if os.path.isdir(out) and any(f.endswith(".json") for _, _, fs in os.walk(out) for f in fs): print("[crtqa] present"); return
    snapshot_download(repo_id="zzh12138/CRT-QA", repo_type="dataset", local_dir=out, token=os.environ.get("HF_TOKEN"))
    n_csv = sum(f.endswith(".csv") for _, _, fs in os.walk(out) for f in fs); n_js = [f for _, _, fs in os.walk(out) for f in fs if f.endswith(".json")]
    print(f"[crtqa] {n_csv} CSV tables, JSON files {n_js} -> {out}")


def verify(cfg):
    ok = True
    for k in cfg["models"]:
        d = os.path.join(cfg["paths"]["models"], cfg["models"][k].replace("/", "__")); good = os.path.isdir(d) and any(f.endswith(".safetensors") for f in os.listdir(d))
        print(f"  model {k:14s} {'OK ' if good else 'MISSING'} {d}"); ok &= good
    for s in ("train", "validation", "test"):
        p = os.path.join(cfg["paths"]["data"], "wtq", f"{s}.jsonl"); n = sum(1 for _ in open(p)) if os.path.exists(p) else 0
        print(f"  wtq {s:10s} {n:6d} questions"); ok &= n > 0
    p = os.path.join(cfg["paths"]["data"], "sparks", "cot.csv"); print(f"  sparks cot.csv {'OK' if os.path.exists(p) else 'MISSING'}"); ok &= os.path.exists(p)
    p = os.path.join(cfg["paths"]["data"], "crtqa", "raw"); good = os.path.isdir(p); print(f"  crtqa          {'OK' if good else 'MISSING'}"); ok &= good
    print("ALL PRESENT" if ok else "SOMETHING IS MISSING"); return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", choices=["models", "wtq", "sparks", "crtqa", "verify"], default=None); a = ap.parse_args()
    cfg = load_config(a.config, smoke=False)
    os.makedirs(cfg["paths"]["models"], exist_ok=True); os.makedirs(cfg["paths"]["data"], exist_ok=True)
    if a.only in (None, "wtq"): dl_wtq(cfg)
    if a.only in (None, "sparks"): dl_sparks(cfg)
    if a.only in (None, "crtqa"): dl_crtqa(cfg)
    if a.only in (None, "models"): dl_models(cfg, list(cfg["models"]))
    verify(cfg)
