"""Run directory layout and logging.

runs/<run_name>[_smoke]/
  config.resolved.yaml            exact settings used
  logs/run.log                    every stage appends here (also printed to stdout -> SLURM .out file)
  logs/<stage>_<timestamp>.log    one file per stage invocation
  data/                           cached examples, teacher programs, graph splits
  adapters/<name>/                LoRA adapters (SFT / GRPO)
  train/<name>.jsonl              per-step training history (loss, reward, ...)
  instances/<bench>/<system>.jsonl   ONE LINE PER TEST INSTANCE: question, gold, raw output, parsed program/SQL/path,
                                     execution status, prediction, correct (official + lenient), tokens, latency
  cf/<bench>/<system>.jsonl       per-instance counterfactual records
  metrics.json, report.md         final numbers (written by the report stage)
  done/<stage>                    completion markers (stages are skipped when done, unless --force)
"""
import json, logging, os, sys, time


def run_paths(cfg):
    R = cfg["paths"]["run_dir"]
    P = {k: os.path.join(R, k) for k in ("logs", "data", "adapters", "train", "instances", "cf", "done")}
    for v in P.values(): os.makedirs(v, exist_ok=True)
    P["root"] = R
    return P


def get_logger(cfg, stage):
    P = run_paths(cfg)
    log = logging.getLogger("g2t")
    log.handlers.clear(); log.setLevel(logging.INFO); log.propagate = False
    from .dist import world, rank
    tag = stage + (f" | rank {rank()}/{world()}" if world() > 1 else "")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | " + tag + " | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(P["logs"], "run.log")),
              logging.FileHandler(os.path.join(P["logs"], f"{stage}{'_rank%d' % rank() if world() > 1 else ''}_{time.strftime('%Y%m%d-%H%M%S')}.log"))):
        h.setFormatter(fmt); log.addHandler(h)
    return log


class JsonlWriter:
    """Append-only JSONL writer (flushes every line, so partial results survive a crash or a SLURM time limit)."""
    def __init__(self, path, mode="w"):
        os.makedirs(os.path.dirname(path), exist_ok=True); self.f = open(path, mode, encoding="utf-8")
    def write(self, rec): self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); self.f.flush()
    def close(self): self.f.close()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def read_jsonl(path):
    if not os.path.exists(path): return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: pass         # a truncated last line after a crash
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=1, ensure_ascii=False, default=str)


def load_json(path):
    with open(path, encoding="utf-8") as f: return json.load(f)


def is_done(cfg, stage): return os.path.exists(os.path.join(run_paths(cfg)["done"], stage))
def mark_done(cfg, stage, info=None):
    save_json(dict(stage=stage, time=time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})), os.path.join(run_paths(cfg)["done"], stage))
