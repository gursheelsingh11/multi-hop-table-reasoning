"""Configuration: config.yaml -> nested dict with attribute access. `smoke: true` deep-merges `smoke_overrides` on top."""
import copy, json, os
import yaml


class Cfg(dict):
    """dict with attribute access (cfg.sft.lr)."""
    def __getattr__(self, k):
        try: v = self[k]
        except KeyError: raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v
    def __setattr__(self, k, v): self[k] = v


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict): out[k] = _merge(out[k], v)
        else: out[k] = copy.deepcopy(v)
    return out


def _to_cfg(d):
    return Cfg({k: _to_cfg(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path, smoke=None, overrides=None):
    raw = yaml.safe_load(open(path))
    if smoke is not None: raw["smoke"] = bool(smoke)
    if raw.get("smoke"): raw = _merge(raw, raw.get("smoke_overrides", {}))
    raw.pop("smoke_overrides", None)
    raw = _merge(raw, overrides or {})
    root = os.path.expandvars(os.path.expanduser(raw["paths"]["root"]))
    P = raw["paths"]
    for k in ("models", "data", "runs", "hf_cache"):
        v = P.get(k) or k
        P[k] = v if os.path.isabs(os.path.expandvars(v)) else os.path.join(root, v)
        P[k] = os.path.expandvars(P[k])
    P["root"] = root
    name = raw["run_name"] + ("_smoke" if raw.get("smoke") else "")
    P["run_dir"] = os.path.join(P["runs"], name)
    return _to_cfg(raw)


def model_path(cfg, key):
    """Local folder of a downloaded model (download.py stores repo 'Org/Name' as models/Org__Name), else the hub id."""
    repo = cfg["models"][key]
    local = os.path.join(cfg["paths"]["models"], repo.replace("/", "__"))
    return local if os.path.isdir(local) else repo


def dump(cfg, path):
    with open(path, "w") as f: yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False)
