"""CRT-QA (Zhang et al., EMNLP 2023): complex multi-step table QA over Wikipedia tables, < 1K questions.
Used for the head-to-head with Sparks, which reports zero-shot exact match on CRT-QA (models are not trained on it).

download.py mirrors the Hugging Face dataset `zzh12138/CRT-QA` to data/crtqa/raw/. The release stores the questions in a JSON file
keyed by table CSV file; the exact file names are not documented, so this loader discovers the JSON / CSV files and logs what it found.
Each question becomes a TableExample (headers = first CSV row), so every WTQ system runs on it unchanged.
"""
import csv, glob, json, os, sys
from .wtq_data import TableExample, subsample

Q_KEYS = ["Question name", "question", "Question", "question_name"]
A_KEYS = ["Answer", "answer", "answers"]
T_KEYS = ["table", "csv", "table_file", "file", "Table"]


def _get(d, keys):
    for k in keys:
        if k in d: return d[k]
    return None


def _read_csv(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8", errors="replace", newline="") as f: rows = list(csv.reader(f))
    if not rows: return None, None
    n = len(rows[0]); return [str(h) for h in rows[0]], [([str(c) for c in r] + [""] * n)[:n] for r in rows[1:] if any(str(c).strip() for c in r)]


def _datapoints(obj):
    """yield (csv_name or None, datapoint dict) from the several layouts the JSON may have."""
    if isinstance(obj, list):
        for d in obj:
            if isinstance(d, dict): yield _get(d, T_KEYS), d
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and _get(v, Q_KEYS) is not None: yield k, v
            elif isinstance(v, dict):
                for d in v.values():
                    if isinstance(d, dict): yield k, d
            elif isinstance(v, list):
                for d in v:
                    if isinstance(d, dict): yield k, d


def load_crtqa(cfg, log=None):
    raw = os.path.join(cfg["paths"]["data"], "crtqa", "raw")
    if not os.path.isdir(raw): raise FileNotFoundError(f"{raw} not found -> python download.py --only crtqa")
    csvs = {}
    for p in glob.glob(os.path.join(raw, "**", "*.csv"), recursive=True):
        csvs.setdefault(os.path.basename(p), p); csvs.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    jsons = [p for p in glob.glob(os.path.join(raw, "**", "*.json"), recursive=True) if "config" not in os.path.basename(p).lower()]
    out, miss, i = [], 0, 0
    for jp in sorted(jsons):
        try: obj = json.load(open(jp, encoding="utf-8"))
        except Exception: continue
        for tname, d in _datapoints(obj):
            q, a = _get(d, Q_KEYS), _get(d, A_KEYS)
            if q is None or a is None: continue
            tp = csvs.get(str(tname)) or csvs.get(os.path.basename(str(tname))) or csvs.get(os.path.splitext(os.path.basename(str(tname)))[0])
            if not tp: miss += 1; continue
            headers, rows = _read_csv(tp)
            if not headers: miss += 1; continue
            answers = [str(x) for x in a] if isinstance(a, list) else [str(a)]
            out.append(TableExample(f"crt-{i}", str(q), headers, rows, answers, task="crtqa")); i += 1
    if log: log.info(f"[crtqa] {len(out)} questions from {len(jsons)} JSON file(s) and {len(set(csvs.values()))} CSV tables ({miss} questions without a matching table)")
    if not out: raise RuntimeError(f"no CRT-QA questions parsed under {raw}: check the files and adapt Q_KEYS / A_KEYS / T_KEYS in g2t/crtqa_data.py")
    return subsample(out, cfg["crtqa"].get("n"), cfg["seed"] + 5)
