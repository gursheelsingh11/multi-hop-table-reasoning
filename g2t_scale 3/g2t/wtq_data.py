"""WikiTableQuestions: local JSONL produced by download.py -> TableExample objects.

Splits (HF `Stanford/wikitablequestions`): train 11,321 | validation 2,831 | test 4,344 (= pristine-unseen-tables, the official test set).
All questions are kept (multi-answer included); the evaluator compares answer SETS.
"""
import json, os, random
from dataclasses import dataclass, field


@dataclass
class TableExample:
    id: str
    question: str
    headers: list
    rows: list
    answers: list
    task: str = "wtq"
    @property
    def answer(self):                     # single string, used by the teacher search and the SFT 'ANS:' target
        return " | ".join(self.answers)


def load_split(cfg, split):
    path = os.path.join(cfg["paths"]["data"], "wtq", f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -> run `python download.py --config <config>` on a node with internet first")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); n = len(d["headers"])
            rows = [([str(c) for c in r] + [""] * n)[:n] for r in d["rows"]]
            out.append(TableExample(d["id"], d["question"], [str(h) for h in d["headers"]], rows, [str(a) for a in d["answers"]]))
    return out


def subsample(exs, n, seed):
    if n is None or n >= len(exs): return list(exs)
    idx = sorted(random.Random(seed).sample(range(len(exs)), n)); return [exs[i] for i in idx]


def load_wtq(cfg):
    W = cfg["wtq"]; s = cfg["seed"]
    train = subsample(load_split(cfg, "train"), W.get("train_n"), s)
    dev = subsample(load_split(cfg, "validation"), W.get("dev_n"), s + 1)
    test = subsample(load_split(cfg, "test"), W.get("test_n"), s + 2)
    return train, dev, test
