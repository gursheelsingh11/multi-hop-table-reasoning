#!/usr/bin/env python
"""Compare metrics.json of several runs (e.g. an LR sweep):  python tools/compare_runs.py runs/sweep_*"""
import json, sys, os
for d in sys.argv[1:]:
    p = os.path.join(d, "metrics.json")
    if not os.path.exists(p): print(f"{d}: no metrics.json yet"); continue
    M = json.load(open(p))
    for b, B in M["benchmarks"].items():
        print(f"{os.path.basename(d):28s} {b}: " + " | ".join(f"{k} {v['acc']:.4f}" for k, v in sorted(B["systems"].items())))
