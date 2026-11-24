"""Report stage: reads instances/ and cf/ JSONL -> metrics.json, report.md, summary_<bench>.csv."""
import csv, glob, math, os, re
import numpy as np
from .logs import read_jsonl, save_json, run_paths

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def paired(a, b, B=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float); idx = np.random.default_rng(seed).integers(0, len(a), (B, len(a))); d = (a[idx] - b[idx]).mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

verdict = lambda lo, hi: "better" if lo > 0 else ("worse" if hi < 0 else "n.s.")
fmt = lambda x: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _load(dirpath, bench):
    """Merges evaluation shards: <system>.jsonl and <system>.shard<i>of<n>.jsonl -> one dict per system."""
    out = {}
    for p in sorted(glob.glob(os.path.join(dirpath, bench, "*.jsonl"))):
        name = re.sub(r"\.shard\d+of\d+$", "", os.path.basename(p)[:-6]); recs = read_jsonl(p)
        for r in recs: out.setdefault(name, {})[r["key"]] = r
    return out


def _sys_metrics(recs, bench):
    R = list(recs.values()); n = len(R); k = sum(r["correct"] for r in R); lo, hi = wilson(k, n)
    m = dict(n=n, acc=k / max(1, n), ci=[lo, hi], tokens=float(np.mean([r.get("n_out", 0) for r in R])) if R else None,
             sec_per_q=float(np.mean([r.get("sec_per_q", 0) for r in R])) if R else None)
    if bench != "graph":
        m["acc_lenient"] = float(np.mean([r.get("correct_lenient", False) for r in R]))
        m["acc_tol"] = float(np.mean([r.get("correct_tol", r["correct"]) for r in R]))
        m["truncated"] = float(np.mean([r.get("truncated", False) for r in R]))
        modes = [r.get("mode") for r in R]
        if "prog" in modes:
            m["prog_rate"] = float(np.mean([x == "prog" for x in modes]))
            m["executed_correct_share"] = sum(r["correct"] and r.get("mode") == "prog" and r.get("status") == "executed" for r in R) / max(1, k)
            m["fallback_rate"] = float(np.mean([r.get("fallback", False) for r in R]))
        if "sql" in modes: m["sql_executes"] = float(np.mean([r.get("status") == "executed" for r in R]))
        if any("format_ok" in r for r in R): m["format_ok"] = float(np.mean([bool(r.get("format_ok")) for r in R]))
    else:
        for sp in ("id", "ood"):
            S = [r for r in R if r.get("split") == sp]; m[f"acc_{sp}"] = float(np.mean([r["correct"] for r in S])) if S else None
        for t in sorted({r.get("task") for r in R}):
            for sp in ("id", "ood"):
                S = [r for r in R if r.get("task") == t and r.get("split") == sp]; m[f"acc_{t}_{sp}"] = float(np.mean([r["correct"] for r in S])) if S else None
        st = [r.get("status") for r in R]
        if any(s in ("executed", "missing_edge", "wrong_end") for s in st):
            m["valid_path"] = float(np.mean([s in ("executed", "none") for s in st]))
            m["hallucinated_hop"] = float(np.mean([s in ("missing_edge", "unknown_edge", "broken_chain") for s in st]))
    return m


def _cf_wtq(recs, only=None):
    """only: restrict to these original question ids (the COMMON set answered correctly by every compared system)."""
    if only is not None: recs = {k: r for k, r in recs.items() if r["orig"] in only}
    by = {}
    for r in recs.values(): by.setdefault(r["orig"], {})[r["key"]] = r
    from .wtq_eval import official_match
    n = fol = stale = nf = kf = nn_ = kn = 0
    for orig, d in by.items():
        o = d.get(orig)
        if not o or not o["correct"]: continue
        for key, r in d.items():
            if r["cf_kind"] == "sens":
                n += 1; fol += official_match(r["pred"], r["expected"]) if r["pred"] else 0; stale += official_match(r["pred"], o["gold"]) if r["pred"] else 0
            elif r["cf_kind"] == "far": nf += 1; kf += official_match(r["pred"], r["expected"]) if r["pred"] else 0
            elif r["cf_kind"] == "near": nn_ += 1; kn += official_match(r["pred"], r["expected"]) if r["pred"] else 0
    n_orig = len(by); ok_orig = sum(1 for o, d in by.items() if d.get(o, {}).get("correct"))
    return dict(acc_orig=ok_orig / max(1, n_orig), n_sens=n, follow=fol / max(1, n), follow_ci=list(wilson(fol, n)), stale=stale / max(1, n),
                far_invariant=kf / max(1, nf), near_invariant=kn / max(1, nn_), n_far=nf, n_near=nn_)


def _cf_graph(recs, only=None):
    if only is not None: recs = {k: r for k, r in recs.items() if r["orig"] in only}
    by = {}
    for r in recs.values(): by.setdefault(r["orig"], {})[r["cf_kind"]] = r
    ok = [d for d in by.values() if d.get("orig", {}).get("correct") and "sens" in d and "inv" in d]; n = len(ok)
    f = lambda L: float(np.mean(L)) if L else float("nan")
    return dict(acc_orig=sum(1 for d in by.values() if d.get("orig", {}).get("correct")) / max(1, len(by)), n=n,
                follow=f([d["sens"]["correct"] for d in ok]), follow_reroute=f([d["sens"]["correct"] for d in ok if not d["sens"].get("disconnect")]),
                follow_none=f([d["sens"]["correct"] for d in ok if d["sens"].get("disconnect")]),
                stale=f([d["sens"]["pred"] == d["orig"]["pred"] for d in ok]), invariant=f([d["inv"]["correct"] for d in ok]))


def _train_summary(P):
    out = {}
    for p in sorted(glob.glob(os.path.join(P["train"], "*.jsonl"))):
        H = read_jsonl(p); name = os.path.basename(p)[:-6]
        if not H: continue
        if "loss" in H[0]: out[name] = dict(kind="sft", steps=len(H), first_loss=H[0]["loss"], last_loss=float(np.mean([h["loss"] for h in H[-10:]])))
        else:
            k = max(1, min(10, len(H) // 4))
            out[name] = dict(kind="grpo", steps=len(H), acc_first=float(np.mean([h["acc"] for h in H[:k]])), acc_last=float(np.mean([h["acc"] for h in H[-k:]])),
                             signal_last=float(np.mean([h["signal"] for h in H[-k:]])), length_last=float(np.mean([h["length"] for h in H[-k:]])))
    return out


def stage_report(cfg, log):
    P = run_paths(cfg); M = dict(smoke=cfg["smoke"], benchmarks={}, cf={}, training=_train_summary(P)); md = []
    md.append(f"# Results — run `{os.path.basename(P['root'])}` (smoke={cfg['smoke']})\n")
    ref = cfg["report"]["reference"]
    for bench in cfg["benchmarks"]:
        S = _load(P["instances"], bench)
        if not S: continue
        full = max(len(v) for v in S.values()); part = {k: len(v) for k, v in S.items() if len(v) < full}
        if part: log.warning(f"[report] {bench}: incomplete systems (missing shards or still running) are excluded from the common set: {part}")
        S = {k: v for k, v in S.items() if len(v) == full}
        keys = set.intersection(*[set(v) for v in S.values()]); log.info(f"[report] {bench}: {len(S)} systems, {len(keys)} common instances")
        metr = {name: _sys_metrics({k: v[k] for k in keys}, bench) for name, v in S.items()}
        rname = ref[bench] if ref[bench] in S else None; comps = {}
        if rname:
            kk = sorted(keys)
            for name in S:
                if name == rname: continue
                d, lo, hi = paired([S[name][k]["correct"] for k in kk], [S[rname][k]["correct"] for k in kk]); comps[name] = dict(diff=d, lo=lo, hi=hi, verdict=verdict(lo, hi))
                if bench == "graph":
                    for sp in ("id", "ood"):
                        ks = [k for k in kk if S[rname][k].get("split") == sp]
                        if ks:
                            d2, l2, h2 = paired([S[name][k]["correct"] for k in ks], [S[rname][k]["correct"] for k in ks]); comps[name][sp] = dict(diff=d2, lo=l2, hi=h2)
        M["benchmarks"][bench] = dict(n_common=len(keys), systems=metr, reference=rname, vs_reference=comps)
        md.append(f"\n## {bench.upper()} — {len(keys)} test instances (common to all systems)\n")
        if bench != "graph":
            md.append("| system | acc (official) [95% CI] | acc (num. tol 0.01) | acc (lenient) | tokens/answer | PROG rate | executed share of correct | SQL executes | format ok | rows truncated | "
                      f"Δ = system − {rname} [95% CI] (better/worse than {rname}) |\n|" + "---|" * 11)
            for name, m in sorted(metr.items(), key=lambda kv: -kv[1]["acc"]):
                c = comps.get(name); dd = f"{c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] {c['verdict']}" if c else "reference"
                md.append(f"| {name} | {m['acc']:.3f} [{m['ci'][0]:.3f}-{m['ci'][1]:.3f}] | {fmt(m.get('acc_tol'))} | {fmt(m.get('acc_lenient'))} | {m['tokens']:.0f} | {fmt(m.get('prog_rate'))} | "
                          f"{fmt(m.get('executed_correct_share'))} | {fmt(m.get('sql_executes'))} | {fmt(m.get('format_ok'))} | {fmt(m.get('truncated'))} | {dd} |")
        else:
            tasks = sorted({k.split('_')[1] for m in metr.values() for k in m if k.startswith("acc_") and k.count("_") == 2})
            md.append("| system | acc all [95% CI] | ID | OOD | " + " | ".join(f"{t} ID/OOD" for t in tasks) + f" | valid path | hallucinated hop | tokens | Δ = system − {rname}, ID / OOD |\n|" + "---|" * (8 + len(tasks)))
            for name, m in sorted(metr.items(), key=lambda kv: -kv[1]["acc"]):
                c = comps.get(name); dd = (f"{c['id']['diff']:+.3f} [{c['id']['lo']:+.3f},{c['id']['hi']:+.3f}] / {c['ood']['diff']:+.3f} [{c['ood']['lo']:+.3f},{c['ood']['hi']:+.3f}]"
                                           if c and "id" in c and "ood" in c else ("reference" if name == rname else "-"))
                per = " | ".join(f"{fmt(m.get(f'acc_{t}_id'))}/{fmt(m.get(f'acc_{t}_ood'))}" for t in tasks)
                md.append(f"| {name} | {m['acc']:.3f} [{m['ci'][0]:.3f}-{m['ci'][1]:.3f}] | {fmt(m.get('acc_id'))} | {fmt(m.get('acc_ood'))} | {per} | {fmt(m.get('valid_path'))} | "
                          f"{fmt(m.get('hallucinated_hop'))} | {m['tokens']:.0f} | {dd} |")
        pub = (cfg["report"].get("published") or {}).get(bench) or []
        if pub:
            md.append(f"\n**Published results ({bench.upper()}), cited from the papers, not re-run here.** Protocols differ (see notes), so compare with care:\n")
            md.append("| system (as reported) | score | metric / protocol | source |\n|---|---|---|---|")
            for p in pub: md.append(f"| {p['system']} | {p['value']} | {p['metric']} | {p['source']} |")
            M["benchmarks"][bench]["published"] = pub
            if rname:
                ours = metr[rname]["acc"] * 100
                for p in pub:
                    if p.get("compare"): md.append(f"\n- head-to-head: **{rname} {ours:.1f}** (this run, {metr[rname]['n']} questions) vs **{p['system']} {p['value']}** (reported): {ours - float(p['value']):+.1f} points")
        with open(os.path.join(P["root"], f"summary_{bench}.csv"), "w", newline="") as f:
            w = csv.writer(f); cols = sorted({k for m in metr.values() for k in m if k != "ci"}); w.writerow(["system"] + cols + ["ci_lo", "ci_hi"])
            for name, m in metr.items(): w.writerow([name] + [m.get(c) for c in cols] + m["ci"])
        CF = _load(P["cf"], bench)
        if CF:
            cfm = {name: (_cf_wtq(r) if bench == "wtq" else _cf_graph(r)) for name, r in CF.items()}; M["cf"][bench] = cfm
            # common set: original questions that EVERY compared system answers correctly -> identical questions for all systems
            cs = [n for n in (cfg["cf"].get("common_systems") or {}).get(bench, []) if n in CF] or list(CF)
            ok_sets = [{r["orig"] for r in CF[n].values() if r.get("cf_kind") == "orig" and r["correct"]} for n in cs]
            common = set.intersection(*ok_sets) if ok_sets else set()
            cfc = {name: (_cf_wtq(r, common) if bench == "wtq" else _cf_graph(r, common)) for name, r in CF.items()}
            M["cf"][bench + "_common"] = dict(n_questions=len(common), systems=cfc)
            md.append(f"\n### {bench.upper()} counterfactual faithfulness (rates conditional on correct on the original table)\n")
            if bench == "wtq":
                md.append("| system | acc orig | follows evidence edit [95% CI] | stale answer | ignores far edit | ignores near edit | n edits |\n|---|---|---|---|---|---|---|")
                for name, m in cfm.items():
                    md.append(f"| {name} | {m['acc_orig']:.3f} | {m['follow']:.3f} [{m['follow_ci'][0]:.2f}-{m['follow_ci'][1]:.2f}] | {m['stale']:.3f} | {m['far_invariant']:.3f} | {m['near_invariant']:.3f} | {m['n_sens']} |")
            else:
                md.append("| system | acc orig | follows deletion | re-route | -> NONE | stale | ignores off-path deletion | n |\n|---|---|---|---|---|---|---|---|")
                for name, m in cfm.items():
                    md.append(f"| {name} | {m['acc_orig']:.3f} | {fmt(m['follow'])} | {fmt(m['follow_reroute'])} | {fmt(m['follow_none'])} | {fmt(m['stale'])} | {fmt(m['invariant'])} | {m['n']} |")
            md.append(f"\n**Same questions for every system** ({len(common)} original questions answered correctly by all of: {', '.join(cs)}; the fairest comparison):\n")
            if bench == "wtq":
                md.append("| system | follows evidence edit [95% CI] | stale answer | ignores far edit | ignores near edit | n edits |\n|---|---|---|---|---|---|")
                for name, m in cfc.items():
                    md.append(f"| {name} | {fmt(m['follow'])} [{m['follow_ci'][0]:.2f}-{m['follow_ci'][1]:.2f}] | {fmt(m['stale'])} | {fmt(m['far_invariant'])} | {fmt(m['near_invariant'])} | {m['n_sens']} |")
            else:
                md.append("| system | follows deletion | re-route | -> NONE | stale | ignores off-path deletion | n |\n|---|---|---|---|---|---|---|")
                for name, m in cfc.items():
                    md.append(f"| {name} | {fmt(m['follow'])} | {fmt(m['follow_reroute'])} | {fmt(m['follow_none'])} | {fmt(m['stale'])} | {fmt(m['invariant'])} | {m['n']} |")
            if len(common) < 30: md.append(f"\n_note: only {len(common)} common questions; widen `cf.wtq_n` / `cf.graph_n` or drop the weakest system from the cf lists for a larger common set._")
    if M["training"]:
        md.append("\n## Training\n\n| run | kind | steps | start | end | groups with signal (end) |\n|---|---|---|---|---|---|")
        for n, t in M["training"].items():
            if t["kind"] == "sft": md.append(f"| {n} | SFT | {t['steps']} | loss {t['first_loss']:.3f} | loss {t['last_loss']:.3f} | - |")
            else: md.append(f"| {n} | GRPO | {t['steps']} | rollout acc {t['acc_first']:.3f} | rollout acc {t['acc_last']:.3f} | {t['signal_last']:.2f} |")
    md.append("\n## Notes\n- `ours_grpo_sc` = execution-guided self-consistency (k samples, majority vote over executed answers); `*_gmask` = goal-conditioned "
              "action-masked decoding; `ours_rft` = SFT + rejection-sampled programs. `sparks_paper` = stage 2 on BIRD with the paper's rewards; `sparks` = stage 2 on WTQ.\n- Primary WTQ metric: official-evaluator logic (set match, number/date typing) with thousands-separator canonicalisation; "
              "'lenient' = the v1–v8 matcher.\n- Table-R1 models are evaluated with their own prompt and decoding (T=0.6, top-p 0.95); their paper additionally re-checks "
              "exact-match failures with a GPT-4.1-mini judge, which is not done here, so numbers are lower than their reported ones.\n"
              "- Sparks is a re-implementation (no checkpoints were released): stage-1 SFT on the authors' released CoT traces, stage-2 GRPO on WTQ with an "
              "execution reward (WTQ has no gold SQL, so their SQL-similarity rewards cannot be used).\n- Graph-benchmark Table-R1 and Sparks results are zero-shot transfer.")
    save_json(M, os.path.join(P["root"], "metrics.json")); open(os.path.join(P["root"], "report.md"), "w").write("\n".join(md) + "\n")
    log.info(f"[report] wrote {os.path.join(P['root'], 'report.md')} and metrics.json")
    for line in md: log.info(line)
