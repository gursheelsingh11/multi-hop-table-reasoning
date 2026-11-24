"""Every system = (which model, which adapter, which prompt/parse/execute 'kind'). Each prediction produces one log record per instance.

WTQ kinds   : wtq_zs (zero-shot direct) | wtq_cot (Table-R1 template; zero-shot CoT and the released Table-R1 models)
              wtq_sql (Sparks: SQL executed in SQLite) | wtq_direct (direct-answer LoRA control) | wtq_hybrid (ours: PROG executed / ANS + fallback)
Graph kinds : graph_zs | graph_cot | graph_sql | graph_direct | graph_path (ours, free decoding) | graph_mask (ours, action-masked decoding)
"""
import random, re, time
import numpy as np
from . import graph_tasks as G
from . import sqlexec
from .engine import run, parse, unparse, ast_cols, evaluate, format_result, parse_num, fmt_num, ExprError, _plain
from .model import chat
from .prompts import (ours_prompt, parse_resp, zs_messages, clean_zs, tr1_messages, tr1_extract, markdown_table, sparks_wtq_messages,
                      sparks_messages, extract_sql, sparks_format_ok, fit_rows)
from .wtq_eval import score, split_answer, official_match

# ------------------------------------------------------------------ system registry
def registry(cfg):
    S = {}
    W = {"zs_direct": ("base", None, "wtq_zs"), "zs_cot": ("base", None, "wtq_cot"),
         "table_r1_zero": ("table_r1_zero", None, "wtq_cot"), "table_r1_sft": ("table_r1_sft", None, "wtq_cot"),
         "sparks_sft": ("base", "sparks_sft", "wtq_sql"), "sparks": ("base", "sparks_grpo", "wtq_sql"),
         "sparks_paper": ("base", "sparks_bird_grpo", "wtq_sql"),                  # stage 2 on BIRD with the paper's rewards
         "ours_rft": ("base", "wtq_ours_rft", "wtq_hybrid"),                       # SFT + rejection-sampled program augmentation
         "ours_grpo_sc": ("base", "wtq_ours_grpo", "wtq_hybrid_sc"),               # + execution-guided self-consistency
         "direct_sft": ("base", "wtq_direct_sft", "wtq_direct"), "direct_grpo": ("base", "wtq_direct_grpo", "wtq_direct"),
         "ours_sft": ("base", "wtq_ours_sft", "wtq_hybrid"), "ours_grpo": ("base", "wtq_ours_grpo", "wtq_hybrid")}
    Gr = {"zs_direct": ("base", None, "graph_zs"), "zs_cot": ("base", None, "graph_cot"),
          "table_r1_zero": ("table_r1_zero", None, "graph_cot"), "table_r1_sft": ("table_r1_sft", None, "graph_cot"),
          "sparks": ("base", "sparks_grpo", "graph_sql"), "sparks_paper": ("base", "sparks_bird_grpo", "graph_sql"),
          "direct_sft": ("base", "graph_direct_sft", "graph_direct"), "direct_grpo": ("base", "graph_direct_grpo", "graph_direct"),
          "ours_sft": ("base", "graph_ours_sft", "graph_path"), "ours_sft_mask": ("base", "graph_ours_sft", "graph_mask"),
          "ours_grpo": ("base", "graph_ours_grpo", "graph_path"), "ours_grpo_mask": ("base", "graph_ours_grpo", "graph_mask"),
          "ours_sft_gmask": ("base", "graph_ours_sft", "graph_gmask"), "ours_grpo_gmask": ("base", "graph_ours_grpo", "graph_gmask")}
    for bench, D in (("wtq", W), ("crtqa", W), ("graph", Gr)):
        if bench not in cfg["benchmarks"]: continue
        for name, (model, adapter, kind) in D.items():
            if name in cfg["systems"][bench]: S[(bench, name)] = dict(bench=bench, name=name, model=model, adapter=adapter, kind=kind)
    return S


def _max_new(cfg, sysd):
    E = cfg["eval"]
    if sysd["kind"].endswith("_cot"):
        m = E["cot_max_new_reasoning"] if sysd["model"] in E["reasoning_models"] else E["cot_max_new"]
        if E["backend"] != "vllm" and m > E["hf_max_new_cap"]: m = E["hf_max_new_cap"]     # HF generate: cap (logged per instance)
        return m
    if sysd["kind"].endswith("_sql"): return E["sql_max_new"]
    if sysd["kind"] in ("wtq_zs", "graph_zs"): return E["zs_max_new"]
    return E["prog_max_new"]


def _sampling(cfg, sysd):
    """Table-R1 inference setup (temperature 0.6, top-p 0.95) for the CoT systems; greedy for everything else."""
    if sysd["kind"].endswith("_cot"): return dict(sample=True, temp=cfg["eval"]["cot_temp"], top_p=cfg["eval"]["cot_top_p"])
    return dict(sample=False)


# ------------------------------------------------------------------ WTQ prompts with row truncation to the token budget
def wtq_prompt(gen, tok, cfg, sysd, e, suffix=""):
    mx = cfg["wtq"]["max_prompt_tokens"]; k_all = len(e.rows); kind = sysd["kind"]
    if kind in ("wtq_hybrid", "wtq_direct", "wtq_hybrid_sc"): build = lambda k: ours_prompt(e, k) + suffix
    elif kind == "wtq_zs": build = lambda k: chat(tok, zs_messages(e, k))
    elif kind == "wtq_cot": build = lambda k: chat(tok, tr1_messages(e.question, markdown_table(e.headers, e.rows[:k])))
    elif kind == "wtq_sql": build = lambda k: chat(tok, sparks_wtq_messages(e, cfg["sparks"]["show_rows"], k))
    else: raise ValueError(kind)
    return fit_rows(build, k_all, gen.n_tok, mx)


def _wtq_rec(sysd, e, raw, n_out, k_shown, **kw):
    pred = kw.pop("pred", None); sc = score(pred, e.answers) if pred else dict(official=False, tol=False, lenient=False)
    return dict(bench=sysd["bench"], system=sysd["name"], id=e.id, question=e.question, gold=e.answers, raw=raw, pred=pred,
                correct=sc["official"], correct_tol=sc["tol"], correct_lenient=sc["lenient"], n_out=n_out, n_rows=len(e.rows), n_rows_shown=k_shown,
                truncated=k_shown < len(e.rows), **kw)


def hybrid_record(sysd, e, raw, n_out, k, forced=None):
    mode, body = parse_resp(raw, forced)
    if mode == "prog":
        ans, err = run(body, e.headers, e.rows)
        return _wtq_rec(sysd, e, raw, n_out, k, mode="prog", program=body, status="executed" if ans is not None else "error", err=err,
                        pred=[ans] if ans is not None else None)
    if mode == "ans": return _wtq_rec(sysd, e, raw, n_out, k, mode="ans", program=None, status="direct", err=None, pred=split_answer(body))
    return _wtq_rec(sysd, e, raw, n_out, k, mode="bad", program=None, status="error", err="no PROG:/ANS: prefix", pred=None)


def sql_answer(headers, rows, sql):
    if not sql: return None, "no SQL found"
    con, _, _ = sqlexec.build_db(headers, rows)
    try: return sqlexec.run_sql(con, sql)
    finally: con.close()


def _vote_key(pred):
    from .wtq_eval import normalize, to_value
    if not pred: return None
    return tuple(sorted(normalize(p) for p in pred))


def predict_wtq_sc(gen, tok, cfg, sysd, exs, log):
    """Execution-guided self-consistency: sample k hybrid outputs, execute every program, majority-vote the answers
    (ties -> the answer backed by more executed programs, then the greedy order). Falls back to greedy ANS if nothing parses."""
    E = cfg["eval"]; t0 = time.time(); built = [wtq_prompt(gen, tok, cfg, sysd, e) for e in exs]
    outs = gen.generate([p for p, _ in built], _max_new(cfg, sysd), sample=True, temp=E["sc_temp"], top_p=1.0, n=E["sc_k"],
                        adapter=sysd["adapter"], log=log, tag=f"wtq/{sysd['name']}")
    recs = []
    for e, (p, k), samples in zip(exs, built, outs):
        votes = {}; cand = []
        for j, (raw, nt) in enumerate(samples):
            r = hybrid_record(sysd, e, raw, nt, k)
            if r["pred"]:
                key = _vote_key(r["pred"]); v = votes.setdefault(key, dict(n=0, prog=0, first=j, pred=r["pred"], program=r.get("program")))
                v["n"] += 1; v["prog"] += r["mode"] == "prog" and r["status"] == "executed"
            cand.append(dict(mode=r["mode"], program=r.get("program"), status=r["status"], pred=r["pred"]))
        n_out = sum(nt for _, nt in samples)
        if votes:
            best = max(votes.values(), key=lambda v: (v["n"], v["prog"], -v["first"]))
            sc = score(best["pred"], e.answers)
            recs.append(dict(bench=sysd["bench"], system=sysd["name"], id=e.id, question=e.question, gold=e.answers, raw=samples[0][0], pred=best["pred"],
                             correct=sc["official"], correct_tol=sc["tol"], correct_lenient=sc["lenient"], n_out=n_out, n_rows=len(e.rows), n_rows_shown=k, truncated=k < len(e.rows),
                             mode="prog" if best["prog"] else "ans", program=best["program"], status="voted", votes=best["n"], k=len(samples),
                             n_distinct=len(votes), samples=cand, fallback=False))
        else:
            recs.append(dict(bench=sysd["bench"], system=sysd["name"], id=e.id, question=e.question, gold=e.answers, raw=samples[0][0], pred=None, correct=False,
                             correct_tol=False, correct_lenient=False, n_out=n_out, n_rows=len(e.rows), n_rows_shown=k, truncated=k < len(e.rows), mode="bad", status="error",
                             votes=0, k=len(samples), n_distinct=0, samples=cand, fallback=False))
    for r in recs: r["sec_per_q"] = (time.time() - t0) / max(1, len(exs))
    return recs


def predict_wtq(gen, tok, cfg, sysd, exs, log, forced=None):
    """forced in {None, 'prog', 'ans'} (hybrid only; used by the counterfactual stage)."""
    kind = sysd["kind"]; t0 = time.time()
    if kind == "wtq_hybrid_sc" and forced is None: return predict_wtq_sc(gen, tok, cfg, sysd, exs, log)
    if kind == "wtq_hybrid_sc": kind = "wtq_hybrid"
    suffix = {"prog": " PROG:", "ans": " ANS:"}.get(forced, "")
    built = [wtq_prompt(gen, tok, cfg, sysd, e, suffix) for e in exs]
    outs = gen.generate([p for p, _ in built], _max_new(cfg, sysd), adapter=sysd["adapter"], log=log, tag=f"wtq/{sysd['name']}", **_sampling(cfg, sysd))
    recs = []
    for e, (p, k), o in zip(exs, built, outs):
        raw, n_out = o[0]
        if forced: raw = suffix.strip() + " " + raw
        if kind == "wtq_hybrid": r = hybrid_record(sysd, e, raw, n_out, k, forced)
        elif kind == "wtq_direct":
            mode, body = parse_resp(raw); body = body if mode != "bad" else clean_zs(raw)
            r = _wtq_rec(sysd, e, raw, n_out, k, mode="ans", status="direct", pred=split_answer(body))
        elif kind == "wtq_zs": r = _wtq_rec(sysd, e, raw, n_out, k, mode="ans", status="direct", pred=split_answer(clean_zs(raw)))
        elif kind == "wtq_cot":
            pred, fmt_ok = tr1_extract(raw); r = _wtq_rec(sysd, e, raw, n_out, k, mode="cot", status="direct", format_ok=fmt_ok, pred=pred)
        elif kind == "wtq_sql":
            sql = extract_sql(raw); vals, err = sql_answer(e.headers, e.rows, sql)
            r = _wtq_rec(sysd, e, raw, n_out, k, mode="sql", sql=sql, status="executed" if vals else "error", err=err,
                         format_ok=sparks_format_ok(raw), pred=vals)
        recs.append(r)
    # v8 fallback for the hybrid: an unparseable / non-executable program -> ask the same model in ANS mode
    if kind == "wtq_hybrid" and forced is None and cfg["eval"]["hybrid_fallback"]:
        fail = [i for i, r in enumerate(recs) if r["status"] == "error"]
        if fail:
            fb_built = [wtq_prompt(gen, tok, cfg, sysd, exs[i], " ANS:") for i in fail]
            fb = gen.generate([p for p, _ in fb_built], 64, adapter=sysd["adapter"], log=log, tag=f"wtq/{sysd['name']}/fallback")
            for i, (p, k), o in zip(fail, fb_built, fb):
                r2 = hybrid_record(sysd, exs[i], "ANS: " + o[0][0], recs[i]["n_out"] + o[0][1], k, "ans")
                r2.update(fallback=True, first_attempt=dict(program=recs[i]["program"], err=recs[i]["err"], raw=recs[i]["raw"])); recs[i] = r2
    for r in recs: r.setdefault("fallback", False); r["sec_per_q"] = (time.time() - t0) / max(1, len(exs)); r["max_new"] = _max_new(cfg, sysd)
    return recs


# ------------------------------------------------------------------ graph
_NONE_RX = re.compile(r"\b(none|impossible|not possible|no (path|route|way)|cannot|unreachable)\b", re.I)

def graph_parse(q, text):
    t = str(text).strip().split("\n")[0].strip()
    t = re.sub(r"^(final answer|answer)\s*:\s*", "", t, flags=re.I).strip().strip("*").strip()
    if _NONE_RX.search(t): return "NONE"
    if q.task in ("hops", "cost"):
        m = re.search(r"-?\d+(\.\d+)?", t); return m.group(0) if m else None
    hits = sorted((t.find(n), n) for n in q.g.nodes if t.find(n) >= 0); return hits[0][1] if hits else t


def graph_zs_messages(q, rep="gtable"):
    v = G.view(q, rep); body = G.graph_table(v.g)
    return [{"role": "user", "content": f"{body}\n\nQuestion: {q.question}\nReply with only the final answer: a number, a place name, or NONE."}]


def graph_tr1_messages(q, rep="gtable"):
    v = G.view(q, rep)
    md = markdown_table(["Edge", "From", "To", "Cost", "Mode"], [[e["id"], e["src"], e["dst"], e["cost"], e["mode"]] for e in v.g.edges])
    return tr1_messages(q.question, md, title="One-way travel connections (each row is a directed edge)")


def _graph_rec(sysd, q, raw, n_out, ans, **kw):
    return dict(bench="graph", system=sysd["name"], **G.to_json(q), raw=raw, pred=ans, correct=bool(G.is_correct(q, ans)), n_out=n_out, **kw)


def masked_decode(gen, tok, exs, rep, adapter, h_max, goal=False):
    """Hop-by-hop decoding over VALID moves only (start: ' NONE' or ' <source>'; then ' > <dst>' for unused out-edges of the current node, or STOP).
    goal=True (improved, 'gmask'): for hops/cost questions STOP is legal only at the destination named in the question, and a walk that hits a
    dead end (no unused out-edge) or the hop limit without reaching it answers NONE. Chain questions are decoded as with goal=False."""
    ids_cache = {}
    def _ids(t):
        if t not in ids_cache: ids_cache[t] = tok(t, add_special_tokens=False).input_ids
        return ids_cache[t]
    EOS = [tok.eos_token_id]; st = []
    for q in exs:
        v = G.view(q, rep); st.append(dict(q=v, ids=tok(G.make_prompt(q, rep), add_special_tokens=False).input_ids + _ids(" PATH:"),
                                         nodes=[], used=set(), done=False, text="PATH:", n=0))
    def choose(pending):
        seqs, lens, owners = [], [], []
        for si, cands in pending:
            if len(cands) < 2: continue
            for ci, (cid, _) in enumerate(cands): seqs.append(st[si]["ids"] + cid); lens.append(len(cid)); owners.append((si, ci))
        sc = gen.score(seqs, lens, adapter=adapter) if seqs else []; best = {}
        for (si, ci), v in zip(owners, sc):
            if si not in best or v > best[si][0]: best[si] = (v, ci)
        return [(si, cands[best[si][1]] if si in best else cands[0]) for si, cands in pending]
    for si, (cid, pay) in choose([(si, [(_ids(" NONE"), "NONE"), (_ids(" " + s["q"].src), "SRC")]) for si, s in enumerate(st)]):
        s = st[si]; s["ids"] = s["ids"] + cid; s["n"] += len(cid)
        if pay == "NONE": s["text"] += " NONE"; s["done"] = True
        else: s["nodes"].append(s["q"].src); s["text"] += " " + s["q"].src
    for hop in range(h_max):
        pending = []
        for si, s in enumerate(st):
            if s["done"]: continue
            cur = s["nodes"][-1]; outs = sorted([e for e in s["q"].g.out[cur] if e["id"] not in s["used"]], key=G._eid)
            cands = [(_ids(f" > {e['dst']}"), e) for e in outs]
            if goal and s["q"].task in ("hops", "cost"):
                if hop > 0 and cur == s["q"].dst: s["done"] = True; continue          # reached the destination: stop
                if not cands: s["text"] = "PATH: NONE"; s["done"] = True; continue     # dead end before the destination
            elif hop > 0 or not cands: cands.append((EOS, None))
            pending.append((si, cands))
        if not pending: break
        for si, (cid, e) in choose(pending):
            s = st[si]; s["n"] += len(cid)
            if e is None: s["done"] = True
            else: s["ids"] = s["ids"] + cid; s["used"].add(e["id"]); s["nodes"].append(e["dst"]); s["text"] += f" > {e['dst']}"
    if goal:
        for s in st:
            if not s["done"] and s["q"].task in ("hops", "cost") and s["nodes"] and s["nodes"][-1] != s["q"].dst: s["text"] = "PATH: NONE"
    return [(s["text"], s["n"]) for s in st]


def predict_graph(gen, tok, cfg, sysd, exs, log):
    kind, rep, t0 = sysd["kind"], cfg["graph"]["rep"], time.time()
    if kind in ("graph_mask", "graph_gmask"):
        outs = [[o] for o in masked_decode(gen, tok, exs, rep, sysd["adapter"], cfg["graph"]["mask_h_max"], goal=kind == "graph_gmask")]
    else:
        if kind in ("graph_path", "graph_direct"): prompts = [G.make_prompt(q, rep) for q in exs]
        elif kind == "graph_zs": prompts = [chat(tok, graph_zs_messages(q, rep)) for q in exs]
        elif kind == "graph_cot": prompts = [chat(tok, graph_tr1_messages(q, rep)) for q in exs]
        elif kind == "graph_sql": prompts = [chat(tok, sparks_messages(q.question, sqlexec.graph_schema_text(G.view(q, rep).g.edges))) for q in exs]
        outs = gen.generate(prompts, _max_new(cfg, sysd), adapter=sysd["adapter"], log=log, tag=f"graph/{sysd['name']}", **_sampling(cfg, sysd))
    recs = []
    for q, o in zip(exs, outs):
        raw, n_out = o[0]; v = G.view(q, rep)
        if kind in ("graph_path", "graph_mask", "graph_gmask"):
            r = G.execute(v, "nodes", raw); recs.append(_graph_rec(sysd, q, raw, n_out, r["ans"], status=r["status"], path_len=r["path_len"]))
        elif kind == "graph_direct":
            r = G.execute(v, "ans", raw); recs.append(_graph_rec(sysd, q, raw, n_out, r["ans"], status=r["status"]))
        elif kind == "graph_zs": recs.append(_graph_rec(sysd, q, raw, n_out, graph_parse(q, raw), status="direct"))
        elif kind == "graph_cot":
            pred, fmt_ok = tr1_extract(raw); recs.append(_graph_rec(sysd, q, raw, n_out, graph_parse(q, pred[0]) if pred else None, status="direct", format_ok=fmt_ok))
        elif kind == "graph_sql":
            sql = extract_sql(raw); con = sqlexec.graph_db(v.g.edges)
            try: vals, err = sqlexec.run_sql(con, sql) if sql else (None, "no SQL found")
            finally: con.close()
            ans = graph_parse(q, vals[0]) if vals else ("NONE" if err == "empty result" else None)
            recs.append(_graph_rec(sysd, q, raw, n_out, ans, sql=sql, status="executed" if (vals or err == "empty result") else "error", err=err))
    for r in recs: r["sec_per_q"] = (time.time() - t0) / max(1, len(exs)); r["max_new"] = _max_new(cfg, sysd)
    return recs


# ------------------------------------------------------------------ GRPO rewards (1 iff the verified answer is correct)
def reward_wtq_hybrid(e, text):
    mode, body = parse_resp(text)
    if mode == "prog":
        ans, _ = run(body, e.headers, e.rows)
        if ans is None: return 0.0, "prog_err"
        return (1.0, "correct") if official_match([ans], e.answers) else (0.0, "prog_wrong")
    if mode == "ans": return (1.0, "correct") if official_match(split_answer(body), e.answers) else (0.0, "ans_wrong")
    return 0.0, "bad_format"

def reward_wtq_direct(e, text):
    mode, body = parse_resp(text)
    if mode == "bad": return 0.0, "bad_format"
    return (1.0, "correct") if official_match(split_answer(body), e.answers) else (0.0, "ans_wrong")

def reward_wtq_sql(e, text):
    sql = extract_sql(text)
    vals, err = sql_answer(e.headers, e.rows, sql)
    if vals is None: return 0.0, "sql_error" if sql else "no_sql"
    return (1.0, "correct") if official_match(vals, e.answers) else (0.0, "sql_wrong")

def reward_graph(out, rep):
    def f(q, t):
        r = G.execute(G.view(q, rep), out, t); ok = G.is_correct(q, r["ans"]); return float(ok), ("correct" if ok else r["status"])
    return f


# ------------------------------------------------------------------ WTQ counterfactual edits (oracle = teacher program; from v8)
FRESH = ["Zorbland", "Quenth", "Vexmoor", "Plimby", "Kradnia"]

def _run_ast(ast, headers, rows):
    try:
        a = format_result(evaluate(ast, headers, rows)); return a if a != "" else None
    except (ExprError, RecursionError, OverflowError, ValueError, ZeroDivisionError):
        return None

def _edit_value(v, rng):
    x = parse_num(v, dates=False)
    if x is not None and _plain(str(v).split()[0] if str(v).split() else v): return fmt_num(x + rng.randint(3, 9))
    return rng.choice(FRESH)

def make_cf_ast(e, expr, rng):
    """sens: change a cell the program reads (expected answer = program re-executed, must differ); far / near: change a cell in an unused
    column (anywhere / in a row the program depends on) -> the answer must not change."""
    from .wtq_eval import wtq_match
    ast = parse(expr); base = _run_ast(ast, e.headers, e.rows)
    if base is None: return dict(sens=None, far=None, near=None)
    used = ast_cols(ast); rows = e.rows; nR, nC = len(rows), len(e.headers); cands, dep_rows = [], set()
    for r in range(nR):
        for c in used:
            nr = [list(x) for x in rows]; nr[r][c] = _edit_value(rows[r][c], rng)
            a = _run_ast(ast, e.headers, nr)
            if a is not None and not wtq_match(a, base): cands.append((nr, a)); dep_rows.add(r)
    sens = rng.choice(cands) if cands else None
    free = [c for c in range(nC) if c not in used]; far = near = None
    if free and nR:
        nr = [list(x) for x in rows]; nr[rng.randrange(nR)][rng.choice(free)] = rng.choice(FRESH)
        a = _run_ast(ast, e.headers, nr); far = nr if (a is not None and wtq_match(a, base)) else None
        if dep_rows:
            nr = [list(x) for x in rows]; nr[rng.choice(sorted(dep_rows))][rng.choice(free)] = rng.choice(FRESH)
            a = _run_ast(ast, e.headers, nr); near = nr if (a is not None and wtq_match(a, base)) else None
    return dict(sens=sens, far=far, near=near)
