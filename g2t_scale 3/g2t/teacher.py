"""Compositional teacher search (from the v8 notebook, unchanged except imports): finds an expression of depth <= D whose
result equals the gold answer. Used ONLY to build training targets and oracle counterfactual edits; never at inference."""
import re, itertools, hashlib
from collections import Counter
from .engine import *
from .engine import _norm, _plain, _cond_holds, _LEAD
from .wtq_eval import wtq_match

_STOP = set("the a an of in on at to for and or is was were are be by with what which who whom whose when where how many much did does do has have had that this from as it its their there than".split())
_stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
_toks = lambda s: {_stem(w) for w in _norm(s).split() if w not in _STOP}

CUE = {  # weak lexical cues, used ONLY to rank candidate programs (never at inference)
    "COUNT": r"\bhow many\b|\bnumber of\b|\bcount\b", "SUM": r"\btotal\b|\bsum\b|\bcombined\b|\baltogether\b|\bin all\b", "AVG": r"\baverage\b|\bmean\b",
    "MAX": r"\bhighest\b|\blargest\b|\bmost\b|\bgreatest\b|\bmaximum\b|\blongest\b|\btallest\b|\bbiggest\b|\bbest\b|\blatest\b",
    "MIN": r"\blowest\b|\bsmallest\b|\bleast\b|\bminimum\b|\bfewest\b|\bshortest\b|\bworst\b|\bearliest\b",
    "FIRST": r"\bfirst\b|\btop\b", "LAST": r"\blast\b|\bfinal\b|\blatest\b",
    "MODE": r"\bmost (common|frequent|often)\b|\bmost\b.*\b(times|appearances|wins|titles|entries|releases|games)\b|\bhow often\b",
    "SUB": r"\bdifference\b|\bhow many more\b|\bhow much more\b|\bmore than\b|\bless than\b|\bfewer\b|\bolder\b|\byounger\b|\blonger\b|\bshorter\b|\bapart\b|\bgap\b|\bmargin\b|\bhigher than\b|\blower than\b",
    "ADD": r"\btotal\b|\bcombined\b|\bsum\b|\baltogether\b|\btogether\b|\bboth\b", "DIV": r"\bratio\b|\bpercent|\bper\b|\btimes as\b", "MULT": r"\btimes\b|\bproduct\b",
}
_cue = lambda key, q: bool(re.search(CUE.get(key, r"$^"), str(q).lower()))
_CMPQ = re.compile(r"\b(more than|less than|fewer than|greater than|larger than|smaller than|higher than|lower than|older than|younger than|longer than|shorter than|at least|at most|over|under|above|below|exceed|exceeds|exceeding|after|before|since|earlier than|later than|between)\b")
_NEG = re.compile(r"\b(not|other than|except|besides|without|excluding|never)\b")

def ekey(ex):
    h = hashlib.md5(repr((list(ex.headers), [list(r) for r in ex.rows])).encode()).hexdigest()[:12]
    return f"{ex.question}||{ex.answer}||{h}"

def lit_clean(t):
    """Literal text as the model sees it: quotes/backslashes -> space, whitespace collapsed, cut at the prompt truncation length."""
    return re.sub(r"\s+", " ", re.sub(r'[\\"]', " ", str(t))).strip()[:TRUNC]

def cell_view(t):
    """Cell text exactly as shown in the prompt."""
    return re.sub(r"\s+", " ", str(t).replace("\\n", " ").replace("\n", " ")).strip()[:TRUNC]

def question_features(question, headers, rows):
    qn = " " + _norm(question) + " "; qt = _toks(question); mentions, seen = [], set()
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            t = _norm(v)
            if len(t) < 2 or (j, t) in seen: continue
            if f" {t} " in qn: seen.add((j, t)); mentions.append((i, j, len(t)))
    mentions.sort(key=lambda m: -m[2])
    return [(i, j) for i, j, _ in mentions[:6]], [len(_toks(h) & qt) for h in headers]

def _bad_op(op, q):
    ql = re.sub(r"\bat (least|most)\b", " ", q.lower())
    if re.match(r"\s*how many\b", ql) and op in ("MIN", "MAX", "AVG", "ARGMAX", "ARGMIN"):
        return not re.search(CUE["MAX"] + "|" + CUE["MIN"], ql)
    return False

def _num_str(v, text):                      # mirrors format_result for a Num
    return text.strip() if (text is not None and not _plain(text)) else fmt_num(v)

def search_exprs(question, headers, rows, gold, max_depth=3, topk=12):
    nR, nC = len(rows), len(headers)
    if nR == 0 or nC == 0: return None
    texts = [[rows[i][j] for i in range(nR)] for j in range(nC)]
    nums = [[parse_num(t) for t in col] for col in texts]
    norms = [[_norm(t) for t in col] for col in texts]
    mentions, hov = question_features(question, headers, rows)
    full = (1 << nR) - 1; negation = bool(_NEG.search(question.lower())); gnum = parse_num(gold, dates=False)

    # ---- condition pool ----
    conds, seen = [], set()
    def add(j, op, lit, ground):
        node = ("cond", j, op, lit); key = (j, op, lit[1] if lit[0] == "num" else _norm(lit[1]))
        if key in seen: return
        mask = 0
        for r in range(nR):
            if _cond_holds(rows[r][j], op, lit): mask |= 1 << r
        if mask == 0: return
        seen.add(key); conds.append(dict(j=j, text=unparse(node), mask=mask, ground=ground))
    for i, j in mentions:
        lit = lit_clean(texts[j][i])
        if _norm(lit): add(j, "==", ("str", lit), 2.0)
    qnums = []
    for x in re.findall(r"\d[\d,]*\.?\d*", question):
        v = parse_num(x, dates=False)
        if v is not None and v not in qnums: qnums.append(v)
    numcol = [j for j in range(nC) if sum(x is not None for x in nums[j]) >= 0.6 * nR]
    numcol.sort(key=lambda j: -hov[j]); tcount = 0
    if _CMPQ.search(question.lower()):                       # numeric thresholds only when the question compares ('over', 'at least', 'between', ...)
        for v in qnums[:3]:
            for j in numcol:
                for op in (">", "<", ">=", "<="):
                    if tcount < 24: add(j, op, ("num", v), 1.25); tcount += 1
    if negation:
        for i, j in mentions[:4]:
            lit = lit_clean(texts[j][i])
            if _norm(lit): add(j, "!=", ("str", lit), 1.0)
    qtoks = [w for w in _norm(question).split() if len(w) >= 4 and w not in _STOP][:6]
    n_before = len(conds)
    for w in qtoks:
        for j in range(nC):
            if any(w in norms[j][r] and norms[j][r] != w for r in range(nR)) and len(conds) - n_before < 6: add(j, "~", ("str", w), 1.0)

    # ---- row sets: all rows | single condition | pairs of conditions (multi-condition FILTER) ----
    cground = lambda cs: sum(c["ground"] for c in cs)
    rowsets = {full: dict(conds=[], text="", ground=0.0)}
    for c in conds:
        if c["mask"] not in rowsets: rowsets[c["mask"]] = dict(conds=[c], text=c["text"], ground=c["ground"])
    top = conds[:16]; npair = 0
    for a, b in itertools.combinations(top, 2):
        m = a["mask"] & b["mask"]
        if m and m != a["mask"] and m != b["mask"] and m not in rowsets and npair < 150:
            rowsets[m] = dict(conds=[a, b], text=a["text"] + ", " + b["text"], ground=cground([a, b])); npair += 1
    RS = []
    for m, d in rowsets.items():
        rids = [r for r in range(nR) if m >> r & 1]
        RS.append(dict(mask=m, rids=rids, text=d["text"], ground=d["ground"], nc=len(d["conds"])))
    vec = lambda rs, t: f"C{t}" if rs["nc"] == 0 else f"FILTER(C{t}, {rs['text']})"

    cands, pool = [], []        # cands: results to compare with gold ; pool: numeric scalars for arithmetic
    def emit(op, rs, cols, expr, depth, res_str, val, extra=0.0):
        if _bad_op(op, question): return
        g = rs["ground"] + sum(min(hov[c], 2) for c in cols) + extra
        if op == "FIRST" and rs["nc"] > 0: has_cue, cue_w = True, 0.5          # a lookup of an entity's value: grounded by its conditions
        else: has_cue = _cue(op, question); cue_w = 1.0
        score = g + (cue_w if has_cue else 0.0) - 0.9 * max(0, rs["nc"] - 1) - 0.5 * max(0, depth - 2)
        if op in ("FIRST", "LAST") and len(rs["rids"]) > 1 and not _cue(op, question): score -= 1.0     # picking one of several rows needs a reason
        ok_ground = g >= 1.0 and (rs["nc"] > 0 or has_cue)
        rec = dict(expr=expr, depth=depth, score=score, g=g, ok=ok_ground, res=res_str, val=val, size=expr.count("(") + expr.count("C"), op=op, rs=rs["text"], nc=rs["nc"])
        if wtq_match(res_str, gold) and ok_ground and depth <= max_depth: cands.append(rec)
        if val is not None and ok_ground and depth <= max_depth - 1: pool.append(rec)

    for rs in RS:
        rids = rs["rids"]; dfilt = 0 if rs["nc"] == 0 else 1
        if not rids: continue
        tb = max(range(nC), key=lambda t: (hov[t], -t))
        emit("COUNT", rs, [tb], f"COUNT({vec(rs, tb)})", dfilt + 1, str(len(rids)), float(len(rids)))
        for t in range(nC):
            vals = [(r, nums[t][r], texts[t][r]) for r in rids if nums[t][r] is not None]
            e = vec(rs, t); d1 = dfilt + 1
            if vals:
                sm = sum(v for _, v, _ in vals)
                emit("SUM", rs, [t], f"SUM({e})", d1, fmt_num(sm), sm); emit("AVG", rs, [t], f"AVG({e})", d1, fmt_num(sm / len(vals)), sm / len(vals))
                mx = max(vals, key=lambda z: z[1]); mn = min(vals, key=lambda z: z[1])
                emit("MAX", rs, [t], f"MAX({e})", d1, _num_str(mx[1], mx[2]), mx[1]); emit("MIN", rs, [t], f"MIN({e})", d1, _num_str(mn[1], mn[2]), mn[1])
            f_, l_ = texts[t][rids[0]].strip(), texts[t][rids[-1]].strip()
            if f_: emit("FIRST", rs, [t], f"FIRST({e})", d1, f_, parse_num(f_))
            if l_ and (len(rids) > 1): emit("LAST", rs, [t], f"LAST({e})", d1, l_, parse_num(l_))
            if len(rids) >= 2:
                cnt = Counter(norms[t][r] for r in rids); top_n, top_c = max(cnt.items(), key=lambda kv: (kv[1], -[norms[t][r] for r in rids].index(kv[0])))
                if top_c >= 2: emit("MODE", rs, [t], f"MODE({e})", d1, texts[t][[r for r in rids if norms[t][r] == top_n][0]].strip(), None)
        for b in numcol:
            ob = [(r, nums[b][r]) for r in rids if nums[b][r] is not None]
            if not ob: continue
            for op, fn in (("ARGMAX", max), ("ARGMIN", min)):
                rid = fn(ob, key=lambda z: z[1])[0]
                for t in range(nC):
                    if t != b and texts[t][rid].strip(): emit(op, rs, [b, t], f"{op}({vec(rs, b)}, {vec(rs, t)})", dfilt + 1, texts[t][rid].strip(), None)

    # ---- arithmetic: meet-in-the-middle over numeric scalars (a OP b = gold  =>  look b up) ----
    if gnum is not None and max_depth >= 2 and pool:
        byval = {}
        for p in sorted(pool, key=lambda p: (p["depth"], p["size"], -p["score"])):
            lst = byval.setdefault(round(p["val"], 6), [])
            if len(lst) < 2: lst.append(p)
        def solve(op, a, need):
            for b in byval.get(round(need, 6), []):
                if b["expr"] == a["expr"]: continue
                if a["nc"] > 0 and b["nc"] > 0 and (a["rs"] == b["rs"] or a["op"] != b["op"]): continue      # two filtered operands: different selections, same aggregator
                depth = 1 + max(a["depth"], b["depth"])
                if depth > max_depth: continue
                cue_ok = _cue(op, question)
                g = a["g"] + b["g"]; score = (a["score"] + b["score"]) / 2 + (1.0 if cue_ok else 0.0) - 0.5 * max(0, depth - 2) - 0.5
                if cue_ok and not _bad_op(op, question):
                    cands.append(dict(expr=f"{op}({a['expr']}, {b['expr']})", depth=depth, score=score, g=g, ok=True, res=fmt_num(gnum), val=None, size=a["size"] + b["size"] + 1))
        for vals_ in list(byval.values()):
            for a in vals_[:1]:
                solve("SUB", a, a["val"] - gnum)
                solve("ADD", a, gnum - a["val"])
                if abs(a["val"]) > 1e-9: solve("MULT", a, gnum / a["val"])
                if abs(gnum) > 1e-9: solve("DIV", a, a["val"] / gnum)

    # ---- verify the best candidates with the REAL engine ----
    cands.sort(key=lambda c: (-c["score"], c["size"], c["depth"])); done = set()
    for c in cands[:topk * 3]:
        if c["expr"] in done: continue
        done.add(c["expr"]); ans, err = run(c["expr"], headers, rows)
        if ans is not None and wtq_match(ans, gold):
            ex_ = _prune(c["expr"], headers, rows, gold)
            return dict(expr=ex_, depth=ast_depth(parse(ex_)), score=c["score"], size=c["size"], n_cands=len(cands))
        if len(done) >= topk: break
    return None

def _prune(expr, headers, rows, gold):
    """Drop non-entity conditions (thresholds, contains, !=) that do not change the result: keep programs minimal."""
    ast = parse(expr)
    def conds_of(node, acc):
        if node[0] == "call":
            if node[1] == "FILTER": acc.append(node)
            for a in node[2]: conds_of(a, acc)
    nodes = []; conds_of(ast, nodes)
    for nd in nodes:
        for c in list(nd[2][1:]):
            if c[2] != "==" and len(nd[2]) > 2:
                trial = list(nd[2]); trial.remove(c); old = nd[2][:]; nd[2][:] = trial
                ans, _ = run(unparse(ast), headers, rows)
                if not (ans is not None and wtq_match(ans, gold)): nd[2][:] = old
    return unparse(ast)

def mine_exprs(examples, max_depth=3):
    pool = {}
    for ex in examples:
        r = search_exprs(ex.question, ex.headers, ex.rows, ex.answer, max_depth)
        if r: pool[ekey(ex)] = dict(question=ex.question, **r)
    return pool

