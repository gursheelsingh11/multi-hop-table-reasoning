"""Graph -> table path-finding benchmark (ported from the v3 notebook; logic unchanged, ids made deterministic).

Random directed graphs with invented place names; each edge = one table row (Edge | From | To | Cost | Mode).
Tasks: hops (fewest one-way connections), cost (cheapest route), chain (follow a sequence of modes).
A path answer `PATH: A > B > C` is EXECUTED against the table: every hop must be a real row; the answer is computed from the rows.
"""
import re, random, heapq
from dataclasses import dataclass, field, replace
from collections import Counter, defaultdict, deque

CFG = dict(TASKS=["hops", "cost", "chain"], ID_NODES=(6, 9), MID_NODES=(10, 11), OOD_NODES=(12, 15), EDGE_FACTOR=1.7, P_UNREACH=0.15,
           ID_MIN_HOPS=2, MID_MIN_HOPS=3, OOD_MIN_HOPS=3, ID_CHAIN=(2, 3), MID_CHAIN=(3, 3), OOD_CHAIN=(3, 4))

def configure(gcfg):
    """Override the defaults with the `graph:` block of config.yaml (keys in lower case, e.g. id_nodes: [6, 9])."""
    for k, v in (gcfg or {}).items():
        K = k.upper()
        if K in CFG: CFG[K] = tuple(v) if isinstance(v, list) and K != "TASKS" else v

MODES = ["road", "rail", "ferry", "air"]
SYL = ["ka", "lo", "mi", "ra", "ven", "dor", "ti", "sa", "bel", "nor", "qu", "xe", "zan", "pol", "ur", "fi", "gar", "mo", "eth", "ix", "bru", "sel"]

class Graph:
    def __init__(self, nodes, edges):
        self.nodes, self.edges = list(nodes), [dict(e) for e in edges]
        self.by_id = {e["id"]: e for e in self.edges}
        self.out = defaultdict(list)
        for e in self.edges: self.out[e["src"]].append(e)
        self.pair = {(e["src"], e["dst"]): e for e in self.edges}

def make_names(rng, n):
    names = set()
    while len(names) < n:
        names.add((rng.choice(SYL) + rng.choice(SYL) + (rng.choice(SYL) if rng.random() < 0.3 else "")).capitalize())
    return sorted(names, key=lambda _: rng.random())

def random_graph(rng, n):
    nodes = make_names(rng, n); m = max(n, round(CFG["EDGE_FACTOR"] * n)); pairs = set()
    order = nodes[:]; rng.shuffle(order)
    for a, b in zip(order, order[1:]):                      # partial backbone so that long paths exist
        if rng.random() < 0.7: pairs.add((a, b))
    while len(pairs) < m:
        a, b = rng.sample(nodes, 2); pairs.add((a, b))
    pairs = list(pairs); rng.shuffle(pairs)
    return Graph(nodes, [dict(id=f"e{i+1}", src=a, dst=b, cost=rng.randint(1, 9), mode=rng.choice(MODES)) for i, (a, b) in enumerate(pairs)])

def renumber(g): return Graph(g.nodes, [dict(e, id=f"e{i+1}") for i, e in enumerate(g.edges)])
def without_edge(g, eid): return renumber(Graph(g.nodes, [e for e in g.edges if e["id"] != eid]))
_eid = lambda e: int(e["id"][1:])

# ---------------- teachers ----------------
def bfs_path(g, s, t):
    prev, dq = {s: None}, deque([s])
    while dq:
        u = dq.popleft()
        if u == t: break
        for e in sorted(g.out[u], key=_eid):
            if e["dst"] not in prev: prev[e["dst"]] = e; dq.append(e["dst"])
    if t not in prev: return None
    path, v = [], t
    while prev[v] is not None: path.append(prev[v]); v = prev[v]["src"]
    return path[::-1]

def cheapest_path(g, s, t):
    num = {_eid(e): e for e in g.edges}; done, heap = set(), [(0, 0, (), s)]
    while heap:
        c, h, ids, u = heapq.heappop(heap)
        if u in done: continue
        done.add(u)
        if u == t: return [num[i] for i in ids]
        for e in g.out[u]:
            if e["dst"] not in done: heapq.heappush(heap, (c + e["cost"], h + 1, ids + (_eid(e),), e["dst"]))
    return None

def follow_modes(g, s, modes):
    cur = {s}
    for m in modes: cur = {e["dst"] for u in cur for e in g.out[u] if e["mode"] == m}
    return cur

def chain_path(g, s, modes):
    def dfs(u, k):
        if k == len(modes): return []
        for e in sorted(g.out[u], key=_eid):
            if e["mode"] == modes[k]:
                r = dfs(e["dst"], k + 1)
                if r is not None: return [e] + r
        return None
    return dfs(s, 0)

def solve(task, g, s, t=None, modes=None):
    """-> (gold answer string, canonical gold path or None)"""
    if task == "chain":
        ends = follow_modes(g, s, modes)
        if len(ends) != 1: return ("NONE" if not ends else None), None
        return next(iter(ends)), chain_path(g, s, modes)
    p = bfs_path(g, s, t) if task == "hops" else cheapest_path(g, s, t)
    if p is None: return "NONE", None
    return str(len(p) if task == "hops" else sum(e["cost"] for e in p)), p

# ---------------- questions ----------------
@dataclass
class GQ:
    uid: str; task: str; g: Graph; src: str; dst: str; modes: list; gold: str; gold_path: list; split: str
    @property
    def question(self):
        if self.task == "hops":
            return f"What is the fewest number of one-way connections needed to get from {self.src} to {self.dst}? Answer NONE if it is impossible."
        if self.task == "cost":
            return f"What is the lowest total cost to get from {self.src} to {self.dst} using one-way connections? Answer NONE if it is impossible."
        steps = ", then ".join(f"{'an' if m[0] in 'aeiou' else 'a'} {m} connection" for m in self.modes)
        return f"Start at {self.src}. Take {steps}. Where do you end up?"


def make_question(rng, task, split, uid):
    key = {"ood": "OOD", "mid": "MID"}.get(split, "ID")
    nr, min_h, chain_r = CFG[f"{key}_NODES"], CFG[f"{key}_MIN_HOPS"], CFG[f"{key}_CHAIN"]
    for _ in range(500):
        g = random_graph(rng, rng.randint(*nr))
        if task in ("hops", "cost"):
            want_none = rng.random() < CFG["P_UNREACH"]
            pairs = [(a, b) for a in g.nodes for b in g.nodes if a != b]; rng.shuffle(pairs)
            for a, b in pairs:
                hp = bfs_path(g, a, b)
                if (want_none and hp is None) or (not want_none and hp is not None and len(hp) >= min_h):
                    gold, path = solve(task, g, a, b)
                    return GQ(uid, task, g, a, b, [], gold, path, split)
        else:
            k = rng.randint(*chain_r)
            for _ in range(60):
                s = rng.choice(g.nodes); u, walk = s, []
                for _ in range(k):
                    if not g.out[u]: break
                    e = rng.choice(g.out[u]); walk.append(e); u = e["dst"]
                if len(walk) < k: continue
                modes = [e["mode"] for e in walk]; gold, path = solve("chain", g, s, modes=modes)
                if gold not in (None, "NONE") and gold != s:
                    return GQ(uid, "chain", g, s, gold, modes, gold, path, split)
    raise RuntimeError("could not generate a question")

def make_split(seed, n_per_task, split):
    rng = random.Random(seed); out = [make_question(rng, t, split, f"{split}{seed}-{t}-{i}") for t in CFG["TASKS"] for i in range(n_per_task)]
    rng.shuffle(out); return out

# ---------------- the graph -> table converter (and the text baseline) ----------------
def graph_table(g):
    lines = ["C0:Edge | C1:From | C2:To | C3:Cost | C4:Mode"]
    lines += [f"{e['id']} | {e['src']} | {e['dst']} | {e['cost']} | {e['mode']}" for e in g.edges]
    return "Table:\n" + "\n".join(lines)

def graph_text(g):
    return "Connections (one-way):\n" + "\n".join(f"{e['id']}: {e['src']} -> {e['dst']}, cost {e['cost']}, by {e['mode']}." for e in g.edges)

_VIEW = {}
def view(q, rep):
    """The question as seen under a representation. 'gtable' = edge table sorted by From (then To) and renumbered in that order;
    the gold path is mapped onto the new row ids. 'table' / 'text' use the graph as generated."""
    if rep != "gtable": return q
    key = (q.uid, rep)
    if key not in _VIEW:
        g2 = renumber(Graph(q.g.nodes, sorted(q.g.edges, key=lambda e: (e["src"], e["dst"]))))
        p2 = [g2.pair[(e["src"], e["dst"])] for e in q.gold_path] if q.gold_path else q.gold_path
        _VIEW[key] = replace(q, g=g2, gold_path=p2)
    return _VIEW[key]

def make_prompt(q, rep):
    q = view(q, rep)
    return f"{graph_table(q.g) if rep in ('table', 'gtable') else graph_text(q.g)}\nQuestion: {q.question}\nResponse:"

def target(q, out):
    if out == "ans": return " ANS: " + q.gold
    if q.gold_path is None: return " PATH: NONE"
    if out == "nodes": return " PATH: " + " > ".join([q.src] + [e["dst"] for e in q.gold_path])
    if out == "hybrid": return " PATH: " + q.src + "".join(f" -{e['id']}-> {e['dst']}" for e in q.gold_path)
    return " PATH: " + " ".join(e["id"] for e in q.gold_path)


HALLUCINATED = {"unknown_edge", "missing_edge", "broken_chain"}       # a hop that does not exist in the table
PATH_OUTS = ("nodes", "edges", "hybrid")

def _num(s):
    try: return float(str(s).replace(",", ""))
    except ValueError: return None

def is_correct(q, ans):
    if ans is None: return False
    a = str(ans).strip().strip(".").strip()
    if q.gold == "NONE": return a.upper() == "NONE"
    if q.task in ("hops", "cost"):
        x, y = _num(a), _num(q.gold); return x is not None and y is not None and abs(x - y) < 1e-9
    return a.lower() == q.gold.lower()

def execute(q, out, text):
    """Parse one model output and run it on q's table. -> dict(status, ans, path_len)"""
    t = text.strip().split("\n")[0].strip()
    if out == "ans":
        m = re.match(r"^ANS:\s*(.+)$", t)
        return dict(status="direct" if m else "bad_format", ans=m.group(1).strip() if m else None, path_len=None)
    m = re.match(r"^PATH:\s*(.+)$", t)
    if not m: return dict(status="bad_format", ans=None, path_len=None)
    body = m.group(1).strip()
    if body.upper().rstrip(".") == "NONE": return dict(status="none", ans="NONE", path_len=0)
    g = q.g; edges = []
    if out == "edges":
        toks = body.replace(",", " ").split()
        if not toks or not all(re.fullmatch(r"e\d+", x) for x in toks): return dict(status="bad_format", ans=None, path_len=None)
        for x in toks:
            if x not in g.by_id: return dict(status="unknown_edge", ans=None, path_len=len(toks))
            edges.append(g.by_id[x])
    elif out == "hybrid":
        parts = body.split("->")
        if len(parts) < 2: return dict(status="bad_format", ans=None, path_len=None)
        names, ids = [], []
        for p in parts[:-1]:
            mm = re.fullmatch(r"\s*(.*?)\s*-\s*(e\d+)\s*", p)
            if not mm: return dict(status="bad_format", ans=None, path_len=None)
            names.append(mm.group(1)); ids.append(mm.group(2))
        names.append(parts[-1].strip())
        for a, b, x in zip(names, names[1:], ids):
            if x not in g.by_id: return dict(status="unknown_edge", ans=None, path_len=len(ids))
            e = g.by_id[x]
            if (e["src"], e["dst"]) != (a, b):   # the cited row does not support this hop
                return dict(status="wrong_citation" if (a, b) in g.pair else "missing_edge", ans=None, path_len=len(ids))
            edges.append(e)
    else:
        names = [x.strip() for x in body.split(">")]
        if len(names) < 2: return dict(status="bad_format", ans=None, path_len=None)
        for a, b in zip(names, names[1:]):
            if (a, b) not in g.pair: return dict(status="missing_edge", ans=None, path_len=len(names) - 1)
            edges.append(g.pair[(a, b)])
    for e1, e2 in zip(edges, edges[1:]):
        if e1["dst"] != e2["src"]: return dict(status="broken_chain", ans=None, path_len=len(edges))
    if edges[0]["src"] != q.src: return dict(status="wrong_start", ans=None, path_len=len(edges))
    if q.task == "chain":
        if [e["mode"] for e in edges] != list(q.modes): return dict(status="wrong_mode", ans=None, path_len=len(edges))
        return dict(status="executed", ans=edges[-1]["dst"], path_len=len(edges))
    if edges[-1]["dst"] != q.dst: return dict(status="wrong_end", ans=None, path_len=len(edges))
    val = len(edges) if q.task == "hops" else sum(e["cost"] for e in edges)
    return dict(status="executed", ans=str(val), path_len=len(edges))

def reward(q, out, text):
    """Binary verifiable reward for later GRPO: 1 iff the executed (or direct) answer is correct."""
    return float(is_correct(q, execute(q, out, text)["ans"]))


def make_cf(q, rng):
    sens = inv = None
    path_ids = {e["id"] for e in q.gold_path}
    cands = []
    for e in rng.sample(q.gold_path, len(q.gold_path)):
        g2 = without_edge(q.g, e["id"]); gold2, p2 = solve(q.task, g2, q.src, q.dst)
        if gold2 != q.gold: cands.append((gold2 == "NONE", g2, gold2, p2))
    if cands:                                                                  # prefer edits that force a RE-ROUTE over edits that disconnect
        _, g2, gold2, p2 = min(cands, key=lambda c: c[0]); sens = replace(q, uid=q.uid + ":sens", g=g2, gold=gold2, gold_path=p2, split="cf_sens")
    on_path = {q.src} | {e["dst"] for e in q.gold_path}
    off = [e for e in q.g.edges if e["id"] not in path_ids]; rng.shuffle(off)
    off.sort(key=lambda e: not (e["src"] in on_path or e["dst"] in on_path))     # "near" edges first
    for e in off:
        g2 = without_edge(q.g, e["id"]); gold2, p2 = solve(q.task, g2, q.src, q.dst)
        if gold2 == q.gold: inv = replace(q, uid=q.uid + ":inv", g=g2, gold=gold2, gold_path=p2, split="cf_inv"); break
    return sens, inv



def to_json(q):
    """Serialisable form of a question (written into the instance logs)."""
    return dict(uid=q.uid, task=q.task, split=q.split, src=q.src, dst=q.dst, modes=list(q.modes), gold=q.gold,
                gold_path=[e["id"] for e in (q.gold_path or [])], n_nodes=len(q.g.nodes), n_edges=len(q.g.edges))


def build_cf(test_id, n, seed):
    rng = random.Random(seed); out = []
    for q in [x for x in test_id if x.task in ("hops", "cost") and x.gold_path]:
        s, i = make_cf(q, rng)
        if s and i: out.append((q, s, i))
        if len(out) >= n: break
    return out


def deletion_augment(train, frac, seed):
    """Improvement (v3 finding: no model ever answers NONE after an edge deletion): add training copies of hops/cost questions with one
    gold-path edge deleted and the gold recomputed (a re-route or NONE). Teaches the model that the answer depends on the rows present."""
    rng = random.Random(seed); out = []
    pool = [q for q in train if q.task in ("hops", "cost") and q.gold_path]
    for q in rng.sample(pool, min(len(pool), int(frac * len(train)))):
        e = rng.choice(q.gold_path); g2 = without_edge(q.g, e["id"]); gold2, p2 = solve(q.task, g2, q.src, q.dst)
        out.append(replace(q, uid=q.uid + ":aug", g=g2, gold=gold2, gold_path=p2, split="train_aug"))
    return out
