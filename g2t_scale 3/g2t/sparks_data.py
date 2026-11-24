"""Stage-1 data of the Sparks baseline: the authors' released synthetic CoT traces (o3-mini generated, LLM-verified; 3,174 examples)
from huggingface.co/datasets/jls205/synthetic_cot_traces_clinton (cot.csv). Column names are detected automatically and logged."""
import csv, os, re, sys
import numpy as np
from .prompts import sparks_messages, extract_sql

CANDS = dict(question=["question", "instruction", "nl_question", "prompt", "query_text", "input_question"],
             schema=["context", "schema", "db_schema", "create_table", "table_schema", "input"],
             reasoning=["cot", "reasoning", "chain_of_thought", "rationale", "thought", "explanation", "response", "output", "completion"],
             sql=["sql", "gold_sql", "query", "answer", "sql_query", "target"])


def _pick(cols, keys, used):
    low = {c.lower().strip(): c for c in cols}
    for k in keys:
        if k in low and low[k] not in used: return low[k]
    for k in keys:
        for c in cols:
            if k in c.lower() and c not in used: return c
    return None


def load_sparks_pairs(cfg, tok, chat, log):
    path = os.path.join(cfg["paths"]["data"], "sparks", "cot.csv")
    if not os.path.exists(path): raise FileNotFoundError(f"{path} missing -> run download.py")
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = list(rows[0].keys()); used, M = set(), {}
    for k in ("sql", "reasoning", "question", "schema"):
        M[k] = _pick(cols, CANDS[k], used)
        if M[k]: used.add(M[k])
    log.info(f"[sparks data] {len(rows)} rows | columns {cols} | detected mapping {M}")
    if not M["question"] or not (M["reasoning"] or M["sql"]):
        raise RuntimeError(f"could not map the columns of cot.csv ({cols}); edit CANDS in g2t/sparks_data.py")
    pairs = []
    for r in rows:
        q = (r.get(M["question"]) or "").strip(); schema = (r.get(M["schema"]) or "").strip() if M["schema"] else ""
        reas = (r.get(M["reasoning"]) or "").strip() if M["reasoning"] else ""
        sql = (r.get(M["sql"]) or "").strip() if M["sql"] else ""
        if not q: continue
        if re.search(r"<sql>.*?</sql>", reas, re.S | re.I): target = reas
        else:
            sql = sql or extract_sql(reas) or ""
            if not sql: continue
            reas_wo = re.sub(r"```.*?```", "", reas, flags=re.S).strip()
            target = (reas_wo + "\n" if reas_wo else "") + f"<sql>{sql.strip()}</sql>"
        pairs.append((chat(tok, sparks_messages(q, schema or "(schema given in the question)")), target))
    log.info(f"[sparks data] {len(pairs)} SFT pairs. example target:\n{pairs[0][1][:600] if pairs else '-'}")
    return pairs


# ====================================================================== stage 2 as in the paper: GRPO on BIRD
# The paper's rewards (no weights given -> equal-weight mean of those we can compute):
#   1. execution reward: paper = LLM judge counting orthographic changes, R = 1/(x+1). Here: run predicted and gold SQL on the BIRD
#      SQLite database; 1 if the result sets are equal, else 0 (the judge's x = 0 case). Used only if the databases are present.
#   2. string matching: difflib ratio (matching characters / total, longest contiguous matching subsequences)   [as in the paper]
#   3. component F1 over SELECT / WHERE / GROUP BY clauses                                                       [as in the paper]
#   4. LLM class judge: dropped (needs o3-mini; not reproducible offline).
import difflib, json as _json, sqlite3, time as _time
from dataclasses import dataclass as _dc


@_dc
class BirdExample:
    id: str
    question: str
    evidence: str
    sql: str
    db_id: str
    schema: str
    db_path: str


def _bird_schema(t):
    tabs = t["table_names_original"]; cols = t["column_names_original"]; types = t.get("column_types", ["TEXT"] * len(cols))
    out = []
    for ti, tn in enumerate(tabs):
        cs = [f'  "{c}" {str(types[k]).upper()}' for k, (tj, c) in enumerate(cols) if tj == ti]
        out.append(f'CREATE TABLE "{tn}" (\n' + ",\n".join(cs) + "\n);")
    return "\n".join(out)


def load_bird(cfg, log):
    root = os.path.join(cfg["paths"]["data"], "bird", "train")
    qp, tp = os.path.join(root, "train.json"), os.path.join(root, "train_tables.json")
    if not (os.path.exists(qp) and os.path.exists(tp)):
        raise FileNotFoundError(f"BIRD not found under {root} (train.json, train_tables.json, optional train_databases/) -> see README section 1")
    tables = {t["db_id"]: t for t in _json.load(open(qp.replace("train.json", "train_tables.json")))}
    out = []
    for i, d in enumerate(_json.load(open(qp))):
        db = d["db_id"]
        if db not in tables: continue
        dbp = os.path.join(root, "train_databases", db, f"{db}.sqlite")
        out.append(BirdExample(f"bird-{i}", d["question"], d.get("evidence", ""), d.get("SQL") or d.get("query", ""), db, _bird_schema(tables[db]), dbp))
    has_db = sum(os.path.exists(e.db_path) for e in out)
    log.info(f"[bird] {len(out)} training questions, {len(tables)} databases, {has_db} questions with a local SQLite file (execution reward)")
    return out


def bird_messages(e):
    q = e.question + (f"\nExternal knowledge: {e.evidence}" if e.evidence else "")
    return sparks_messages(q, e.schema)


_CLAUSE = re.compile(r"\b(select|from|where|group\s+by|having|order\s+by|limit)\b", re.I)

def _clauses(sql):
    s = re.sub(r"\s+", " ", (sql or "").strip().rstrip(";")).lower(); parts = {}; idx = [(m.start(), m.group(1).replace("  ", " ")) for m in _CLAUSE.finditer(s)]
    for k, (st, name) in enumerate(idx):
        end = idx[k + 1][0] if k + 1 < len(idx) else len(s); name = re.sub(r"\s+", " ", name)
        parts.setdefault(name, []).extend(re.findall(r"[a-z0-9_\.\"'`*<>=!]+", s[st + len(name):end]))
    return parts


def component_f1(pred, gold):
    P, G = _clauses(pred), _clauses(gold); scores = []
    for c in ("select", "where", "group by"):
        if c not in P and c not in G: continue
        a, b = set(P.get(c, [])), set(G.get(c, [])); inter = len(a & b)
        scores.append(0.0 if not a or not b else (2 * inter / (len(a) + len(b))))
    return float(np.mean(scores)) if scores else 0.0


def string_match(pred, gold):
    norm = lambda x: re.sub(r"\s+", " ", (x or "").strip().rstrip(";")).lower()
    return difflib.SequenceMatcher(None, norm(pred), norm(gold)).ratio()


def _exec(db_path, sql, timeout_s=5.0):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True); deadline = _time.time() + timeout_s
    con.set_progress_handler(lambda: 1 if _time.time() > deadline else 0, 10000)
    try: return set(map(tuple, con.execute(sql).fetchmany(5000)))
    finally: con.close()


def bird_reward(e, text):
    sql = extract_sql(text)
    if not sql: return 0.0, "no_sql"
    rs = [string_match(sql, e.sql), component_f1(sql, e.sql)]; tag = "partial"
    if os.path.exists(e.db_path):
        try:
            ok = _exec(e.db_path, sql) == _exec(e.db_path, e.sql); rs.append(1.0 if ok else 0.0); tag = "correct" if ok else "exec_wrong"
        except Exception: rs.append(0.0); tag = "sql_error"
    elif rs[0] == 1.0: tag = "correct"
    return float(np.mean(rs)), tag
