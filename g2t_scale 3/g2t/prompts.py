"""Prompt builders and output parsers for every system.

ours / direct control : v8 format  'Table:\\nC0:h | C1:h ...\\n<rows>\\nQuestion: q\\nResponse:'  ->  ' PROG: <expr>' or ' ANS: <answer>'
zero-shot direct      : chat prompt, 'reply with only the final answer'
Table-R1 (CoT)        : the paper's system prompt + TQA template (Appendix C), markdown table, <think>/<answer> with {"answer": [...]}
Sparks (Text2SQL)     : chat prompt with the SQLite schema (+ rows), reasoning in natural language, final query in <sql></sql>
"""
import json, re
from .teacher import cell_view
from .wtq_eval import split_answer
from . import sqlexec


# ------------------------------------------------------------------ table rendering (row truncation to a token budget)
def table_text(headers, rows):
    out = [" | ".join(f"C{j}:{cell_view(h)}" for j, h in enumerate(headers))]
    out += [" | ".join(cell_view(c) for c in r) for r in rows]
    return "\n".join(out)


def markdown_table(headers, rows, maxlen=60):
    esc = lambda s: str(s).replace("|", "/").replace("\n", " ")[:maxlen]
    return "\n".join(["| " + " | ".join(esc(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def fit_rows(build, n_rows, n_tok, max_tok):
    """Largest k <= n_rows such that n_tok(build(k)) <= max_tok (binary search). Returns (prompt, k)."""
    p = build(n_rows)
    if n_tok(p) <= max_tok: return p, n_rows
    lo, hi = 0, n_rows
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tok(build(mid)) <= max_tok: lo = mid
        else: hi = mid - 1
    return build(lo), lo


# ------------------------------------------------------------------ ours (hybrid) and the direct control
def ours_prompt(e, k=None):
    rows = e.rows if k is None else e.rows[:k]
    return f"Table:\n{table_text(e.headers, rows)}\nQuestion: {e.question}\nResponse:"


def ours_target(e, teacher):
    from .teacher import ekey
    r = teacher.get(ekey(e)) if len(e.answers) == 1 else None
    if r: return " PROG: " + r["expr"], "PROG"
    a = e.answer.strip()
    return (" ANS: " + a, "ANS") if 0 < len(a) <= 80 else (None, None)


def direct_target(e):
    a = e.answer.strip()
    return (" ANS: " + a, "ANS") if 0 < len(a) <= 80 else (None, None)


def parse_resp(text, forced=None):
    t = text.strip()
    if forced == "prog": return "prog", re.sub(r"^PROG:\s*", "", t).split("\n")[0].strip()
    if forced == "ans": return "ans", re.sub(r"^ANS:\s*", "", t).split("\n")[0].strip()
    if t.startswith("PROG:"): return "prog", t[5:].strip().split("\n")[0].strip()
    if t.startswith("ANS:"): return "ans", t[4:].strip().split("\n")[0].strip()
    return "bad", t


# ------------------------------------------------------------------ zero-shot direct (chat)
ZS_SYSTEM = "You answer questions about a table. Reply with ONLY the final answer, no explanation. If there are several answers, separate them with ' | '."

def zs_messages(e, k=None):
    rows = e.rows if k is None else e.rows[:k]
    return [{"role": "system", "content": ZS_SYSTEM}, {"role": "user", "content": f"{table_text(e.headers, rows)}\n\nQuestion: {e.question}\nAnswer:"}]


def clean_zs(o):
    t = str(o).strip().split("\n")[0].strip()
    return re.sub(r"^(the )?(final )?answer( is)?\s*:?\s*", "", t, flags=re.I).strip().rstrip(".").strip()


# ------------------------------------------------------------------ Table-R1 (Yang et al., EMNLP 2025, Appendix C)
TR1_SYSTEM = ("A conversation between User and Assistant. The user asks a question, and the assistant solves it. "
              "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
              "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
              "i.e., <think> reasoning process here </think> <answer> answer here </answer>.")
TR1_FORMAT = ('The final answer should be concise and use the following format:\n```json\n{\n    "answer": [\n        "answer1",\n'
              '        "answer2",\n        ...\n    ]\n}\n```')

def tr1_messages(question, table_md, title=""):
    user = (f"This is a short-answer table QA task. Answer the question based on the provided table.\nTable Title: {title}\n"
            f"Table Content:\n{table_md}\nQuestion: {question}\n\n{TR1_FORMAT}")
    return [{"role": "system", "content": TR1_SYSTEM}, {"role": "user", "content": user}]


def tr1_extract(text):
    """-> list of answer strings (or None), and whether the output was format-conformant."""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S); body = m[-1] if m else text[-600:]
    try:
        obj = json.loads(re.search(r"\{.*\}", body, re.S).group(0)); a = obj.get("answer")
        if isinstance(a, list): return [str(x).strip() for x in a if str(x).strip()], bool(m)
        if a is not None: return [str(a).strip()], bool(m)
    except Exception: pass
    mm = re.search(r'"answer"\s*:\s*\[(.*?)\]', body, re.S)
    if mm: return [x.strip().strip('"').strip("'") for x in mm.group(1).split(",") if x.strip().strip('"')], False
    last = body.strip().split("\n")[-1].strip() if body.strip() else ""
    return (split_answer(last) if last else None), False


# ------------------------------------------------------------------ Sparks (Stoisser et al., TRL @ ACL 2025): Text2SQL
SPARKS_SYSTEM = "You are an expert data analyst who answers questions about tables by writing SQLite queries."
SPARKS_INSTR = ("Given the database below, write a SQLite query that answers the question. First explain your reasoning step by step "
                "in natural language, then give the final query inside <sql></sql> tags.")

def sparks_messages(question, schema):
    return [{"role": "system", "content": SPARKS_SYSTEM},
            {"role": "user", "content": f"{SPARKS_INSTR}\n\nDatabase:\n{schema}\n\nQuestion: {question}"}]


def sparks_wtq_messages(e, show_rows="all", k=None):
    return sparks_messages(e.question, sqlexec.schema_text(e.headers, e.rows, show_rows=show_rows, max_rows=k))


def extract_sql(text):
    m = re.findall(r"<sql>(.*?)</sql>", text, re.S | re.I)
    if m: return m[-1].strip()
    m = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.S | re.I)
    if m: return m[-1].strip()
    m = re.search(r"((?:WITH|SELECT)\b.*?)(;|$)", text, re.S | re.I)
    return m.group(1).strip() if m else None


def sparks_format_ok(text): return bool(re.search(r"<sql>.*?</sql>", text, re.S | re.I))
