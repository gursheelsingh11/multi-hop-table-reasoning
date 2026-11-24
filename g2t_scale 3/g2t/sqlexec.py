"""SQLite execution for the Sparks (Text2SQL-RL) baseline.

A WTQ table becomes an in-memory table `t`. Header names are sanitised into SQL identifiers; a column whose non-empty cells are
>= 80% plain numbers ('1,200', '45.5', '$3', '12%') is stored as REAL (parsed with the same robust number reader as our engine),
otherwise TEXT. Graph tasks use a table `edges(edge, src, dst, cost, mode)`.
Only read-only queries are executed (SELECT / WITH, recursive CTEs allowed), with a wall-clock limit.
"""
import re, sqlite3, time
from .engine import parse_num, fmt_num

_BAD = re.compile(r"\b(attach|detach|pragma|insert|update|delete|drop|alter|create|replace|vacuum|load_extension)\b", re.I)


def sanitize_headers(headers):
    names, seen = [], set()
    for j, h in enumerate(headers):
        n = re.sub(r"[^a-z0-9]+", "_", str(h).lower()).strip("_") or f"col{j}"
        if n[0].isdigit(): n = "c_" + n
        base, k = n, 2
        while n in seen: n = f"{base}_{k}"; k += 1
        seen.add(n); names.append(n)
    return names


def column_types(rows, n_cols):
    types = []
    for j in range(n_cols):
        vals = [str(r[j]).strip() for r in rows if str(r[j]).strip() not in ("", "-", "—", "–", "n/a", "N/A")]
        ok = sum(parse_num(v, dates=False) is not None for v in vals)
        types.append("REAL" if vals and ok >= 0.8 * len(vals) else "TEXT")
    return types


def build_db(headers, rows, table="t"):
    names = sanitize_headers(headers); types = column_types(rows, len(headers))
    con = sqlite3.connect(":memory:")
    con.execute(f'CREATE TABLE {table} (' + ", ".join(f'"{n}" {t}' for n, t in zip(names, types)) + ")")
    data = []
    for r in rows:
        data.append([(parse_num(c, dates=False) if t == "REAL" else str(c)) if str(c).strip() != "" else None for c, t in zip(r, types)])
    con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(names))})", data)
    return con, names, types


def schema_text(headers, rows, table="t", show_rows="all", max_rows=None):
    """CREATE TABLE statement + the rows as stored (what the SQL model sees)."""
    names = sanitize_headers(headers); types = column_types(rows, len(headers))
    ddl = f"CREATE TABLE {table} (\n" + ",\n".join(f'  "{n}" {t}  -- {str(h)[:40]}' for n, t, h in zip(names, types, headers)) + "\n);"
    if show_rows == "schema": return ddl
    shown = rows if max_rows is None else rows[:max_rows]
    def cell(c, t):
        if str(c).strip() == "": return "NULL"
        if t == "REAL":
            v = parse_num(c, dates=False); return fmt_num(v) if v is not None else "NULL"
        return str(c).replace("\n", " ")[:60]
    lines = [" | ".join(names)] + [" | ".join(cell(c, t) for c, t in zip(r, types)) for r in shown]
    return ddl + f"\n-- rows of {table}:\n" + "\n".join(lines)


def run_sql(con, sql, timeout_s=3.0, max_values=50):
    """-> (list_of_answer_strings, None) or (None, 'error text'). Never raises."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s: return None, "empty SQL"
    if not re.match(r"^(select|with)\b", s, re.I): return None, "only SELECT/WITH queries are allowed"
    if _BAD.search(s): return None, "forbidden statement"
    deadline = time.time() + timeout_s
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 5000)
    try:
        cur = con.execute(s); res = cur.fetchmany(1000)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        con.set_progress_handler(None, 0)
    vals = []
    for row in res:
        for v in row:
            if v is None: continue
            vals.append(fmt_num(float(v)) if isinstance(v, (int, float)) else str(v).strip())
            if len(vals) >= max_values: break
    if not vals: return None, "empty result"
    return vals, None


def graph_db(edges):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE edges (edge TEXT, src TEXT, dst TEXT, cost REAL, mode TEXT)")
    con.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", [(e["id"], e["src"], e["dst"], float(e["cost"]), e["mode"]) for e in edges])
    return con


def graph_schema_text(edges):
    ddl = ("CREATE TABLE edges (\n  \"edge\" TEXT,  -- edge id\n  \"src\" TEXT,  -- from (one-way)\n  \"dst\" TEXT,  -- to\n"
           "  \"cost\" REAL,\n  \"mode\" TEXT  -- road / rail / ferry / air\n);")
    lines = ["edge | src | dst | cost | mode"] + [f"{e['id']} | {e['src']} | {e['dst']} | {e['cost']} | {e['mode']}" for e in edges]
    return ddl + "\n-- rows of edges:\n" + "\n".join(lines)
