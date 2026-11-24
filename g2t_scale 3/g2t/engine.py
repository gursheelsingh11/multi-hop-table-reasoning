"""
(From the v8 notebook, unchanged.) Recursive expression engine for table QA (drop-in replacement for the fixed single-step operations in engine/ops.py).

A program is a nested functional expression over table COLUMNS (C0, C1, ...), e.g.
    SUB(MAX(C2), MIN(C2))
    SUM(FILTER(C1, C0=="Lakers"))
    ARGMAX(FILTER(C2, C3>=2000, C3<=2006), FILTER(C0, C3>=2000, C3<=2006))
Values flowing through the tree are  Vec (a column, remembering each cell's row id),  Num (a number, optionally remembering the
original cell text so unit-bearing answers like '4.5 km' survive MAX/MIN),  or  Str.

Primitives
  selection        FILTER(vec, cond, cond, ...)         AND of conditions  Cj <op> literal ;  op in == != > < >= <= ~(contains)
  aggregation      MAX MIN SUM AVG COUNT(vec)            FIRST LAST MODE(vec)
  selection by key ARGMAX(by_vec, report_vec)  ARGMIN(...)   report-value in the row where by_vec is extreme (row ids align)
  arithmetic       ADD SUB MULT DIV(x, y)  ABS(x)        operands may be Num, a 1-element Vec, or a cell string (parsed robustly)
All numeric reading goes through parse_num(): units, %, currency, thousands separators, ranks (1st), times (2:15.3), dates.
Errors never crash: run() returns (answer, None) or (None, 'ErrorType: message').
"""
import re, math, datetime

TRUNC = 32          # prompts show at most this many characters per cell; string literals of this length match by PREFIX


class ExprError(Exception): pass
class ParseError(ExprError): pass
class EvalError(ExprError): pass


# ------------------------------------------------------------------ robust cell parsing
_LEAD = re.compile(r"^(?:[$€£¥#~≈<>]\s?)?([-+\u2212]?\d[\d,]*(?:\.\d+)?|[-+\u2212]?\.\d+)(.*)$")
_TIME = re.compile(r"^(\d+):(\d{1,2})(?::(\d{1,2}))?(?:\.(\d+))?$")
_MONTHS = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())}
_MONTHS.update({m[:3]: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())})
_MON = r"(?P<mon>[A-Za-z]{3,9})\.?"

def parse_date(s):
    """'March 5, 2006' | '5 March 2006' | '2006-03-05' | 'March 2006' -> datetime.date, else None."""
    s = str(s).strip().replace("\u2013", "-")
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    try:
        if m: return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.fullmatch(_MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)
        if m and m.group("mon").lower() in _MONTHS: return datetime.date(int(m.group(3)), _MONTHS[m.group("mon").lower()], int(m.group(2)))
        m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r",?\s+(\d{4})", s)
        if m and m.group("mon").lower() in _MONTHS: return datetime.date(int(m.group(3)), _MONTHS[m.group("mon").lower()], int(m.group(1)))
        m = re.fullmatch(_MON + r",?\s+(\d{4})", s)
        if m and m.group("mon").lower() in _MONTHS: return datetime.date(int(m.group(2)), _MONTHS[m.group("mon").lower()], 1)
    except ValueError:
        return None
    return None

def parse_num(v, dates=True):
    """Number inside a cell, or None. '1,200 kg'->1200  '$5'->5  '12.3%'->12.3  '4.5 km/h'->4.5  '1st'->1  '2:15.3'->135.3 (seconds)
    'November 12, 2005'->date ordinal.  Rejects text that merely CONTAINS more digits ('10-5', 'Model 03')."""
    s = str(v).strip()
    m = _TIME.match(s)
    if m:
        a, b, c, frac = m.groups()
        return float((int(a) * 60 + int(b)) if c is None else (int(a) * 3600 + int(b) * 60 + int(c))) + (float("0." + frac) if frac else 0.0)
    m = _LEAD.match(s)
    if m and not re.search(r"\d", m.group(2)):
        try: return float(m.group(1).replace(",", "").replace("\u2212", "-"))
        except ValueError: return None
    if dates:
        d = parse_date(s)
        if d is not None: return float(d.toordinal())
    return None

def fmt_num(x):
    if abs(x - round(x)) < 1e-9: return str(int(round(x)))
    return ("%.6f" % x).rstrip("0").rstrip(".")

def _plain(s):
    try: float(str(s).replace(",", "")); return True
    except ValueError: return False

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()

def str_equal(cell, lit):
    """Normalised text equality; a literal as long as the prompt truncation also matches by prefix."""
    a, b = _norm(cell), _norm(lit)
    if not b: return False
    if a == b: return True
    na, nb = parse_num(cell, dates=False), parse_num(lit, dates=False)
    if na is not None and nb is not None and _plain(cell) and _plain(lit): return abs(na - nb) < 1e-9
    return len(str(lit)) >= TRUNC and a.startswith(b)


# ------------------------------------------------------------------ values
class Vec:
    def __init__(self, rows): self.rows = rows            # list of (row_id, text)
class Num:
    def __init__(self, v, text=None): self.v, self.text = float(v), text
class Str:
    def __init__(self, s): self.s = s


# ------------------------------------------------------------------ parser
_TOK = re.compile(r'\s*(?:(?P<STR>"(?:[^"\\]|\\.)*")|(?P<NUM>-?\d+(?:\.\d+)?)|(?P<CMP>==|!=|>=|<=|>|<|~)|(?P<ID>[A-Za-z_][A-Za-z0-9_]*)|(?P<LP>\()|(?P<RP>\))|(?P<COMMA>,))')
ARITY = {"FILTER": (2, 99), "MAX": (1, 1), "MIN": (1, 1), "SUM": (1, 1), "AVG": (1, 1), "COUNT": (1, 1), "FIRST": (1, 1), "LAST": (1, 1), "MODE": (1, 1),
         "ARGMAX": (2, 2), "ARGMIN": (2, 2), "ADD": (2, 2), "SUB": (2, 2), "MULT": (2, 2), "DIV": (2, 2), "ABS": (1, 1)}
_unq = lambda s: re.sub(r"\\(.)", r"\1", s[1:-1])

def tokenize(s):
    s = str(s).strip(); pos, out = 0, []
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m or m.end() == pos: raise ParseError(f"unexpected text at position {pos}: {s[pos:pos+12]!r}")
        pos = m.end(); out.append((m.lastgroup, m.group(m.lastgroup)))
    if not out: raise ParseError("empty program")
    return out

class _Parser:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else (None, None)
    def parse(self):
        node = self.expr()
        if self.i != len(self.t): raise ParseError("unexpected trailing tokens")
        return node
    def expr(self):
        k, v = self.peek()
        if k == "NUM": self.i += 1; return ("num", float(v))
        if k == "STR": self.i += 1; return ("str", _unq(v))
        if k != "ID": raise ParseError(f"unexpected token {v!r}")
        self.i += 1
        if self.peek()[0] == "LP":
            name = v.upper(); self.i += 1; args = []
            if name not in ARITY: raise ParseError(f"unknown function {v}")
            if self.peek()[0] == "RP": self.i += 1
            else:
                while True:
                    args.append(self.arg()); k2, _ = self.peek()
                    if k2 == "COMMA": self.i += 1; continue
                    if k2 == "RP": self.i += 1; break
                    raise ParseError("expected ',' or ')'")
            lo, hi = ARITY[name]
            if not lo <= len(args) <= hi: raise ParseError(f"{name} takes {lo}..{hi} arguments, got {len(args)}")
            conds = [a[0] == "cond" for a in args]
            if name == "FILTER":
                if conds[0] or not all(conds[1:]): raise ParseError("FILTER(vec, cond, ...) needs a column first, then conditions")
            elif any(conds): raise ParseError("conditions are only allowed inside FILTER")
            return ("call", name, args)
        m = re.fullmatch(r"C(\d+)", v)
        if not m: raise ParseError(f"unknown identifier {v!r}")
        return ("col", int(m.group(1)))
    def arg(self):
        k, v = self.peek()
        if k == "ID" and re.fullmatch(r"C\d+", v) and self.i + 1 < len(self.t) and self.t[self.i + 1][0] == "CMP":
            j = int(v[1:]); op = self.t[self.i + 1][1]; self.i += 2; lk, lv = self.peek()
            if lk == "NUM": self.i += 1; return ("cond", j, op, ("num", float(lv)))
            if lk == "STR": self.i += 1; return ("cond", j, op, ("str", _unq(lv)))
            raise ParseError("a condition needs a number or a quoted string")
        return self.expr()

def parse(expr): return _Parser(tokenize(expr)).parse()

def _lit_text(lit): return json_escape(lit[1]) if lit[0] == "str" else fmt_num(lit[1])
def json_escape(s): return '"' + str(s).replace("\\", "").replace('"', "") + '"'

def unparse(node):
    """Canonical text of an AST (what the model is trained to emit)."""
    k = node[0]
    if k == "col": return f"C{node[1]}"
    if k == "num": return fmt_num(node[1])
    if k == "str": return json_escape(node[1])
    if k == "cond": return f"C{node[1]}{node[2]}{_lit_text(node[3])}"
    return f"{node[1]}(" + ", ".join(unparse(a) for a in node[2]) + ")"

def ast_depth(node):
    if node[0] != "call": return 0
    return 1 + max([ast_depth(a) for a in node[2]] or [0])
def ast_size(node):
    return 1 + (sum(ast_size(a) for a in node[2]) if node[0] == "call" else 0)
def ast_cols(node):
    k = node[0]
    if k == "col" or k == "cond": return {node[1]}
    if k == "call": return set().union(*[ast_cols(a) for a in node[2]]) if node[2] else set()
    return set()


# ------------------------------------------------------------------ evaluator
def _cond_holds(cell, op, lit):
    if lit[0] == "str":
        if op == "==": return str_equal(cell, lit[1])
        if op == "!=": return not str_equal(cell, lit[1])
        if op == "~":  return _norm(lit[1]) in _norm(cell)
        x, y = parse_num(cell), parse_num(lit[1])
        if x is None or y is None: return False
    else:
        y = lit[1]; x = parse_num(cell, dates=False)
        if x is None and 1000 <= y <= 2999:                          # compare a date cell with a year literal by its year
            d = parse_date(cell); x = float(d.year) if d else None
        if op == "~": return fmt_num(y) in str(cell)
        if x is None: return op == "!="
    return {"==": abs(x - y) < 1e-9, "!=": abs(x - y) >= 1e-9, ">": x > y, "<": x < y, ">=": x >= y, "<=": x <= y}[op]

def _vec(x, what):
    if isinstance(x, Vec): return x
    raise EvalError(f"{what} expects a column/filtered column")

def _num(x):
    if isinstance(x, Num): return x.v
    if isinstance(x, Vec) and len(x.rows) == 1: x = Str(x.rows[0][1])
    if isinstance(x, Str):
        v = parse_num(x.s)
        if v is None: raise EvalError(f"not a number: {x.s!r}")
        return v
    raise EvalError("expected a number")

def _numeric_rows(vec):
    out = [(rid, parse_num(t), t) for rid, t in vec.rows]
    out = [o for o in out if o[1] is not None]
    if not out: raise EvalError("no numeric values")
    return out

def evaluate(node, headers, rows):
    k = node[0]; nR, nC = len(rows), len(headers)
    if k == "col":
        if not 0 <= node[1] < nC: raise EvalError(f"column C{node[1]} does not exist (table has {nC})")
        return Vec([(i, rows[i][node[1]]) for i in range(nR)])
    if k == "num": return Num(node[1])
    if k == "str": return Str(node[1])
    if k == "cond": raise EvalError("condition outside FILTER")
    name, args = node[1], node[2]
    if name == "FILTER":
        vec = _vec(evaluate(args[0], headers, rows), "FILTER"); keep = vec.rows
        for c in args[1:]:
            _, j, op, lit = c
            if not 0 <= j < nC: raise EvalError(f"column C{j} does not exist (table has {nC})")
            keep = [(rid, t) for rid, t in keep if _cond_holds(rows[rid][j], op, lit)]
        return Vec(keep)
    vals = [evaluate(a, headers, rows) for a in args]
    if name in ("MAX", "MIN"):
        o = _numeric_rows(_vec(vals[0], name)); best = (max if name == "MAX" else min)(o, key=lambda z: z[1])
        return Num(best[1], best[2])
    if name in ("SUM", "AVG"):
        o = _numeric_rows(_vec(vals[0], name)); s = sum(z[1] for z in o); return Num(s if name == "SUM" else s / len(o))
    if name == "COUNT": return Num(len(_vec(vals[0], name).rows))
    if name in ("FIRST", "LAST"):
        v = _vec(vals[0], name)
        if not v.rows: raise EvalError("empty selection")
        return Str((v.rows[0] if name == "FIRST" else v.rows[-1])[1])
    if name == "MODE":
        v = _vec(vals[0], name)
        if not v.rows: raise EvalError("empty selection")
        cnt, first = {}, {}
        for _, t in v.rows:
            n_ = _norm(t); cnt[n_] = cnt.get(n_, 0) + 1; first.setdefault(n_, t)
        return Str(first[max(cnt, key=lambda z: (cnt[z], -list(cnt).index(z)))])
    if name in ("ARGMAX", "ARGMIN"):
        by, rep = _vec(vals[0], name), _vec(vals[1], name); o = _numeric_rows(by)
        rid = (max if name == "ARGMAX" else min)(o, key=lambda z: z[1])[0]
        for r_, t in rep.rows:
            if r_ == rid: return Str(t)
        raise EvalError("the selected row is not in the report column")
    if name == "ABS": return Num(abs(_num(vals[0])))
    a, b = _num(vals[0]), _num(vals[1])
    if name == "ADD": return Num(a + b)
    if name == "SUB": return Num(a - b)
    if name == "MULT": return Num(a * b)
    if name == "DIV":
        if abs(b) < 1e-12: raise EvalError("division by zero")
        return Num(a / b)
    raise EvalError(f"unsupported function {name}")

def format_result(v):
    if isinstance(v, Num):
        if v.text is not None and not _plain(v.text): return v.text.strip()              # keep units / dates of a source cell
        return fmt_num(v.v)
    if isinstance(v, Str): return v.s.strip()
    if isinstance(v, Vec):
        if len(v.rows) == 1: return v.rows[0][1].strip()
        raise EvalError("the program returns a list, not a single value")
    raise EvalError("no result")

def run(expr, headers, rows):
    """-> (answer_string, None) on success, (None, 'ErrorType: message') otherwise. Never raises."""
    try:
        ans = format_result(evaluate(parse(expr), headers, rows))
        return (ans, None) if ans != "" else (None, "EvalError: empty answer")
    except ExprError as e:
        return None, f"{type(e).__name__}: {e}"
    except (RecursionError, OverflowError, ValueError, ZeroDivisionError) as e:
        return None, f"EvalError: {type(e).__name__}"


