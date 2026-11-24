"""WTQ answer evaluation.

* `official_match(pred_list, gold_list)`: a re-implementation of the logic of the official WikiTableQuestions evaluator
  (Pasupat & Liang 2015, evaluator.py): normalise strings, parse numbers/dates, then compare the answers as SETS
  (same size, every gold value matched). This is the PRIMARY metric for every system.
  The official data ships CoreNLP-canonicalised targets (targetCanon), which the HF copy lacks. To approximate them,
  numbers written with thousands separators ('1,200', '36,271.5') are canonicalised to plain numbers on BOTH sides
  before typing, so '1,200' (gold) matches '1200' (prediction), as it does with the official canonical targets.
* `lenient_match`: the v8 notebook's `wtq_match` (numbers compared after removing thousands separators), applied per element.
  Reported as a SECONDARY metric for continuity with v1-v8.
"""
import re, json, math, unicodedata


# ---------------------------------------------------------------- official-style evaluator
def normalize(x):
    x = str(x)
    x = "".join(c for c in unicodedata.normalize("NFKD", x) if unicodedata.category(c) != "Mn")
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", '"', x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old = x
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())      # citations / footnote marks
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())                         # trailing details in parentheses
        x = re.sub(r'^"([^"]*)"$', r"\1", x.strip())                                 # outer quotes
        if x == old: break
    if x and x[-1] == ".": x = x[:-1]
    return re.sub(r"\s+", " ", x).lower().strip()


class _Value:
    def match(self, other): raise NotImplementedError

class StringValue(_Value):
    def __init__(self, content): self.normalized = normalize(content)
    def match(self, other): return self.normalized == other.normalized

TOL = [1e-6]                                      # official evaluator tolerance; official_match(..., tol=0.01) mimics Table-R1's judge rule

class NumberValue(_Value):
    def __init__(self, amount, original):
        self.amount = float(amount); self.normalized = normalize(original)
    def match(self, other):
        if isinstance(other, NumberValue): return abs(self.amount - other.amount) < TOL[0]
        return self.normalized == other.normalized
    @staticmethod
    def parse(text):
        try: return int(text)
        except (ValueError, TypeError):
            try:
                v = float(text)
                return None if (math.isnan(v) or math.isinf(v)) else v
            except (ValueError, TypeError): return None

class DateValue(_Value):
    def __init__(self, y, m, d, original): self.ymd = (y, m, d); self.normalized = normalize(original)
    def match(self, other):
        if isinstance(other, DateValue): return self.ymd == other.ymd
        return self.normalized == other.normalized
    @staticmethod
    def parse(text):
        try:
            y, m, d = text.lower().split("-")
            y = -1 if y in ("xx", "xxxx") else int(y); m = -1 if m == "xx" else int(m); d = -1 if d == "xx" else int(d)
            if y == m == d == -1: return None
            if m != -1 and not 1 <= m <= 12: return None
            if d != -1 and not 1 <= d <= 31: return None
            return y, m, d
        except (ValueError, AttributeError): return None

_THOUS = re.compile(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$")

def to_value(s):
    s = str(s).strip()
    amount = NumberValue.parse(s.replace(",", "") if _THOUS.match(s) else s)
    if amount is not None: return NumberValue(amount, s)
    ymd = DateValue.parse(s)
    if ymd is not None: return DateValue(*ymd, s)
    return StringValue(s)

def official_match(pred_list, gold_list, tol=None):
    if pred_list is None: return False
    if tol is not None:
        old = TOL[0]; TOL[0] = tol
        try: return official_match(pred_list, gold_list)
        finally: TOL[0] = old
    P = list({normalize(p): to_value(p) for p in pred_list if str(p).strip() != ""}.values())
    T = list({normalize(g): to_value(g) for g in gold_list}.values())
    if len(P) != len(T) or not T: return False
    return all(any(t.match(p) for p in P) for t in T)


# ---------------------------------------------------------------- v8 lenient matcher (unchanged)
def _wtq_norm(s):
    s = unicodedata.normalize("NFKD", str(s)); s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = (s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-")).lower().strip()
    s = re.sub(r"\s*\[[^\]]*\]\s*$", "", s).strip().strip('"').strip("'").strip()
    return re.sub(r"\s+", " ", s)

def _as_num(s):
    try: return float(s.replace(",", ""))
    except ValueError: return None

def wtq_match(pred, gold):
    a, b = _wtq_norm(pred), _wtq_norm(gold)
    if not a: return False
    if a == b: return True
    na, nb = _as_num(a), _as_num(b)
    return na is not None and nb is not None and abs(na - nb) < 1e-6

def lenient_match(pred_list, gold_list):
    if pred_list is None: return False
    P = [p for p in pred_list if str(p).strip()]
    if len(set(_wtq_norm(p) for p in P)) != len(set(_wtq_norm(g) for g in gold_list)): return False
    return all(any(wtq_match(p, g) for p in P) for g in gold_list)


# ---------------------------------------------------------------- turning raw outputs into answer lists
def split_answer(s):
    """A direct answer string -> list. Lists may be written 'a | b', '["a", "b"]' or 'a; b'."""
    if s is None: return None
    t = str(s).strip()
    if t.startswith("[") and t.endswith("]"):
        try:
            v = json.loads(t)
            if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
        except Exception: pass
    for sep in (" | ", "|", "; "):
        if sep in t: return [x.strip() for x in t.split(sep) if x.strip()]
    return [t] if t else None


def score(pred_list, gold_list):
    """official: primary metric | tol: official logic with numbers equal within 0.01 (the numeric rule of Table-R1's judge)
    | lenient: the v1-v8 matcher."""
    return dict(official=bool(official_match(pred_list, gold_list)), tol=bool(official_match(pred_list, gold_list, tol=0.01)),
                lenient=bool(lenient_match(pred_list, gold_list)))
