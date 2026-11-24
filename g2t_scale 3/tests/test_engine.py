import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2t.engine import *
# ---- unit tests (fail loudly) ----
_H = ["Team", "Points", "Season", "Date", "Speed"]
_R = [["Lakers", "50 pts", "2019", "March 5, 2006", "120 km/h"], ["Bulls", "$20", "2018", "2005-11-12", "300 km/h"], ["Lakers", "33", "2019", "12 November 2005", "1,200 km/h"], ["Nets", "41", "2020", "July 1, 2007", "45 km/h"]]
_T = lambda e: run(e, _H, _R)
for _e, _x in [("SUB(MAX(C1), MIN(C1))", "30"), ('SUM(FILTER(C1, C0=="Lakers"))', "83"), ("COUNT(FILTER(C0, C2==2019))", "2"), ("ARGMAX(C1, C0)", "Lakers"), ("MAX(C4)", "1,200 km/h"),
               ("MAX(C3)", "July 1, 2007"), ("AVG(C1)", "36"), ("MODE(C0)", "Lakers"), ('FIRST(FILTER(C1, C0=="Nets"))', "41"), ("LAST(C0)", "Nets"), ("COUNT(FILTER(C0, C3>=2006))", "2"),
               ('COUNT(FILTER(C0, C0~"lak", C2==2019))', "2"), ("DIV(SUM(C1), COUNT(C1))", "36"), ("ADD(MULT(2, 3), 1)", "7"), ("ABS(SUB(MIN(C1), MAX(C1)))", "30"),
               ("ARGMIN(FILTER(C1, C2>=2019), FILTER(C0, C2>=2019))", "Lakers")]:
    assert _T(_e)[0] == _x, (_e, _T(_e))
for _e, _k in [("SUB(MAX(C1)", "ParseError"), ("FOO(C1)", "ParseError"), ("MAX(C9)", "EvalError"), ("DIV(1, 0)", "EvalError"), ("SUM(C0)", "EvalError"), ("", "ParseError"), ("FILTER(C1)", "ParseError")]:
    assert (_T(_e)[1] or "").startswith(_k), (_e, _T(_e))
assert unparse(parse("sub( max(C2 ), MIN(C2))")) == "SUB(MAX(C2), MIN(C2))" and ast_depth(parse('SUM(FILTER(C1, C0=="Lakers"))')) == 2
print("expression-engine unit tests passed (nested programs, units/currency/dates, error handling)"); 