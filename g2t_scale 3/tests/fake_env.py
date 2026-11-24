"""CPU-only end-to-end test of the non-GPU code paths: data -> eval -> cf -> report, with a fake model/tokenizer/torch."""
import sys, types, os, json, random, re, csv
torch = types.ModuleType("torch"); torch.bfloat16 = "bf16"; torch.float16 = "fp16"; torch.float32 = "fp32"
torch.no_grad = lambda: (lambda f: f); torch.manual_seed = lambda s: None
torch.cuda = types.SimpleNamespace(manual_seed_all=lambda s: None, empty_cache=lambda: None, memory_allocated=lambda *a: 0)
sys.modules["torch"] = torch
sys.modules["torch.nn"] = types.ModuleType("torch.nn"); sys.modules["torch.nn.functional"] = types.ModuleType("torch.nn.functional")
ROOT = "/tmp/g2troot"
def mk_table(rng):
    teams = rng.sample(["Lakers", "Bulls", "Nets", "Heat", "Suns", "Jazz", "Kings", "Magic"], 5)
    rows = [[t, str(rng.randint(10, 99)), str(rng.choice([2018, 2019, 2020])), f"{rng.randint(1,30):,}00"] for t in teams]
    return ["Team", "Points", "Season", "Attendance"], rows
def mk_q(rng, i):
    h, r = mk_table(rng); k = rng.randrange(3)
    if k == 0: t = rng.choice(r); return dict(id=f"nt-{i}", question=f"how many points did {t[0]} score?", headers=h, rows=r, answers=[t[1]])
    if k == 1: best = max(r, key=lambda x: int(x[1])); return dict(id=f"nt-{i}", question="which team scored the most points?", headers=h, rows=r, answers=[best[0]])
    s = rng.choice(["2018", "2019", "2020"]); ts = [x[0] for x in r if x[2] == s]
    if len(ts) >= 2 and rng.random() < 0.3: return dict(id=f"nt-{i}", question=f"which teams played in {s}?", headers=h, rows=r, answers=ts)
    return dict(id=f"nt-{i}", question=f"how many teams played in {s}?", headers=h, rows=r, answers=[str(len(ts))])
rng = random.Random(0); os.makedirs(f"{ROOT}/data/wtq", exist_ok=True); os.makedirs(f"{ROOT}/data/sparks", exist_ok=True)
for split, n in (("train", 120), ("validation", 30), ("test", 60)):
    with open(f"{ROOT}/data/wtq/{split}.jsonl", "w") as f:
        for i in range(n): f.write(json.dumps(mk_q(rng, f"{split}{i}")) + "\n")
with open(f"{ROOT}/data/sparks/cot.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["question", "context", "cot", "sql"])
    for i in range(10): w.writerow([f"how many rows {i}", "CREATE TABLE t (a TEXT)", "Count all rows.", "SELECT COUNT(*) FROM t"])
os.makedirs(f"{ROOT}/data/crtqa/raw/all_csv", exist_ok=True); crt = {}
for i in range(15):
    h, r = mk_table(rng); name = f"t{i}.csv"
    with open(f"{ROOT}/data/crtqa/raw/all_csv/{name}", "w", newline="") as f: csv.writer(f).writerows([h] + r)
    best = max(r, key=lambda x: int(x[1]))
    crt[name] = [{"Question name": f"which team has the most points in table {i}?", "Title": "t", "Answer": best[0], "Directness": "Explicit"}]
json.dump(crt, open(f"{ROOT}/data/crtqa/raw/dataset.json", "w"))
class FakeTok:
    eos_token = "<eos>"; eos_token_id = 0; pad_token_id = 0; pad_token = "<pad>"; padding_side = "left"
    def __call__(self, text, add_special_tokens=False): return types.SimpleNamespace(input_ids=[hash(w) % 997 + 1 for w in str(text).split()])
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True): return "<chat>" + "\n".join(m["content"] for m in msgs)
sys.path.insert(0, os.getcwd())
import g2t.model as M
M.load_tokenizer = lambda path: FakeTok()
EX, GQ, TEACH = {}, {}, {}
def register_examples(exs):
    for e in exs: EX.setdefault(e.question, e)
class FakeGen:
    def __init__(self, cfg): self.cfg = cfg; self.r = random.Random(1)
    def n_tok(self, s): return len(str(s).split())
    def score(self, seqs, lens, adapter=None, bs=None): return [self.r.random() for _ in seqs]
    def generate(self, prompts, max_new, sample=False, temp=1.0, top_p=1.0, n=1, adapter=None, log=None, tag=""):
        import g2t.graph_tasks as G
        from g2t.teacher import ekey
        out = []
        for p in prompts:
            m = re.search(r"Question: (.*?)(\n|$)", p); q = m.group(1).strip() if m else ""
            good = self.r.random() < 0.6
            if tag.startswith("graph"):
                gq = GQ.get(q)
                if gq is None: out.append([("ANS: 3", 3)]); continue
                v = G.view(gq, "gtable")
                if "graph/ours" in tag: t = G.target(v, "nodes").strip() if good else f"PATH: {gq.src} > Nowhere"
                elif "graph/direct" in tag: t = G.target(v, "ans").strip() if good else "ANS: 2"
                elif "sparks" in tag: t = "<sql>SELECT dst FROM edges LIMIT 1</sql>"
                elif "cot" in tag or "table_r1" in tag: t = '<think>x</think><answer>{"answer": ["%s"]}</answer>' % (gq.gold if good else "7")
                else: t = gq.gold if good else "5"
                out.append([(t, len(t.split()))]); continue
            e = EX.get(q); gold = " | ".join(e.answers) if e else "x"
            if "sparks" in tag: t = "Reasoning.\n<sql>SELECT team FROM t ORDER BY points DESC LIMIT 1</sql>" if good else "<sql>SELECT nope FROM t</sql>"
            elif "zs_cot" in tag or "table_r1" in tag: t = '<think>hmm</think> <answer>{"answer": %s}</answer>' % json.dumps(e.answers if (e and good) else ["9"])
            elif "fallback" in tag: t = gold
            elif "ours" in tag:
                T = TEACH.get(ekey(e)) if e else None
                t = ("PROG: " + T["expr"]) if (T and good) else ("PROG: MAX(C9)" if self.r.random() < 0.2 else "ANS: " + (gold if good else "1"))
                if p.endswith(" PROG:"): t = (T["expr"] if T else "COUNT(C0)")
                if p.endswith(" ANS:"): t = gold if good else "1"
            elif "zs_direct" in tag: t = gold if good else "wrong"
            elif "direct" in tag: t = "ANS: " + (gold if good else "1")
            else: t = gold if good else "wrong"
            out.append([(t, len(t.split()))] * n)
        return out
