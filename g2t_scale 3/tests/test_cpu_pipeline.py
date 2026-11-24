import sys, os, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fake_env as FE
import run as RUN
from g2t.config import load_config
from g2t.logs import get_logger, run_paths, mark_done
shutil.rmtree("/tmp/g2troot/runs", ignore_errors=True)
cfg = load_config("config.yaml", smoke=True, overrides=dict(paths=dict(root="/tmp/g2troot"), wtq=dict(teacher_workers=2)))
P = run_paths(cfg)
log = get_logger(cfg, "data"); RUN.stage_data(cfg, log)
D = RUN.wtq_data(cfg); FE.register_examples(D["train"] + D["dev"] + D["test"])
from g2t.crtqa_data import load_crtqa
FE.register_examples(load_crtqa(cfg))
FE.TEACH.update(D["teacher"]["test"]); FE.TEACH.update(D["teacher"]["train"])
gs = RUN.graph_splits(cfg)
for k in ("test_id", "test_ood"):
    for q in gs[k]: FE.GQ[q.question] = q
import g2t.graph_tasks as G
for q, s, i in G.build_cf(gs["test_id"], cfg["cf"]["graph_n"], cfg["graph"]["data_seed"] + 7):
    for x in (q, s, i): FE.GQ.setdefault(x.question, x)
# fake adapters exist
for a in ["sparks_sft", "sparks_grpo", "sparks_bird_grpo", "wtq_direct_sft", "wtq_direct_grpo", "wtq_ours_sft", "wtq_ours_rft", "wtq_ours_grpo",
          "graph_direct_sft", "graph_direct_grpo", "graph_ours_sft", "graph_ours_grpo"]:
    os.makedirs(os.path.join(P["adapters"], a), exist_ok=True)
RUN._make_gen = lambda cfg, key, adapters, log: (FE.FakeGen(cfg), FE.FakeTok())
RUN._close = lambda g: None
for sh in (0, 1):                                   # two evaluation shards, merged by the report
    log = get_logger(cfg, f"eval_shard{sh}"); RUN.stage_eval(cfg, log, sh, 2)
log = get_logger(cfg, "cf"); RUN.stage_cf(cfg, log)
# goal-conditioned masked decoding: never a hallucinated hop; hops/cost paths end at the destination or are NONE
from g2t.systems import masked_decode
for goal in (False, True):
    outs = masked_decode(FE.FakeGen(cfg), FE.FakeTok(), gs["test_id"] + gs["test_ood"], "gtable", "graph_ours_grpo", 10, goal=goal)
    for q, (t, n) in zip(gs["test_id"] + gs["test_ood"], outs):
        r = G.execute(G.view(q, "gtable"), "nodes", t); assert r["status"] not in ("missing_edge", "unknown_edge", "broken_chain"), (t, r)
        if goal and q.task in ("hops", "cost"): assert r["status"] in ("executed", "none"), (q.task, t, r)
print("masked decoding checks passed (free + goal-conditioned)")
from g2t.sparks_data import bird_reward, BirdExample
be = BirdExample("b1", "q", "", "SELECT a FROM t WHERE b = 1", "db", "CREATE TABLE t (a, b)", "/nonexistent")
assert bird_reward(be, "<sql>SELECT a FROM t WHERE b = 1</sql>")[0] == 1.0 and bird_reward(be, "no sql")[0] == 0.0
print("bird reward checks passed")
from g2t.report import stage_report
log = get_logger(cfg, "report"); stage_report(cfg, log)
# resumability: second eval call must do nothing
log = get_logger(cfg, "eval"); RUN.stage_eval(cfg, log)
rep = open(os.path.join(P["root"], "report.md")).read()
assert "## CRTQA" in rep and "head-to-head" in rep and "Published results (CRTQA)" in rep and "Published results (WTQ)" in rep, "CRT-QA / published rows missing"
print(rep[:6000])
