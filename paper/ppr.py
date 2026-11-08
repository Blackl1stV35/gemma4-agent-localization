"""HippoRAG-style retrieval: Personalized PageRank from seed symbols over the code graph (hippocampal-index analogy).
Compares PPR with the 2-hop graph, BM25, and fusions. Writes paper/ppr.md. Needs paper/data and results.json."""
import json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study

D = study.D
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.path.join(D, "tasks.jsonl"), encoding="utf8")}
per = json.load(open(os.path.join(HERE, "results.json"), encoding="utf8"))["per_task"]
n = len(per)
rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(10000, n))


def rr(r):
    return 1.0 / r if r else 0.0


def ppr(R, seeds, alpha=0.5, iters=30):
    N = len(R.ids)
    deg = np.array([max(1, len(R.adj[i])) for i in R.ids], dtype=np.float64)
    p0 = np.zeros(N)
    for s in seeds:
        p0[R.idx[s]] = 1.0 / len(seeds)
    p = p0.copy()
    nbr = [[R.idx[v] for v in R.adj[i]] for i in R.ids]
    for _ in range(iters):
        nxt = np.zeros(N)
        share = p / deg
        for u in range(N):
            if share[u] > 0:
                for v in nbr[u]:
                    nxt[v] += share[u]
        p = (1 - alpha) * p0 + alpha * nxt
    order = np.argsort(-p)
    return [R.ids[k] for k in order if p[k] > 0][:200]


res = collections.OrderedDict((k, []) for k in ("lexical_bm25", "graph_2hop", "ppr", "rrf_bm25_graph", "rrf_bm25_ppr"))
cache = {}
for row in per:
    t = tasks[row["id"]]
    key = f"{t['repo'].split('/')[-1]}_{t['base_commit']}"
    if key not in cache:
        cache.clear()
        cache[key] = study.Repo(key)
    R = cache[key]
    gm = {f: study.best_mods(f, R.ids) for f in study.gold_files(t["patch"])}
    gm = {f: m for f, m in gm.items() if m}
    gold = set(gm)
    q = t["problem_statement"]
    sd = R.seeds(q)
    lex = R.bm25(q)
    g2 = R.graph_rank(sd) if sd else []
    pp = ppr(R, sd) if sd else []
    runs = {"lexical_bm25": lex, "graph_2hop": g2, "ppr": pp, "rrf_bm25_graph": study.rrf([lex, g2]), "rrf_bm25_ppr": study.rrf([lex, pp])}
    for m, nodes in runs.items():
        files = study.to_files(nodes, gm)
        res[m].append(rr(next((i + 1 for i, f in enumerate(files) if f in gold), None)))
M = {k: np.array(v) for k, v in res.items()}


def ci(v):
    b = v[idx].mean(axis=1)
    return f"{v.mean():.3f} [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]"


lines = [f"n = {n}", "", "| Method | MRR [95% CI] |", "|---|---|"] + [f"| {k} | {ci(v)} |" for k, v in M.items()]
lines += ["", "| Comparison | mean diff [95% CI] | P(diff<=0) |", "|---|---|---|"]
for a, b_ in [("ppr", "graph_2hop"), ("rrf_bm25_ppr", "lexical_bm25"), ("rrf_bm25_ppr", "rrf_bm25_graph")]:
    d = M[a] - M[b_]
    bd = d[idx].mean(axis=1)
    lines.append(f"| {a} minus {b_} | {d.mean():+.3f} [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}] | {(bd<=0).mean():.3f} |")
open(os.path.join(HERE, "ppr.md"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
