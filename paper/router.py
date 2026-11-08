"""Router and oracle analysis from results.json: does choosing the signal per issue beat a fixed method?"""
import json, os, numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
per = json.load(open(os.path.join(OUT, "results.json"), encoding="utf8"))["per_task"]
methods = ["lexical_bm25", "seed_only", "graph_2hop", "embedding_seed", "hybrid_rrf", "hybrid_bm25_graph"]
rng = np.random.default_rng(0)
n = len(per)
idx = rng.integers(0, n, size=(10000, n))


def rr(r):
    return 1.0 / r if r else 0.0


M = {m: np.array([rr(r.get(m)) for r in per]) for m in methods}
# rule router: a seed exists -> BM25 + graph fusion, otherwise BM25
M["router_seed_rule"] = np.array([rr(r["hybrid_bm25_graph"]) if r["n_seeds"] > 0 else rr(r["lexical_bm25"]) for r in per])
# oracle router: best method per task (upper bound, uses gold labels)
M["oracle_best_of_all"] = np.array([max(rr(r.get(m)) for m in methods) for r in per])
M["oracle_bm25_or_graph"] = np.array([max(rr(r["lexical_bm25"]), rr(r["hybrid_bm25_graph"])) for r in per])
lines = ["| Method | MRR | 95% CI |", "|---|---|---|"]
for k, v in M.items():
    b = v[idx].mean(axis=1)
    lines.append(f"| {k} | {v.mean():.3f} | [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}] |")
lines += ["", "Router minus BM25 and router minus the fixed fusion (paired bootstrap):", "", "| Comparison | mean diff | 95% CI | P(diff<=0) |", "|---|---|---|---|"]
for a, b_ in [("router_seed_rule", "lexical_bm25"), ("router_seed_rule", "hybrid_bm25_graph")]:
    d = M[a] - M[b_]
    bd = d[idx].mean(axis=1)
    lines.append(f"| {a} minus {b_} | {d.mean():+.3f} | [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}] | {(bd<=0).mean():.3f} |")
open(os.path.join(OUT, "router.md"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
