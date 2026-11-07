"""Bootstrap confidence intervals, paired tests and per-repo breakdown from results.json (written by study.py)."""
import json, os, numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "results.json"), encoding="utf8"))
per = R["per_task"]
methods = ["lexical_bm25", "seed_only", "graph_2hop", "embedding_seed", "hybrid_rrf", "hybrid_bm25_graph"]
rng = np.random.default_rng(0)


def rr(row, m):
    r = row.get(m)
    return 1.0 / r if r else 0.0


def hit(row, m, k):
    r = row.get(m)
    return 1.0 if r and r <= k else 0.0


def repo_of(i):
    return i.split("_")[0]


M = {m: np.array([rr(r, m) for r in per]) for m in methods}
H10 = {m: np.array([hit(r, m, 10) for r in per]) for m in methods}
n = len(per)
B = 10000
idx = rng.integers(0, n, size=(B, n))
lines = [f"n = {n} tasks, {B} bootstrap resamples (seed 0)", "", "| Method | MRR | 95% CI | hit@10 | 95% CI |", "|---|---|---|---|---|"]
for m in methods:
    b = M[m][idx].mean(axis=1)
    h = H10[m][idx].mean(axis=1)
    lines.append(f"| {m} | {M[m].mean():.3f} | [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}] | {H10[m].mean():.3f} | [{np.percentile(h,2.5):.3f}, {np.percentile(h,97.5):.3f}] |")
lines += ["", "Paired differences in MRR (bootstrap over tasks):", "", "| Comparison | mean diff | 95% CI | P(diff<=0) |", "|---|---|---|---|"]
for a, b_ in [("hybrid_bm25_graph", "lexical_bm25"), ("hybrid_rrf", "lexical_bm25"), ("graph_2hop", "seed_only"), ("embedding_seed", "graph_2hop"), ("hybrid_bm25_graph", "hybrid_rrf")]:
    d = M[a] - M[b_]
    bd = d[idx].mean(axis=1)
    lines.append(f"| {a} minus {b_} | {d.mean():+.3f} | [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}] | {(bd<=0).mean():.3f} |")
repos = sorted(set(repo_of(r["id"]) for r in per))
lines += ["", "Per-repository MRR:", "", "| Repository | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
for rp in repos:
    sel = np.array([repo_of(r["id"]) == rp for r in per])
    lines.append(f"| {rp} | {int(sel.sum())} | " + " | ".join(f"{M[m][sel].mean():.3f}" for m in methods) + " |")
open(os.path.join(OUT, "analysis.md"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
