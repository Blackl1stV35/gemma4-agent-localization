"""Sensitivity study (graph hops, fusion constant) and a learned router with leave-one-out validation.
Writes paper/extra.md. Needs paper/results.json from study.py and the data in paper/data."""
import json, os, re, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study

D = study.D
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.path.join(D, "tasks.jsonl"), encoding="utf8")}
per = json.load(open(os.path.join(HERE, "results.json"), encoding="utf8"))["per_task"]
rng = np.random.default_rng(0)
n = len(per)
idx = rng.integers(0, n, size=(10000, n))


def rr(r):
    return 1.0 / r if r else 0.0


def ci(v):
    b = v[idx].mean(axis=1)
    return f"{v.mean():.3f} [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]"


lines = []
# ---------------- sensitivity ----------------
variants = collections.OrderedDict()
for h in (1, 2, 3):
    variants[f"graph_{h}hop"] = []
for k in (10, 60, 100):
    variants[f"bm25_graph_rrf_k{k}"] = []
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
    goldset = set(gm)
    sd = R.seeds(t["problem_statement"])
    lex = R.bm25(t["problem_statement"])
    for h in (1, 2, 3):
        g = R.graph_rank(sd, hops=h) if sd else []
        files = study.to_files(g, gm)
        variants[f"graph_{h}hop"].append(rr(next((i + 1 for i, f in enumerate(files) if f in goldset), None)))
    g2 = R.graph_rank(sd) if sd else []
    for k in (10, 60, 100):
        fused = study.rrf([lex, g2], k=k)
        files = study.to_files(fused, gm)
        variants[f"bm25_graph_rrf_k{k}"].append(rr(next((i + 1 for i, f in enumerate(files) if f in goldset), None)))
lines += [f"## Sensitivity (MRR with 95 percent bootstrap CI, n = {n})", "", "| Variant | MRR [95% CI] |", "|---|---|"]
for k, v in variants.items():
    lines.append(f"| {k} | {ci(np.array(v))} |")

# ---------------- learned router ----------------
from sklearn.linear_model import LogisticRegression

M_bm25 = np.array([rr(r["lexical_bm25"]) for r in per])
M_fus = np.array([rr(r["hybrid_bm25_graph"]) for r in per])
M_rrf = np.array([rr(r["hybrid_rrf"]) for r in per])
M_graph = np.array([rr(r["graph_2hop"]) for r in per])


def feats(row, with_repo):
    t = tasks[row["id"]]
    s = t["problem_statement"]
    ids = re.findall(r"\b[a-z]+_[a-z_0-9]+\b|\b[A-Z][a-z]+[A-Z]\w*\b", s)
    f = [np.log1p(len(s)), row["n_seeds"], np.log1p(len(ids)), float("Traceback" in s), s.count("`") / 2.0, float(row["n_seeds"] > 0)]
    if with_repo:
        rp = t["repo"].split("/")[-1].lower()
        f += [float(rp == "fastapi"), float(rp == "rich"), float(rp == "requests")]
    return f


def loo(with_repo, a, b):
    """Leave-one-out: predict whether method b beats method a on a task."""
    X = np.array([feats(r, with_repo) for r in per])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = (b > a).astype(int)
    out = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        if len(set(y[tr])) < 2:
            out[i] = a[i]
            continue
        clf = LogisticRegression(C=0.5, max_iter=200).fit(X[tr], y[tr])
        out[i] = b[i] if clf.predict(X[i:i + 1])[0] == 1 else a[i]
    return out


routed = {
    "router_loo_with_repo (BM25 vs BM25+graph)": loo(True, M_bm25, M_fus),
    "router_loo_no_repo (BM25 vs BM25+graph)": loo(False, M_bm25, M_fus),
    "router_loo_with_repo (BM25 vs graph)": loo(True, M_bm25, M_graph),
}
lines += ["", "## Learned router (logistic regression, leave-one-out)", "", "| Method | MRR [95% CI] |", "|---|---|", f"| lexical_bm25 | {ci(M_bm25)} |", f"| hybrid_bm25_graph | {ci(M_fus)} |"]
for k, v in routed.items():
    lines.append(f"| {k} | {ci(v)} |")
lines += ["", "Paired differences (bootstrap):", "", "| Comparison | mean diff [95% CI] | P(diff<=0) |", "|---|---|---|"]
for k, v in routed.items():
    for name, base in (("lexical_bm25", M_bm25), ("hybrid_bm25_graph", M_fus)):
        d = v - base
        bd = d[idx].mean(axis=1)
        lines.append(f"| {k} minus {name} | {d.mean():+.3f} [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}] | {(bd<=0).mean():.3f} |")
open(os.path.join(HERE, "extra.md"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
