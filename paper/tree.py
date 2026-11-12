"""Hierarchical (tree-structured) localization with multi-frequency memories and a cross-level parity check.
Inspired by Hope / Nested Learning (Behrouz et al. 2025): memories that update at different frequencies (continuum memory system).
Levels: fast = symbol, mid = module (file), slow = package. Parity = agreement between the symbol-level and module-level views.
Variants: bm25 (symbol), mod_bm25, fast_mid, cms3 (fast+mid+slow), cms3_parity, cms3_parity_graph (adds 2-hop graph evidence). Writes paper/tree.md."""
import json, os, sys, math, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study

D = study.D
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.path.join(D, "tasks.jsonl"), encoding="utf8")}
per = json.load(open(os.path.join(HERE, "results.json"), encoding="utf8"))["per_task"]
rng = np.random.default_rng(0)


def bm25_generic(docs_tf, dl, q_terms, N, df, avg):
    s = np.zeros(N)
    for t, qc in q_terms.items():
        if t not in df:
            continue
        idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
        for k, tf in enumerate(docs_tf):
            f = tf.get(t, 0)
            if f:
                s[k] += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl[k] / avg))
    return s


def norm(v):
    m = v.max()
    return v / m if m > 0 else v


def metrics(files, gold):
    g = len(gold)
    pos = [i + 1 for i, f in enumerate(files) if f in gold]
    out = {}
    for k in (1, 5, 10):
        out[f"hit@{k}"] = float(any(p <= k for p in pos))
    for k in (5, 10, 25):
        out[f"rec@{k}"] = sum(1 for p in pos if p <= k) / g
    dcg = sum(1 / math.log2(p + 1) for p in pos if p <= 10)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(g, 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    out["allgold@10"] = float(sum(1 for p in pos if p <= 10) == g)
    return out


def graph_vec(R, seeds):
    s = np.zeros(R.N)
    if not seeds:
        return s
    dist = {x: 0 for x in seeds}
    fr = list(seeds)
    for h in (1, 2):
        nx = []
        for u in fr:
            for v in R.adj[u]:
                if v not in dist:
                    dist[v] = h
                    nx.append(v)
        fr = nx
    for k, d in dist.items():
        s[R.idx[k]] = 1.0 / (1 + d)
    return s


class Tree:
    def __init__(self, R):
        self.R = R
        mod = [".".join(i.split(".")[:-1]) if "." in i else i for i in R.ids]
        self.mod = mod
        self.mods = sorted(set(mod))
        self.midx = {m: k for k, m in enumerate(self.mods)}
        pkg = lambda m: ".".join(m.split(".")[:-1]) if "." in m else m
        self.pkgs = sorted(set(pkg(m) for m in self.mods))
        self.pidx = {p: k for k, p in enumerate(self.pkgs)}
        self.m2p = [self.pidx[pkg(m)] for m in self.mods]
        self.node_mod = np.array([self.midx[m] for m in mod])
        mtf = [collections.Counter() for _ in self.mods]
        for k, tf in enumerate(R.tf):
            mtf[self.node_mod[k]].update(tf)
        self.mtf = mtf
        self.mdl = np.array([sum(c.values()) for c in mtf], dtype=np.float64)
        self.mdf = collections.Counter(t for c in mtf for t in c)
        ptf = [collections.Counter() for _ in self.pkgs]
        for mi, c in enumerate(mtf):
            ptf[self.m2p[mi]].update(c)
        self.ptf = ptf
        self.pdl = np.array([sum(c.values()) for c in ptf], dtype=np.float64)
        self.pdf = collections.Counter(t for c in ptf for t in c)

    def views(self, q):
        R = self.R
        qt = collections.Counter(study.split_ident(q) + study.tok(q))
        fast = np.zeros(R.N)
        for t, qc in qt.items():
            if t not in R.df:
                continue
            idf = math.log(1 + (R.N - R.df[t] + 0.5) / (R.df[t] + 0.5))
            for k, tf in enumerate(R.tf):
                f = tf.get(t, 0)
                if f:
                    fast[k] += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * R.dl[k] / R.avg))
        mid = bm25_generic(self.mtf, self.mdl, qt, len(self.mods), self.mdf, self.mdl.mean() + 1e-9)
        slow = bm25_generic(self.ptf, self.pdl, qt, len(self.pkgs), self.pdf, self.pdl.mean() + 1e-9)
        return fast, mid, slow

    def module_scores(self, fast, mid, slow, graph=None, mode="cms3"):
        nm = len(self.mods)
        fmax = np.zeros(nm)
        gmax = np.zeros(nm)
        for k in range(self.R.N):
            m = self.node_mod[k]
            if fast[k] > fmax[m]:
                fmax[m] = fast[k]
            if graph is not None and graph[k] > gmax[m]:
                gmax[m] = graph[k]
        nf, nmid = norm(fmax), norm(mid)
        nslow = norm(slow)[self.m2p]
        if mode == "mod_bm25":
            return nmid, fmax
        if mode == "fast_mid":
            return nf + nmid, fmax
        s = nf + nmid + 0.5 * nslow
        if mode in ("cms3_parity", "cms3_parity_graph"):
            rf = np.argsort(np.argsort(-nf))
            rm = np.argsort(np.argsort(-nmid))
            agree = ((rf < 5) & (rm < 5)).astype(float)
            disagree = (np.abs(rf - rm) > 25).astype(float)
            s = s + 0.5 * agree - 0.25 * disagree
        if mode == "cms3_parity_graph":
            s = s + norm(gmax)
        return s, fmax

    def ranking(self, ms, fast):
        best = {}
        for k in range(self.R.N):
            m = self.node_mod[k]
            if m not in best or fast[k] > fast[best[m]]:
                best[m] = k
        order = np.argsort(-ms)
        return [self.R.ids[best[m]] for m in order[:300] if ms[m] > 0 and m in best]


NAMES = ["bm25", "mod_bm25", "fast_mid", "cms3", "cms3_parity", "cms3_parity_graph"]
rows = {k: [] for k in NAMES}
cache = {}
for row in per:
    t = tasks[row["id"]]
    key = f"{t['repo'].split('/')[-1]}_{t['base_commit']}"
    if key not in cache:
        cache.clear()
        R = study.Repo(key)
        cache[key] = (R, Tree(R))
    R, T = cache[key]
    gm = {f: study.best_mods(f, R.ids) for f in study.gold_files(t["patch"])}
    gm = {f: m for f, m in gm.items() if m}
    gold = set(gm)
    if not gold:
        continue
    q = t["problem_statement"]
    fast, mid, slow = T.views(q)
    gr = graph_vec(R, R.seeds(q))
    runs = {"bm25": [R.ids[k] for k in np.argsort(-fast)[:300] if fast[k] > 0]}
    for mode in NAMES[1:]:
        ms, _ = T.module_scores(fast, mid, slow, gr, mode)
        runs[mode] = T.ranking(ms, fast)
    for k in NAMES:
        rows[k].append(metrics(study.to_files(runs[k], gm), gold))

KEYS = list(rows["bm25"][0].keys())
M = {k: {m: np.array([r[m] for r in rows[k]]) for m in KEYS} for k in NAMES}
nn = len(rows["bm25"])
idx = rng.integers(0, nn, size=(5000, nn))
lines = [f"n = {nn}", "", "| Method | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
for k in NAMES:
    lines.append(f"| {k} | " + " | ".join(f"{M[k][m].mean():.3f}" for m in KEYS) + " |")
lines += ["", "Difference vs bm25 (mean, 95% bootstrap CI):", "", "| Method | hit@5 | recall@10 | ndcg@10 |", "|---|---|---|---|"]
for k in NAMES[1:]:
    cells = []
    for m in ("hit@5", "rec@10", "ndcg@10"):
        d = M[k][m] - M["bm25"][m]
        b = d[idx].mean(axis=1)
        cells.append(f"{d.mean():+.3f} [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}]")
    lines.append(f"| {k} | " + " | ".join(cells) + " |")
open(os.path.join(HERE, "tree.md"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
