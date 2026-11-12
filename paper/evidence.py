"""Evidence-integration localization (classical form of the 'evidence transfer' idea) with the extended metric set.
Signals per symbol: lex (BM25), graph (2-hop from seeds), emb (similarity to seed embeddings), path (behavioral: file/module path or traceback frame named in the issue).
Fusions: rrf3, additive, interference ((sum of sqrt-normalised amplitudes)^2, cross terms reward agreeing signals), diffuse (BM25 relevance passed to graph neighbours).
Metrics at file level: Hit@1/5/10, Recall@5/10/25, nDCG@10, all-gold@10, with bootstrap CI for differences vs BM25. Writes paper/evidence.md."""
import json, os, sys, math, re, collections, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study

D = study.D
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.path.join(D, "tasks.jsonl"), encoding="utf8")}
per = json.load(open(os.path.join(HERE, "results.json"), encoding="utf8"))["per_task"]
n = len(per)
rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(5000, n))


def bm25_vec(R, q):
    qt = collections.Counter(study.split_ident(q) + study.tok(q))
    s = np.zeros(R.N, dtype=np.float64)
    for t, qc in qt.items():
        if t not in R.df:
            continue
        idf = math.log(1 + (R.N - R.df[t] + 0.5) / (R.df[t] + 0.5))
        for k, tf in enumerate(R.tf):
            f = tf.get(t, 0)
            if f:
                s[k] += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * R.dl[k] / R.avg))
    return s


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


def emb_vec(R, seeds):
    s = np.zeros(R.N)
    sv = [R.eidx[x] for x in seeds if x in R.eidx]
    if not sv:
        return s
    q = R.E[sv].mean(axis=0)
    sc = R.E @ q
    for j, i in enumerate(R.eids):
        s[R.idx[i]] = max(0.0, float(sc[j]))
    return s


def path_vec(R, q):
    """Behavioral evidence: dotted module paths of symbols whose file path (or traceback frame) is quoted in the issue."""
    s = np.zeros(R.N)
    paths = set(m.replace("/", ".").removesuffix(".py") for m in re.findall(r"[A-Za-z_][\w/]*\.py", q))
    if not paths:
        return s
    for k, i in enumerate(R.ids):
        mod = ".".join(i.split(".")[:-1]) if "." in i else i
        if any(mod.endswith(p) or p.endswith(mod) and len(mod) > 3 for p in paths):
            s[k] = 1.0
    return s


def norm(v):
    m = v.max()
    return v / m if m > 0 else v


def diffuse(R, s, beta=0.5):
    out = s.copy()
    for k, i in enumerate(R.ids):
        nb = R.adj[i]
        if nb:
            out[k] += beta * np.mean([s[R.idx[j]] for j in nb])
    return out


def order(R, s):
    return [R.ids[k] for k in np.argsort(-s)[:300] if s[k] > 0]


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


NAMES = ["bm25", "graph", "emb", "path", "rrf3", "additive", "interference", "diffuse", "diffuse+path"]
rows = {k: [] for k in NAMES}
cache = {}
t0 = time.time()
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
    if not gold:
        continue
    q = t["problem_statement"]
    sd = R.seeds(q)
    lex, gr, em, pa = bm25_vec(R, q), graph_vec(R, sd), emb_vec(R, sd), path_vec(R, q)
    nl, ng, ne, npth = norm(lex), norm(gr), norm(em), norm(pa)
    amp = np.sqrt(nl) + np.sqrt(ng) + np.sqrt(ne) + np.sqrt(npth)
    sig = {
        "bm25": lex, "graph": gr, "emb": em, "path": pa,
        "additive": nl + ng + ne + npth,
        "interference": amp ** 2,
        "diffuse": diffuse(R, nl),
        "diffuse+path": diffuse(R, nl) + npth,
    }
    runs = {k: order(R, v) for k, v in sig.items()}
    runs["rrf3"] = study.rrf([runs["bm25"], runs["graph"], runs["path"]])
    for k in NAMES:
        rows[k].append(metrics(study.to_files(runs[k], gm), gold))
print("tasks", len(rows["bm25"]), "secs", round(time.time() - t0))

KEYS = list(rows["bm25"][0].keys())
M = {k: {m: np.array([r[m] for r in rows[k]]) for m in KEYS} for k in NAMES}
nn = len(rows["bm25"])
idx = rng.integers(0, nn, size=(5000, nn))
lines = [f"n = {nn} tasks with at least one gold file in the graph", "", "| Method | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
for k in NAMES:
    lines.append(f"| {k} | " + " | ".join(f"{M[k][m].mean():.3f}" for m in KEYS) + " |")
lines += ["", "Difference vs bm25 (mean, 95% bootstrap CI) for hit@5, recall@10 and ndcg@10:", "", "| Method | hit@5 | recall@10 | ndcg@10 |", "|---|---|---|---|"]
for k in NAMES[1:]:
    cells = []
    for m in ("hit@5", "rec@10", "ndcg@10"):
        d = M[k][m] - M["bm25"][m]
        b = d[idx].mean(axis=1)
        cells.append(f"{d.mean():+.3f} [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}]")
    lines.append(f"| {k} | " + " | ".join(cells) + " |")
open(os.path.join(HERE, "evidence.md"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
