"""Seeded localization study: which retrieval signal reaches the files a gold patch edits.

Inputs (paper/data): tasks.jsonl, graphs/graphs/<repo>_<commit>.json, embeddings/embeddings/<repo>_<commit>.npz
Output: paper/results.json, paper/results_table.md, paper/fig_recall.png
"""
import json, re, math, collections, os, sys
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.dirname(os.path.abspath(__file__))
KS = [1, 5, 10, 25]
STOP = set("the a an and or of to in is it for on with as this that be not are was from by at if when then there use using used get set none true false self def class return import error issue bug fix should would could can will does do dont".split())


def tok(s):
    return [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s) if len(t) > 2 and t.lower() not in STOP]


def split_ident(s):
    out = []
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s):
        out.append(t.lower())
        out += [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", t.replace("_", " "))]
    return [t for t in out if len(t) > 2 and t not in STOP]


def gold_files(patch):
    fs = re.findall(r"^\+\+\+ b/(\S+)", patch, flags=re.M)
    return [f for f in fs if f.endswith(".py") and "/tests/" not in "/" + f and not os.path.basename(f).startswith("test_")]


def file_of(node_id, gold_mods):
    """Return the gold file a node belongs to, or None."""
    for f, mods in gold_mods.items():
        for m in mods:
            if node_id == m or node_id.startswith(m + "."):
                return f
    return None


def mods_for(f):
    parts = f[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts)) if parts[i:]]


def best_mods(f, ids):
    """Longest dotted suffix of the file path that matches at least one node id."""
    for m in mods_for(f):
        if any(i == m or i.startswith(m + ".") for i in ids):
            return [m]
    return []


class Repo:
    def __init__(self, key):
        g = json.load(open(os.path.join(D, "graphs", "graphs", key + ".json"), encoding="utf8"))
        self.ids = [n["id"] for n in g["nodes"]]
        self.text = {n["id"]: n.get("text", "") for n in g["nodes"]}
        self.idx = {i: k for k, i in enumerate(self.ids)}
        self.adj = collections.defaultdict(set)
        for e in g["edges"]:
            s, t = e["source"], e["target"]
            if s in self.idx and t in self.idx:
                self.adj[s].add(t)
                self.adj[t].add(s)
        z = np.load(os.path.join(D, "embeddings", "embeddings", key + ".npz"), allow_pickle=True)
        have = [i for i in self.ids if i in z.files]
        self.eids = have
        M = np.stack([z[i] for i in have]).astype(np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        self.E = M
        self.eidx = {i: k for k, i in enumerate(have)}
        # BM25 over id + text
        docs = [split_ident(i) + tok(self.text[i]) for i in self.ids]
        self.df = collections.Counter(t for d in docs for t in set(d))
        self.tf = [collections.Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = self.dl.mean() + 1e-9
        self.N = len(docs)
        self.short = collections.defaultdict(list)
        for i in self.ids:
            self.short[i.split(".")[-1]].append(i)

    def bm25(self, q):
        qt = collections.Counter(split_ident(q) + tok(q))
        s = np.zeros(self.N, dtype=np.float32)
        for t, qc in qt.items():
            if t not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for k, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[k] += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self.dl[k] / self.avg))
        return [self.ids[k] for k in np.argsort(-s)[:200] if s[k] > 0]

    def seeds(self, q):
        """Nodes whose short name or full id appears verbatim in the issue text (what an agent would pass to the tools)."""
        found = []
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", q))
        for w in sorted(words):
            w2 = w.strip(".")
            if w2 in self.idx:
                found.append(w2)
            elif len(w2) > 3 and w2 in self.short and len(self.short[w2]) <= 3:
                found += self.short[w2]
        return list(dict.fromkeys(found))

    def graph_rank(self, seeds, hops=2):
        dist = {s: 0 for s in seeds}
        fr = list(seeds)
        for h in range(1, hops + 1):
            nx = []
            for u in fr:
                for v in sorted(self.adj[u]):
                    if v not in dist:
                        dist[v] = h
                        nx.append(v)
            fr = nx
        deg = lambda n: len(self.adj[n])
        return sorted(dist, key=lambda n: (dist[n], -deg(n), n))

    def emb_rank(self, seeds, k=200):
        sv = [self.eidx[s] for s in seeds if s in self.eidx]
        if not sv:
            return []
        q = self.E[sv].mean(axis=0)
        sc = self.E @ q
        return [self.eids[j] for j in np.argsort(-sc)[:k]]


def rrf(lists, k=60):
    sc = collections.defaultdict(float)
    for L in lists:
        for r, n in enumerate(L):
            sc[n] += 1.0 / (k + r + 1)
    return sorted(sc, key=lambda n: (-sc[n], n))


def to_files(nodes, gold_mods):
    """Ordered distinct gold-file hits are computed against file-level ranking: each node maps to a file via module path."""
    seen, out = set(), []
    for n in nodes:
        top = ".".join(n.split(".")[:-1]) if "." in n else n
        # file key = node's module prefix (all but last component) as a proxy for the defining file
        f = file_of(n, gold_mods) or ("MOD:" + top)
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main():
    tasks = [json.loads(l) for l in open(os.path.join(D, "tasks.jsonl"), encoding="utf8")]
    methods = ["lexical_bm25", "seed_only", "graph_2hop", "embedding_seed", "hybrid_rrf", "hybrid_bm25_graph"]
    agg = {m: {"mrr": [], **{f"r@{k}": [] for k in KS}, "tokens@10": []} for m in methods}
    cov = {"tasks": 0, "usable": 0, "with_seed": 0, "seed_hits_gold": 0}
    per = []
    flags = []
    cache = {}
    for t in tasks:
        cov["tasks"] += 1
        key = f"{t['repo'].split('/')[-1]}_{t['base_commit']}"
        try:
            if key not in cache:
                cache.clear()
                cache[key] = Repo(key)
            R = cache[key]
        except FileNotFoundError:
            continue
        gf = gold_files(t["patch"])
        gm = {f: best_mods(f, R.ids) for f in gf}
        gm = {f: m for f, m in gm.items() if m}
        if not gm:
            continue
        cov["usable"] += 1
        q = t["problem_statement"]
        sd = R.seeds(q)
        if sd:
            cov["with_seed"] += 1
        lex = R.bm25(q)
        graph = R.graph_rank(sd) if sd else []
        emb = R.emb_rank(sd) if sd else []
        runs = {
            "lexical_bm25": lex,
            "seed_only": sd,
            "graph_2hop": graph,
            "embedding_seed": emb,
            "hybrid_rrf": rrf([lex, graph, emb]),
            "hybrid_bm25_graph": rrf([lex, graph]),
        }
        goldset = set(gm)
        if sd and any(file_of(s, gm) for s in sd):
            cov["seed_hits_gold"] += 1
        flags.append(bool(sd))
        row = {"id": t["instance_id"], "gold": sorted(goldset), "n_seeds": len(sd)}
        for m, nodes in runs.items():
            files = to_files(nodes, gm)
            rank = next((r + 1 for r, f in enumerate(files) if f in goldset), None)
            agg[m]["mrr"].append(1.0 / rank if rank else 0.0)
            for k in KS:
                top = set(files[:k])
                agg[m][f"r@{k}"].append(len(top & goldset) / len(goldset))
            agg[m]["tokens@10"].append(sum(len(R.text.get(n, "")) // 4 + 8 for n in nodes[:10]))
            row[m] = rank
        per.append(row)
    summ = {m: {k: float(np.mean(v)) if v else 0.0 for k, v in a.items()} for m, a in agg.items()}
    fl = np.array(flags)
    summ_seeded = {m: {k: float(np.mean(np.array(v)[fl])) if fl.any() else 0.0 for k, v in a.items()} for m, a in agg.items()}
    json.dump({"coverage": cov, "summary": summ, "summary_seeded_only": summ_seeded, "per_task": per}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    lines = ["| Method | MRR | R@1 | R@5 | R@10 | R@25 | tokens for top 10 |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        s = summ[m]
        lines.append(f"| {m} | {s['mrr']:.3f} | {s['r@1']:.3f} | {s['r@5']:.3f} | {s['r@10']:.3f} | {s['r@25']:.3f} | {s['tokens@10']:.0f} |")
    lines += ["", f"Seeded subset (n={int(fl.sum())}):", "", "| Method | MRR | R@1 | R@5 | R@10 | R@25 | tokens for top 10 |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        s2 = summ_seeded[m]
        lines.append(f"| {m} | {s2['mrr']:.3f} | {s2['r@1']:.3f} | {s2['r@5']:.3f} | {s2['r@10']:.3f} | {s2['r@25']:.3f} | {s2['tokens@10']:.0f} |")
    open(os.path.join(OUT, "results_table.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps(cov))
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.13
        for i, m in enumerate(methods):
            ax.bar(np.arange(len(KS)) + i * w, [summ[m][f"r@{k}"] for k in KS], w, label=m)
        ax.set_xticks(np.arange(len(KS)) + 2.5 * w)
        ax.set_xticklabels([f"R@{k}" for k in KS])
        ax.set_ylabel("file recall")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_recall.png"), dpi=150)
    except Exception as e:
        print("plot skipped", e)


if __name__ == "__main__":
    main()
