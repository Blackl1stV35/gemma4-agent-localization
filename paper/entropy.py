"""Information-theoretic retrieval variants vs BM25 on the 124-task localization benchmark.
Variants (all over symbol id + text tokens, query = issue text):
  bm25          baseline from study.py
  bits_binary   binary presence, weight = surprisal -log2(df/N) in bits, no length normalisation
  bits_sat      bits_binary with per-token saturation (sqrt of bits) so one rare token cannot dominate
  conc          bits weighted by token concentration 1 - H(files|token)/log2(#files)  (low-entropy tokens point to few files)
  cov           bits_binary x coverage (fraction of the query's total bits matched), rewards explaining more of the issue
  rrf_*         fusions with BM25
Writes paper/entropy.md. Needs paper/data and results.json."""
import json, os, sys, math, collections
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


class Info:
    def __init__(self, R):
        self.R = R
        docs = [set(study.split_ident(i) + study.tok(R.text[i])) for i in R.ids]
        self.inv = collections.defaultdict(list)
        for k, d in enumerate(docs):
            for t in d:
                self.inv[t].append(k)
        self.N = len(docs)
        modof = [".".join(i.split(".")[:-1]) if "." in i else i for i in R.ids]
        self.mod = modof
        self.bits, self.conc = {}, {}
        for t, ks in self.inv.items():
            df = len(ks)
            self.bits[t] = math.log2((self.N + 1) / (df + 0.5))
            c = collections.Counter(modof[k] for k in ks)
            if len(c) <= 1:
                self.conc[t] = 1.0
            else:
                p = np.array(list(c.values()), dtype=np.float64) / df
                H = -(p * np.log2(p)).sum()
                self.conc[t] = 1.0 - H / math.log2(len(c))

    def rank(self, q, mode):
        qt = set(study.split_ident(q) + study.tok(q))
        qt = [t for t in qt if t in self.inv]
        total = sum(self.bits[t] for t in qt) + 1e-9
        s = np.zeros(self.N)
        for t in qt:
            b = self.bits[t]
            if mode == "bits_sat":
                w = math.sqrt(b)
            elif mode == "conc":
                w = b * (0.25 + 0.75 * self.conc[t])
            else:
                w = b
            for k in self.inv[t]:
                s[k] += w
        if mode == "cov":
            matched = np.zeros(self.N)
            for t in qt:
                for k in self.inv[t]:
                    matched[k] += self.bits[t]
            s = s * (matched / total)
        return [self.R.ids[k] for k in np.argsort(-s)[:200] if s[k] > 0]


MODES = ["bits_binary", "bits_sat", "conc", "cov"]
names = ["lexical_bm25"] + MODES + ["rrf_bm25_bits", "rrf_bm25_conc", "rrf_bm25_cov", "rrf_bm25_graph_conc"]
res = collections.OrderedDict((k, []) for k in names)
cache = {}
for row in per:
    t = tasks[row["id"]]
    key = f"{t['repo'].split('/')[-1]}_{t['base_commit']}"
    if key not in cache:
        cache.clear()
        R = study.Repo(key)
        cache[key] = (R, Info(R))
    R, I = cache[key]
    gm = {f: study.best_mods(f, R.ids) for f in study.gold_files(t["patch"])}
    gm = {f: m for f, m in gm.items() if m}
    gold = set(gm)
    q = t["problem_statement"]
    lex = R.bm25(q)
    runs = {"lexical_bm25": lex}
    for m in MODES:
        runs[m] = I.rank(q, m)
    sd = R.seeds(q)
    g2 = R.graph_rank(sd) if sd else []
    runs["rrf_bm25_bits"] = study.rrf([lex, runs["bits_binary"]])
    runs["rrf_bm25_conc"] = study.rrf([lex, runs["conc"]])
    runs["rrf_bm25_cov"] = study.rrf([lex, runs["cov"]])
    runs["rrf_bm25_graph_conc"] = study.rrf([lex, g2, runs["conc"]])
    for m, nodes in runs.items():
        files = study.to_files(nodes, gm)
        res[m].append(rr(next((i + 1 for i, f in enumerate(files) if f in gold), None)))
M = {k: np.array(v) for k, v in res.items()}


def ci(v):
    b = v[idx].mean(axis=1)
    return f"{v.mean():.3f} [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]"


lines = [f"n = {n}", "", "| Method | MRR [95% CI] |", "|---|---|"] + [f"| {k} | {ci(v)} |" for k, v in M.items()]
lines += ["", "| Comparison | mean diff [95% CI] | P(diff<=0) |", "|---|---|---|"]
for a in names[1:]:
    d = M[a] - M["lexical_bm25"]
    bd = d[idx].mean(axis=1)
    lines.append(f"| {a} minus lexical_bm25 | {d.mean():+.3f} [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}] | {(bd<=0).mean():.3f} |")
open(os.path.join(HERE, "entropy.md"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
