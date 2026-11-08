"""Build paper/writeup.pdf from paper/writeup.md using python-markdown and headless Chrome (no color, Times New Roman)."""
import os, subprocess, sys, markdown

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = "Lexical Search Is Hard to Beat for Small Offline Coding Agents"
SUB = "Seeded graph and embedding signals add little over BM25 for locating issue files in the competition's own code graphs."
extra = sys.argv[1] if len(sys.argv) > 1 else ""  # optional note printed under the title, such as a submission label
md = open(os.path.join(HERE, "writeup.md"), encoding="utf8").read()
md = md[md.index("## Abstract"):]  # the title and subtitle are printed from constants above
html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_R = json.load(open(os.path.join(HERE, "results.json"), encoding="utf8"))["summary"]
_ms = ["lexical_bm25", "seed_only", "graph_2hop", "embedding_seed", "hybrid_rrf", "hybrid_bm25_graph"]
_ks = [1, 5, 10, 25]
_fig, _ax = plt.subplots(figsize=(7, 4))
for _i, _m in enumerate(_ms):
    _ax.bar(np.arange(4) + _i * 0.13, [_R[_m][f"r@{k}"] for k in _ks], 0.13, label=_m, color=str(0.15 + 0.14 * _i), hatch=["", "//", "..", "xx", "--", "\\\\"][_i], edgecolor="black")
_ax.set_xticks(np.arange(4) + 0.325)
_ax.set_xticklabels([f"R@{k}" for k in _ks])
_ax.set_ylabel("file recall")
_ax.legend(fontsize=7)
_fig.tight_layout()
_fig.savefig(os.path.join(HERE, "fig_gray.png"), dpi=150)
fig = os.path.join(HERE, "fig_gray.png")
if os.path.exists(fig):
    marker = "The intervals overlap heavily.</p>"
    html_body = html_body.replace(marker, marker + '<p class="fig"><img src="fig_gray.png" width="520"><br><em>Figure 1. File recall at k for each retrieval signal (124 tasks).</em></p>', 1)
css = """
@page { size: A4; margin: 22mm 20mm; }
body { font-family: 'Times New Roman', Times, serif; color: #000; font-size: 11.5pt; line-height: 1.35; }
h1 { font-size: 18pt; text-align: center; margin: 0 0 4pt 0; }
.sub { text-align: center; font-style: italic; margin-bottom: 4pt; }
.note { text-align: center; font-size: 10pt; margin-bottom: 14pt; }
h2 { font-size: 14pt; margin: 16pt 0 4pt 0; }
h3 { font-size: 12pt; margin: 12pt 0 3pt 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #000; padding: 2pt 4pt; text-align: left; }
code { font-family: 'Courier New', monospace; font-size: 10pt; }
.fig { text-align: center; }
p, li { text-align: justify; }
"""
note = f'<div class="note">{extra}</div>' if extra else ""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body><h1>{TITLE}</h1><div class='sub'>{SUB}</div>{note}{html_body}</body></html>"
hp = os.path.join(HERE, "writeup.html")
open(hp, "w", encoding="utf8").write(html)
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
out = os.path.join(HERE, "writeup.pdf")
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", "file:///" + hp.replace("\\", "/")], check=True, timeout=120)
print(out, os.path.getsize(out))
