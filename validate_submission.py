"""Offline structural check of a submission directory against the harness rules, then builds submission.zip."""
import sys, zipfile, pathlib, yaml

ALLOWED_MODELS = {"gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                  "gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct"}
TOOLS = {"run_command", "submit_patch", "get_status", "read_file", "edit_file", "write_file",
         "get_code_neighbors", "search_similar_code", "get_code_subgraph"}
EXT = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX = 3 * 1024**3

class Loader(yaml.SafeLoader): pass
def _inc(loader, node):
    rel = loader.construct_scalar(node)
    assert not rel.startswith("/") and ".." not in pathlib.PurePosixPath(rel).parts, f"bad include {rel}"
    p = ROOT / rel
    assert p.is_file(), f"missing include {rel}"
    return p.read_text(encoding="utf-8") if p.suffix in (".md", ".txt") else yaml.load(p.read_text(encoding="utf-8"), Loader)
Loader.add_constructor("!include", _inc)

def agents(cfg, out):
    out.append(cfg)
    for t in cfg.get("tools", []):
        if isinstance(t, dict) and "agent_tool" in t:
            agents(yaml.load((ROOT / t["agent_tool"]["config_path"]).read_text(encoding="utf-8"), Loader), out)
    for s in cfg.get("sub_agents", []):
        agents(yaml.load((ROOT / s["config_path"]).read_text(encoding="utf-8"), Loader), out)
    return out

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "submission").resolve()
errs = []
roots = [n for n in ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml") if (ROOT / n).exists()]
if len(roots) != 1: errs.append(f"need exactly one root config, found {roots}")
files = [p for p in ROOT.rglob("*") if p.is_file()]
for p in files:
    if p.is_symlink(): errs.append(f"symlink {p}")
    if p.suffix not in EXT: errs.append(f"extension not allowed: {p.relative_to(ROOT)}")
if sum(p.stat().st_size for p in files) >= MAX: errs.append("over 3 GiB")
if not errs:
    al = agents(yaml.load((ROOT / roots[0]).read_text(encoding="utf-8"), Loader), [])
    models = {a.get("model") for a in al}
    if len(models) != 1 or not models <= ALLOWED_MODELS: errs.append(f"model rule violated: {models}")
    for a in al:
        for t in a.get("tools", []):
            if isinstance(t, str) and t not in TOOLS: errs.append(f"unknown tool {t} in {a['name']}")
        g = a.get("generate_content_config", {})
        if g.get("max_output_tokens", 1) > 32768: errs.append("max_output_tokens too large")
        if g.get("thinking_config", {}).get("thinking_budget", 0) > 32768: errs.append("thinking_budget too large")
        ad = a.get("adapter")
        if ad and not (ROOT / "adapters" / ad).exists(): errs.append(f"adapter {ad} missing")
if errs:
    print("INVALID"); [print(" -", e) for e in errs]; sys.exit(1)
out = ROOT.parent / "submission.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files): z.write(p, p.relative_to(ROOT).as_posix())
print(f"OK: {len(files)} files, {len(al)} agents, model {models}, wrote {out} ({out.stat().st_size} bytes)")
