# Gemma 4 Developer Agent: localization study and agent submission

This repository holds a small, reproducible study of issue localization for small offline coding agents, and the agent configuration used for the Gemma 4 Developer Agent Competition on Kaggle.

## Contents

- `paper/writeup.md`: the paper (Paper Track writeup).
- `paper/study.py`: builds the benchmark and runs six retrieval strategies. Writes `results.json`, `results_table.md`, `fig_recall.png`.
- `paper/analysis.py`: bootstrap confidence intervals, paired tests, per-repository table. Writes `analysis.md`.
- `submission/`: the agent configuration (YAML and prompts) for the main track.
- `validate_submission.py`: offline checker that builds `submission.zip`.
- `colab/STATUS.md`: notes from an attempt to test the model on a T4 GPU.

## Reproduce the study

The study uses the competition's own data, which is not redistributed here. After accepting the competition rules, download these files with the Kaggle CLI into `paper/data/` (keep the folder names):

```bash
kaggle competitions download gemma-4-developer-agent -f tasks.jsonl -p paper/data
```

Download the graph and embedding files the same way, one per repository snapshot, into `paper/data/graphs/graphs/` and `paper/data/embeddings/embeddings/`. Then run:

```bash
pip install numpy matplotlib
python paper/study.py
python paper/analysis.py
```

Both scripts are deterministic and run on a CPU.

## License

MIT.
