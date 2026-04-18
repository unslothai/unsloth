# Mixed Corpus Benchmark

A small mixed-input corpus: Python source files, a markdown paper with arXiv citations, and one image. Tests graphify on different file types in a single run.

## Corpus (5 files)

```
raw/
├── analyze.py          — graph analysis module (god_nodes, surprising_connections)
├── build.py            — graph builder (build_from_json, NetworkX wrapper)
├── cluster.py          — Leiden community detection (cluster, score_all)
├── attention_notes.md  — Transformer paper notes (Vaswani et al., 2017) with arXiv citation
```

Note: the original benchmark included `attention_arabic.png` (an Arabic-language figure from the Attention paper). PNG files are not stored in this repo. To reproduce with the image, save any diagram from the Attention Is All You Need paper as `raw/attention_arabic.png`.

## How to run

```bash
pip install graphifyy

graphify install                        # Claude Code
graphify install --platform codex       # Codex
graphify install --platform opencode    # OpenCode
graphify install --platform claw        # OpenClaw
```

Then open your AI coding assistant in this directory and type:

```
/graphify ./raw
```

## What to expect

- ~20 nodes, ~19 edges from AST alone (3 Python modules)
- 3 communities: Graph Analysis, Clustering and Scoring, Graph Building
- God nodes: `analyze.py`, `cluster.py`, `build.py`
- `attention_notes.md` classified as `paper` (arXiv heuristic fires on `1706.03762`)
- If you include the image: 1 extra node describing the figure content via vision
- Token reduction: 5.4x

Actual output is in this folder: `GRAPH_REPORT.md` and `graph.json`. Full eval: `review.md`.
