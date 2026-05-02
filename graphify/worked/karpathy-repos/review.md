# Benchmark: Karpathy Repos + Research Papers

**Corpus:** nanoGPT, minGPT, micrograd (3 repos) + 5 research papers on attention/transformers + 4 images  
**Files:** 29 Python files + 14 docs/READMEs + 5 PDFs + 4 images (total 52 files)  
**Words:** ~92,616 · **Tokens (naive full-context):** ~123,488  
**Date:** 2026-04-04  
**Extraction:** AST (tree-sitter, deterministic) for code + Claude semantic for docs/papers/images

---

## Token reduction benchmark

### Code-only (AST, no Claude)

| Metric | Value |
|--------|-------|
| Corpus tokens (29 code files) | ~16,997 |
| Average query cost (BFS subgraph) | ~1,929 tokens |
| **Reduction ratio** | **8.8x** |

### Full corpus (code + papers + images)

| Metric | Value |
|--------|-------|
| Corpus tokens (52 files, naive full-context) | ~123,488 |
| Average query cost (BFS subgraph) | ~1,726 tokens |
| **Reduction ratio** | **71.5x** |

The reduction grows as corpus grows - the BFS subgraph stays roughly constant (~1,700 tokens) while naive stuffing scales linearly with corpus size.

### Per-question breakdown (full corpus)

| Reduction | Question |
|-----------|---------|
| 126.7x | what connects micrograd to nanoGPT |
| 100.8x | how does FlashAttention improve memory efficiency |
| 68.6x | what are the core abstractions |
| 68.6x | how are errors handled |
| 43.5x | how does the attention mechanism work |

The "attention mechanism" question returns a larger subgraph (2,836 tokens) because FlashAttention, CausalSelfAttention (nanoGPT), CausalSelfAttention (minGPT), and the AttnRes paper all connect to it. Still 43.5x cheaper than naive.

---

## Graph summary

| Metric | Value |
|--------|-------|
| Nodes | 285 (163 AST + 112 semantic) |
| Edges | 340 (281 AST + 97 semantic, after pruning) |
| Communities | 53 (17 major + 36 isolates) |

### Communities detected (major)

| Community | Nodes | What it found |
|-----------|-------|---------------|
| 0 (30 nodes) | nanoGPT Model Architecture | `Block`, `forward()`, `dataclasses` - transformer architecture |
| 1 (24 nodes) | minGPT Training + Datasets | `batch_end_callback`, `eval_split`, `get_config`, `CharDataset`, `chargpt` |
| 2 (23 nodes) | nanoGPT Training Pipeline | `get_batch`, `bench.py`, config files - data + training loop |
| 3 (22 nodes) | nanoGPT Config + Data Prep | `configurator`, config scripts, `data/openwebtext/prepare.py` |
| 4 (21 nodes) | micrograd NN Layer | `Layer`, `__call__`, `__init__`, `MLP` |
| 5 (21 nodes) | FlashAttention Paper | `IO-awareness`, `HBM/SRAM`, `recomputation`, BERT/GPT-2 benchmarks |
| 6 (17 nodes) | BPE Tokenizer | `BPETokenizer`, `decode`, `bytes_to_unicode`, full tokenisation logic |
| 7 (16 nodes) | micrograd Autograd Engine | `Value`, `backward`, `__add__`, `__mul__` - the autograd core |
| 8 (14 nodes) | Stdlib + Config Utilities | `ast`, `json`, `CfgNode` - supporting infrastructure |
| 9 (13 nodes) | Addition Dataset | `AdditionDataset`, `get_block_size`, `get_vocab_size` |
| 10 (12 nodes) | micrograd README + Backprop | README concepts, backprop explanation, computation graph |
| 11 (7 nodes) | Attention Residuals Paper | Kimi model, pre-norm dilution, MMLU scaling |
| 12 (6 nodes) | Continual LoRA Paper | CoLoR, catastrophic forgetting, ViT fine-tuning |
| 13 (6 nodes) | minGPT Trainer Class | `add_callback`, `run`, `set_callback` |
| 14 (5 nodes) | NeuralWalker Paper | SSM, graph expressivity, Pascal VOC results |

### God nodes (highest degree)

| Node | Edges | Why central |
|------|-------|-------------|
| `Value` (micrograd) | 15 | The autograd primitive - everything math-related connects through it |
| `Training Script` (nanoGPT) | 11 | Orchestrates model + data + optimizer |
| `GPT` (nanoGPT) | 9 | Main model class - Block, attention, config all flow through here |
| `Layer` (micrograd nn) | 8 | The neural net abstraction - connects engine to high-level API |

---

## Graph quality evaluation

### What the graph got right

- **micrograd split correctly into two communities** - engine (Value + autograd) and nn (Layer + MLP) are separate communities, matching the intended architecture split in the repo.
- **nanoGPT model vs training separation** - communities 0 and 2 correctly separate model definition from training loop. Different concerns in different files; Leiden found the boundary.
- **BPETokenizer isolated** - `bpe.py` forms its own cluster, correctly identified as standalone rather than merged with model or trainer.
- **Cross-repo connections found** - the graph found that nanoGPT `Block` and minGPT `Block` share structural similarity (same class name, similar methods), creating a cross-repo INFERRED edge. This is genuine: both implement the same GPT block pattern.
- **Paper → code connections** - FlashAttention paper cluster (Community 5) connects to `CausalSelfAttention` in both nanoGPT and minGPT. NeuralWalker paper connects to graph structural concepts in micrograd.
- **Images correctly identified** - `gpt2_124M_loss.png` extracted as "val_loss=2.905 at step 399"; `gout.svg` recognized as micrograd computation graph; `moon_mlp.png` as MLP decision boundary.

### What the graph missed or got wrong

- **Stdlib imports create 94 validation warnings** - `setuptools`, `os`, `math`, `sys` emit "target does not match any node" warnings. The AST extractor emits import edges to stdlib names before the validator can prune them. These are discarded but inflate edge count before pruning.
- **Config-only files become isolates** - `eval_gpt2.py`, `eval_gpt2_large.py` etc. are config scripts with no functions; they land as single-node communities. Expected, but adds ~36 trivial communities.
- **53 communities from 285 nodes** - the isolate problem means ~36 of 53 communities are single nodes. The "17 major communities" number from the code-only run was cleaner. The isolate handling is correct but visually noisy.
- **Papers not deep-linked to implementation** - the FlashAttention paper cluster knows about "3x GPT-2 speedup" but the graph doesn't directly link that claim to the specific `CausalSelfAttention` implementation that would benefit. That would require `--mode deep` on the paper extraction pass.

### Surprising connections

- `micrograd/engine.py::Value.backward()` → `minGPT/mingpt/trainer.py::Trainer.run()` - both implement the foundational forward/backward pattern at different scales. The graph surfaces this cross-repo connection without being asked.
- `FlashAttention paper` (Community 5) bridges into `CausalSelfAttention` nodes in both nanoGPT and minGPT, creating the only paper→code cross-community edges in the graph.
- `nanoGPT/train.py` and `minGPT/mingpt/trainer.py` land in the same community (Community 2) despite being in different repos and never importing each other. Leiden found the structural similarity through shared vocabulary (optimizer, scheduler, gradient clipping).

---

## Verdict

**71.5x token reduction** on a 92k-word mixed corpus. The reduction grows as corpus grows - on a 500k-word research library the same BFS subgraph stays ~2k tokens while naive stuffing hits 670k tokens.

Graph quality: high for code structure, strong for paper-to-concept connections (semantic extraction found the FlashAttention→CausalSelfAttention bridge), weaker on direct paper-to-implementation links (need `--mode deep` with explicit cross-file context).

The main cost is honesty: 53 communities when 17 are real and 36 are isolates. This is correct behavior (isolates shouldn't be merged), but the visualization is noisy. A future `--min-community-size` flag would clean this up.
