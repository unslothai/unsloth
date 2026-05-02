# Graph Report - /home/safi/graphify-benchmark  (2026-04-04)

## Corpus Check
- 49 files · ~92,616 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 285 nodes · 340 edges · 53 communities detected
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS
- Token cost: 6,000 input · 3,500 output

## God Nodes (most connected - your core abstractions)
1. `Value` - 15 edges
2. `Training Script` - 11 edges
3. `GPT` - 9 edges
4. `Layer` - 8 edges
5. `CharDataset` - 7 edges
6. `AdditionDataset` - 7 edges
7. `CfgNode` - 7 edges
8. `Encoder` - 7 edges
9. `Neuron` - 7 edges
10. `FlashAttention Algorithm` - 7 edges

## Surprising Connections (you probably didn't know these)
- `from_pretrained()` --calls--> `get_default_config()`  [INFERRED]
  /home/safi/graphify-benchmark/repos/nanoGPT/model.py → /home/safi/graphify-benchmark/repos/minGPT/mingpt/model.py
- `get_batch()` --conceptually_related_to--> `get_batch()`  [INFERRED]
  /home/safi/graphify-benchmark/repos/nanoGPT/train.py → /home/safi/graphify-benchmark/repos/nanoGPT/bench.py
- `Training Script` --produces--> `GPTConfig Dataclass`  [INFERRED]
  repos/nanoGPT/train.py → repos/nanoGPT/model.py
- `GPT Language Model (minGPT)` --conceptually_related_to--> `GPT Model Class`  [INFERRED]
  repos/minGPT/mingpt/model.py → repos/nanoGPT/model.py
- `CausalSelfAttention (minGPT)` --conceptually_related_to--> `CausalSelfAttention Module`  [INFERRED]
  repos/minGPT/mingpt/model.py → repos/nanoGPT/model.py

## Communities

### Community 0 - "nanoGPT Model Architecture"
Cohesion: 0.11
Nodes (12): dataclasses, inspect, Block, CausalSelfAttention, from_pretrained(), get_default_config(), GPT, GPTConfig (+4 more)

### Community 1 - "minGPT Training + Datasets"
Cohesion: 0.12
Nodes (17): batch_end_callback(), eval_split(), get_config(), get_default_config(), get_config(), get_default_config(), collections, mingpt_bpe (+9 more)

### Community 2 - "nanoGPT Training Pipeline"
Cohesion: 0.13
Nodes (15): get_batch(), contextlib, datasets, math, numpy, os, pickle, tiktoken (+7 more)

### Community 3 - "nanoGPT Config + Data Prep"
Cohesion: 0.1
Nodes (22): Benchmarking Script, Config: Finetune GPT-2-XL on Shakespeare, Config: Train GPT-2 (124M), Config: Train Character-Level Shakespeare, Configurator (exec-based Override System), OpenWebText Data Preparation, Shakespeare Char-Level Data Preparation, Shakespeare (BPE) Data Preparation (+14 more)

### Community 4 - "micrograd NN Layer"
Cohesion: 0.13
Nodes (6): micrograd_engine, Layer, MLP, Module, Neuron, random

### Community 5 - "FlashAttention Paper"
Cohesion: 0.12
Nodes (21): FlashAttention Algorithm, GPU HBM vs On-Chip SRAM Memory Hierarchy, FlashAttention: Fast Memory-Efficient Attention, Selective Gradient Checkpointing (Recomputation), Result: 15% faster BERT-large vs MLPerf, Result: 3x GPT-2 training speedup, Tiling for Attention Computation, Self-Attention Mechanism (Q, K, V) (+13 more)

### Community 6 - "BPE Tokenizer"
Cohesion: 0.19
Nodes (8): BPETokenizer, bytes_to_unicode(), Encoder, get_encoder(), get_file(), get_pairs(), regex, requests

### Community 7 - "micrograd Autograd Engine"
Cohesion: 0.12
Nodes (1): Value

### Community 8 - "Stdlib + Config Utilities"
Cohesion: 0.18
Nodes (5): ast, json, sys, CfgNode, setup_logging()

### Community 9 - "Addition Dataset"
Cohesion: 0.15
Nodes (3): AdditionDataset, CharDataset, Dataset

### Community 10 - "micrograd README + Backprop"
Cohesion: 0.21
Nodes (11): Value (autograd scalar), Value.backward, Micrograd Computation Graph (operations + gradients), Backpropagation / Reverse-Mode Autodiff, Dynamically Built DAG (computation graph), micrograd, GPT.configure_optimizers, GPT.forward (minGPT) (+3 more)

### Community 11 - "Attention Residuals Paper"
Cohesion: 0.33
Nodes (7): Block Attention Residuals, Full Attention Residuals, Attention Residuals (AttnRes) - Kimi Team, PreNorm Dilution Problem, Result: AttnRes improves MMLU 73.5→74.6, BBH 76.3→78.0, Result: Block AttnRes matches 1.25x more compute baseline, Residual Connections in Deep Networks

### Community 12 - "Continual LoRA Paper"
Cohesion: 0.33
Nodes (6): Catastrophic Forgetting Problem, CoLoR Method, Low Rank Adaptation (LoRA), CoLoR: Continual Learning with Low Rank Adaptation, Vision Transformer (ViT-B-16) Backbone, Multi-Head Attention

### Community 13 - "minGPT Trainer Class"
Cohesion: 0.4
Nodes (1): Trainer

### Community 14 - "NeuralWalker Paper"
Cohesion: 0.4
Nodes (5): Mamba State Space Model, NeuralWalker Architecture, NeuralWalker: Learning Long Range Dependencies on Graphs, Result: NeuralWalker is strictly more expressive than 1-WL, Result: NeuralWalker +10% PascalVOC-SP, +13% COCO-SP over SOTA

### Community 15 - "Dataset Abstractions"
Cohesion: 0.67
Nodes (3): AdditionDataset, CharDataset, GPT.generate (minGPT)

### Community 16 - "BPETokenizer (minGPT)"
Cohesion: 1.0
Nodes (2): BPETokenizer, BPE Encoder

### Community 17 - "OpenWebText Dataset"
Cohesion: 1.0
Nodes (2): OpenWebText Dataset, OpenWebText Dataset (~9B tokens, 17GB, 8M documents)

### Community 18 - "torch.compile Performance"
Cohesion: 1.0
Nodes (2): Performance: torch.compile reduces iter time from 250ms to 135ms, torch.compile (PyTorch 2.0)

### Community 19 - "Behavior Token Paper"
Cohesion: 1.0
Nodes (2): Behavior Tokens Concept, LCBM: Large Content and Behavior Model

### Community 20 - "Setup"
Cohesion: 1.0
Nodes (1): setuptools

### Community 21 - "Nanogpt Complexity Metaphor"
Cohesion: 1.0
Nodes (2): GPT Complexity Metaphor: Battleship vs Speedboat, nanogpt_readme_design_simplicity

### Community 22 - "Mingpt Readme Design Education"
Cohesion: 1.0
Nodes (2): Design Decision: minGPT prioritizes education (~300 lines), Design Decision: nanoGPT prioritizes speed over education

### Community 23 - "Mingpt Readme Mingpt"
Cohesion: 1.0
Nodes (2): mingpt_readme_mingpt, Attention Is All You Need (Transformer Paper)

### Community 24 - "Init"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Train Gpt2"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Eval Gpt2 Xl"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Eval Gpt2"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Eval Gpt2 Large"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Train Shakespeare Char"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Eval Gpt2 Medium"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "Model Layernorm"
Cohesion: 1.0
Nodes (1): LayerNorm with Optional Bias

### Community 32 - "Model Meta Pkl Schema"
Cohesion: 1.0
Nodes (1): meta.pkl Vocabulary Schema

### Community 33 - "Config Eval Gpt2"
Cohesion: 1.0
Nodes (1): Config: Eval GPT-2 (124M)

### Community 34 - "Config Eval Gpt2 Medium"
Cohesion: 1.0
Nodes (1): Config: Eval GPT-2 Medium

### Community 35 - "Config Eval Gpt2 Large"
Cohesion: 1.0
Nodes (1): Config: Eval GPT-2 Large

### Community 36 - "Config Eval Gpt2 Xl"
Cohesion: 1.0
Nodes (1): Config: Eval GPT-2 XL

### Community 37 - "Mingpt Model Newgelu"
Cohesion: 1.0
Nodes (1): NewGELU Activation

### Community 38 - "Mingpt Model Gpt From Pretrained"
Cohesion: 1.0
Nodes (1): GPT.from_pretrained (minGPT)

### Community 39 - "Mingpt Trainer Trainer"
Cohesion: 1.0
Nodes (1): Trainer (minGPT)

### Community 40 - "Mingpt Utils Cfgnode"
Cohesion: 1.0
Nodes (1): CfgNode Configuration Class

### Community 41 - "Mingpt Utils Set Seed"
Cohesion: 1.0
Nodes (1): set_seed

### Community 42 - "Mingpt Utils Setup Logging"
Cohesion: 1.0
Nodes (1): setup_logging

### Community 43 - "Mingpt Bpe Get Encoder"
Cohesion: 1.0
Nodes (1): get_encoder

### Community 44 - "Mingpt Readme Gpt2 Arch Changes"
Cohesion: 1.0
Nodes (1): GPT-2 Architectural Changes: pre-norm LayerNorm, scaled residual init

### Community 45 - "Shakespeare Char Readme Char Dataset"
Cohesion: 1.0
Nodes (1): Tiny Shakespeare Char Dataset (1M train tokens)

### Community 46 - "Mingpt Readme Adder Project"
Cohesion: 1.0
Nodes (1): minGPT Adder Project (GPT trained to add numbers)

### Community 47 - "Chargpt Readme Tiny Shakespeare"
Cohesion: 1.0
Nodes (1): Tiny Shakespeare Dataset

### Community 48 - "2205 14135 Io Awareness"
Cohesion: 1.0
Nodes (1): IO-Aware Attention Computation

### Community 49 - "2205 14135 Result Memory Linear"
Cohesion: 1.0
Nodes (1): Result: FlashAttention memory scales linearly

### Community 50 - "2311 17601 Result Domainnet"
Cohesion: 1.0
Nodes (1): Result: CoLoR 69.7% on DomainNet (+19% over S-Prompts)

### Community 51 - "2309 00359 Result Behavior Sim"
Cohesion: 1.0
Nodes (1): Result: LCBM outperforms GPT-3.5/4 on behavior simulation (10x smaller)

### Community 52 - "Concept Positional Encoding"
Cohesion: 1.0
Nodes (1): Positional Encoding in Transformers

## Knowledge Gaps
- **65 isolated node(s):** `MLP Module`, `LayerNorm with Optional Bias`, `Checkpoint Data Schema (ckpt.pt)`, `meta.pkl Vocabulary Schema`, `Sampling/Inference Script` (+60 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `BPETokenizer (minGPT)`** (2 nodes): `BPETokenizer`, `BPE Encoder`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `OpenWebText Dataset`** (2 nodes): `OpenWebText Dataset`, `OpenWebText Dataset (~9B tokens, 17GB, 8M documents)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `torch.compile Performance`** (2 nodes): `Performance: torch.compile reduces iter time from 250ms to 135ms`, `torch.compile (PyTorch 2.0)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Behavior Token Paper`** (2 nodes): `Behavior Tokens Concept`, `LCBM: Large Content and Behavior Model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Setup`** (2 nodes): `setup.py`, `setuptools`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Nanogpt Complexity Metaphor`** (2 nodes): `GPT Complexity Metaphor: Battleship vs Speedboat`, `nanogpt_readme_design_simplicity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Readme Design Education`** (2 nodes): `Design Decision: minGPT prioritizes education (~300 lines)`, `Design Decision: nanoGPT prioritizes speed over education`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Readme Mingpt`** (2 nodes): `mingpt_readme_mingpt`, `Attention Is All You Need (Transformer Paper)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Train Gpt2`** (1 nodes): `train_gpt2.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Eval Gpt2 Xl`** (1 nodes): `eval_gpt2_xl.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Eval Gpt2`** (1 nodes): `eval_gpt2.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Eval Gpt2 Large`** (1 nodes): `eval_gpt2_large.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Train Shakespeare Char`** (1 nodes): `train_shakespeare_char.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Eval Gpt2 Medium`** (1 nodes): `eval_gpt2_medium.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Model Layernorm`** (1 nodes): `LayerNorm with Optional Bias`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Model Meta Pkl Schema`** (1 nodes): `meta.pkl Vocabulary Schema`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Eval Gpt2`** (1 nodes): `Config: Eval GPT-2 (124M)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Eval Gpt2 Medium`** (1 nodes): `Config: Eval GPT-2 Medium`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Eval Gpt2 Large`** (1 nodes): `Config: Eval GPT-2 Large`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Eval Gpt2 Xl`** (1 nodes): `Config: Eval GPT-2 XL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Model Newgelu`** (1 nodes): `NewGELU Activation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Model Gpt From Pretrained`** (1 nodes): `GPT.from_pretrained (minGPT)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Trainer Trainer`** (1 nodes): `Trainer (minGPT)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Utils Cfgnode`** (1 nodes): `CfgNode Configuration Class`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Utils Set Seed`** (1 nodes): `set_seed`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Utils Setup Logging`** (1 nodes): `setup_logging`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Bpe Get Encoder`** (1 nodes): `get_encoder`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Readme Gpt2 Arch Changes`** (1 nodes): `GPT-2 Architectural Changes: pre-norm LayerNorm, scaled residual init`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Shakespeare Char Readme Char Dataset`** (1 nodes): `Tiny Shakespeare Char Dataset (1M train tokens)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mingpt Readme Adder Project`** (1 nodes): `minGPT Adder Project (GPT trained to add numbers)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Chargpt Readme Tiny Shakespeare`** (1 nodes): `Tiny Shakespeare Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `2205 14135 Io Awareness`** (1 nodes): `IO-Aware Attention Computation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `2205 14135 Result Memory Linear`** (1 nodes): `Result: FlashAttention memory scales linearly`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `2311 17601 Result Domainnet`** (1 nodes): `Result: CoLoR 69.7% on DomainNet (+19% over S-Prompts)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `2309 00359 Result Behavior Sim`** (1 nodes): `Result: LCBM outperforms GPT-3.5/4 on behavior simulation (10x smaller)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Concept Positional Encoding`** (1 nodes): `Positional Encoding in Transformers`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Training Script` connect `nanoGPT Config + Data Prep` to `nanoGPT Training Pipeline`?**
  _High betweenness centrality (0.176) - this node is a cross-community bridge._
- **Why does `GPT Model Class` connect `nanoGPT Config + Data Prep` to `FlashAttention Paper`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `estimate_loss()` connect `nanoGPT Training Pipeline` to `nanoGPT Config + Data Prep`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Value` (e.g. with `.__add__()` and `.__mul__()`) actually correct?**
  _`Value` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Training Script` (e.g. with `GPTConfig Dataclass` and `Performance: ~2.85 val loss in 4 days on 8xA100`) actually correct?**
  _`Training Script` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Layer` (e.g. with `.__init__()` and `.__call__()`) actually correct?**
  _`Layer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `MLP Module`, `LayerNorm with Optional Bias`, `Checkpoint Data Schema (ckpt.pt)` to the rest of the system?**
  _65 weakly-connected nodes found - possible documentation gaps or missing edges._