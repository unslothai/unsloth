# Reproducible Example

A small document pipeline — parser, validator, processor, storage, API — with architecture notes and research notes. Seven files, two languages, clear call relationships between modules.

Run graphify on it and you get a knowledge graph showing how the modules connect, which functions call which, and how the architecture notes relate to the code.

## Input files

```
raw/
├── parser.py        — reads files, detects format, kicks off the pipeline
├── validator.py     — schema checks, calls processor for text normalization
├── processor.py     — keyword extraction, cross-reference detection
├── storage.py       — persists everything, maintains the index
├── api.py           — HTTP handlers that orchestrate the above four modules
├── architecture.md  — design decisions and module responsibilities
└── notes.md         — open questions and tradeoffs
```

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

No PDF or image extraction — runs entirely on AST and markdown with no token cost for semantic extraction.

## What to expect

- `api.py` as a hub node connected to all four modules
- `storage.py` as the highest-degree god node (everything reads and writes through it)
- `parser.py` calling `validator.py` and `storage.py`
- `architecture.md` and `notes.md` linked to the code modules they discuss
- 2 communities: the four Python modules together, the two markdown files together (or api.py in its own cluster given high connectivity)

## After it runs

Ask questions from your AI coding assistant:

- "what calls storage directly?"
- "what is the shortest path between parser and processor?"
- "which module has the most connections?"
- "what does the architecture doc say about the storage design?"

The graph lives in `graphify-out/` and persists across sessions.
