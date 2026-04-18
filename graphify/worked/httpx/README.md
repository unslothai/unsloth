# httpx Corpus Benchmark

A synthetic 6-file Python codebase modeled after httpx's architecture. Tests graphify on a realistic library with clean layering: exceptions → models → auth/transport → client.

## Corpus (6 files)

```
raw/
├── exceptions.py   — HTTPError hierarchy
├── models.py       — URL, Headers, Cookies, Request, Response
├── auth.py         — BasicAuth, BearerAuth, DigestAuth, NetRCAuth
├── utils.py        — header normalization, query params, content-type parsing
├── transport.py    — ConnectionPool, HTTPTransport, AsyncHTTPTransport, MockTransport
└── client.py       — Timeout, Limits, BaseClient, Client, AsyncClient
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

## What to expect

- 144 nodes, 330 edges, 6 communities
- God nodes: `Client`, `AsyncClient`, `Response`, `Request`, `BaseClient`, `HTTPTransport`
- Surprising connection: `DigestAuth` linked to `Response` — auth.py reads Response to parse WWW-Authenticate headers
- Token reduction: ~1x — 6 files fits in a context window, so there is no compression win here

The graph value on a small corpus is structural, not compressive: you can see the full dependency graph, identify god nodes, and understand architecture at a glance. Token reduction scales with corpus size — at 52 files (Karpathy benchmark) graphify achieves 71.5x.

Run `graphify benchmark worked/httpx/graph.json` to verify the numbers. Actual output is in this folder: `GRAPH_REPORT.md` and `graph.json`. Full eval: `review.md`.
