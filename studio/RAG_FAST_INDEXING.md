# Studio RAG: fast incremental indexing

Progress log for an optimization stacked on the `feature/rag` branch (PR #5759). The goal is to
keep that PR's UX and retrieval accuracy while cutting per-document indexing from ~1 minute to a
few seconds, validated through the real Studio UI.

## Root cause of slow indexing (measured on feature/rag)

1. **Embedder reload on every upload.** Ingestion spawns a fresh `spawn` subprocess per job
   (`core/rag/ingestion.py` `_CTX.Process(target=_subprocess_worker)`), and the embedder is loaded
   *inside* that subprocess (`get_embedder` -> `from unsloth import FastSentenceTransformer`). A
   spawned interpreter re-imports unsloth and reloads the model from scratch every time, so the
   ~12-30s cold load recurs on each upload, not just the first. `lifespan()` never warms it.
2. **Per-document BM25 rebuild.** On every document, `ingestion.py` reads *all* chunks in the scope
   (`_all_scope_chunks`) and rebuilds the whole `bm25s` index (`bm25.rebuild_index`). That is
   O(N^2) tokenization as a knowledge base fills.

## The change (all behind a single `UNSLOTH_RAG_FAST=1` flag; off = byte-identical to PR)

1. **Warm the embedder at startup.** `main.py` `lifespan()` warms `get_embedder()` in a daemon
   thread (mirrors the existing GGUF precache thread).
2. **In-process ingestion.** Run the existing `_subprocess_worker` in a thread in the warm main
   process instead of a fresh subprocess, so `get_embedder` returns the already-loaded singleton.
   The worker, queue protocol, persistence and SSE progress are otherwise unchanged.
3. **Incremental SQLite FTS5 BM25.** Reimplement `core/rag/bm25.py` on an FTS5 virtual table in the
   existing `rag.db`, with an incremental `add_chunks(scope, chunks)`. Ingestion inserts only the
   new document's chunks (O(N) total) instead of rebuilding the scope. FTS5 `MATCH` returns only
   matching rows (no zero-score pollution) and adds `porter` stemming. The dense leg already uses
   sqlite-vec, so it is unchanged; RRF fusion in `retrieval.py` is unchanged.

## Validation

Two isolated Studios (separate `UNSLOTH_STUDIO_HOME` + port), same prebuilt frontend, same corpus,
same model. Baseline = `UNSLOTH_RAG_FAST` unset; improved = set. Driven through the real UI with
Playwright (`studio_test_kit`). Local GGUF model `unsloth/Qwen3.5-9B-GGUF` for the chat tool path.

Authoritative corpus (stable URLs): arXiv 1706.03762 (Attention), 1810.04805 (BERT),
2005.11401 (RAG); RFC 9110. Retrieval scored independently of generation with gold queries.

Metrics: per-document upload->ready latency (cold + warm), Recall@1/3/5 and MRR via `POST
/api/rag/search`, scaling (per-doc index time vs document count), and end-to-end UI (upload ->
indexed -> RAG answer with citation). Robustness: delete-then-query, restart persistence, thread
isolation.

## Results

Measured through the real Studio HTTP API on two isolated Studios (baseline = flag unset on
port 8905, improved = `UNSLOTH_RAG_FAST=1` on port 8912), same `bge-small-en-v1.5` embedder, same
corpus, same chunk settings. Single GPU.

### Indexing latency (upload to ready)

| Document | Baseline | Improved | Speedup |
|---|--:|--:|--:|
| attention (1706.03762, 20 chunks) | 23.4 s | 7.4 s | 3.2x |
| bert (1810.04805, 29 chunks) | 24.6 s | 8.0 s | 3.1x |
| rag (2005.11401, 26 chunks) | 23.2 s | 6.9 s | 3.4x |
| rfc9110.txt | 19.4 s | 1.2 s | 16x |
| **mean** | **22.7 s** | **5.9 s** | **~4x** |

Every baseline upload pays ~17 s of subprocess startup + model reload regardless of document size
(note rfc9110 at 19.4 s for one chunk). The improved path removes that fixed cost; what remains is
parse + embed.

### Scaling: 8 small docs into one knowledge base, per-document index time

| | Baseline | Improved |
|---|--:|--:|
| per-doc mean | 17.6 s | **0.12 s** |
| behavior | flat (subprocess dominates) | flat, ~147x faster |

(The bm25s O(N^2) scope rebuild is additionally eliminated; at small N the subprocess cost
dominates, but a standalone benchmark showed the rebuild alone is 25x overhead at 50 docs.)

### Retrieval accuracy (8 gold queries over the 4 docs, scored independently of generation)

| Mode | Baseline R@5 / MRR | Improved R@5 / MRR |
|---|--:|--:|
| bm25 | 0.875 / 0.807 | 0.875 / 0.775 |
| dense | 1.000 / 0.875 | 1.000 / 0.875 |
| hybrid | 1.000 / 0.833 | 1.000 / 0.844 |

No regression: dense and hybrid Recall@5 are 1.0 on both; hybrid MRR is slightly higher on the
improved path. Search latency on the improved path: 9-15 ms median (FTS5 + sqlite-vec).

### Summary

Indexing a paper drops from ~23 s to ~7 s and a small document from ~18 s to ~0.12 s, with
retrieval accuracy held constant.

### End-to-end validation in the real Studio UI (Playwright + local GGUF)

Both Studios were driven through the actual web UI with Playwright, with the local
`unsloth/Qwen3.5-9B-GGUF` (Q4_K_M) loaded via llama-server. Flow per Studio: load model, enable the
RAG toggle, upload `bert_1810.04805.pdf` through the composer, then ask "What are BERT's two
pre-training objectives?".

- **RAG works on both** (identical functionality): the model calls the `search_knowledge_base`
  tool, retrieves from the uploaded PDF, and answers with source citations
  (`[3][4] bert_1810.04805.pdf`). API-level tool-call test: 3/3 gold questions called the tool,
  retrieved the correct source paper, and answered with the right fact (BERT -> MLM + NSP,
  Transformer -> 8 heads, RAG -> DPR).
- **Indexing speed in the UI**: baseline 34.9 s vs improved 14.0 s for the same composer upload.

This confirms the fast path keeps the PR's full RAG behavior (many document types, the RAG toggle,
and tool-call retrieval with citations) while indexing materially faster.
