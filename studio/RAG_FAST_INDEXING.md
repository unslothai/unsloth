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

Pending - populated as runs complete (see commits below).
