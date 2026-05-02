# Phase 3 RAG Updates

## Scope
This document summarizes the backend RAG changes made to unblock end-to-end wiki retrieval for GGUF chat, and adds practical debugging and cleanup workflows.

## Incremental Update (2026-04-30, diagnostics + logging)

- Added rank-mode logging parity for transformer/safetensors RAG path (`studio/backend/core/inference/inference.py`), so logs now include `rank_mode`, selected pages, context chars, and query similar to GGUF route diagnostics.
- Wiki `log.md` headings now include UTC hour:minute (`YYYY-MM-DD HH:MM`) instead of date-only headings.
- Wiki behaviour UI parity audit completed: runtime env coverage is complete (68 variables, no uncategorized variables), and category rendering now preserves backend runtime-spec order within each category.
- Chunked-ingest robustness improved for large sources: default chunk context window is now capped by extraction/query limits, propagated into per-chunk query context overrides, and auto-replanned to a smaller window when a large source still yields only one chunk.
- Low-quality answer gating now catches shorter repeated-token degeneration runs earlier, reducing cases where repetitive gibberish slips through as `Answer Mode: llm`.
- Added explicit adaptive chunk replan diagnostics: runtime marker (`WIKI_CHUNK_REPLAN ...`) and `wiki/log.md` fields now show whether adaptive replan was applied and the window/chunk-count before→after values.
- Wiki data dialog now provides explicit top-row hide/show controls for source, entity, and concept nodes (alongside analysis visibility).
- Chunked wiki ingestion now removes intermediate `analysis/*--chunk-###-of-###--analysis` pages after successful merge, while preserving the merged analysis page and source/chunk source pages.
- Chunk-merge outputs now enforce watcher-style `Section A` through `Section I` structure; non-conforming merge LLM output automatically falls back to a sectioned extractive merge answer.
- Chunk-merge filenames are now content-aware and unique (`...--chunk-merged--<topic>`), instead of a single generic `--chunk-merged-analysis` slug.
- Watcher auto-analysis now skips additional source-first query generation when the same ingest already ran in chunked mode and produced a merged analysis page, preventing duplicate extra analysis files.

## Incremental Update (2026-04-28)

This incremental update hardens wiki maintenance quality in three areas:

- semantic relevance for analysis-page Enrichment links
- semantic relevance for analysis Context Pages selection
- stable index summaries for resolved fallback pages

### A) Enrichment link selection is semantic-first

Updated in `studio/backend/core/wiki/engine.py` and `studio/backend/tests/test_wiki_rag_pipeline.py`:

- Enrichment now treats LLM selection as authoritative when valid JSON is returned, including explicit empty selections.
- Lexical selection is now used only when LLM output is invalid/unparseable schema.
- Candidate snippets in the selector prompt now include compact page summaries to improve semantic picks.
- Enrichment section strategy text now reflects semantic LLM selector mode when enabled.

### B) Context Pages ranking is semantic-first when LLM rerank is enabled

Updated in `studio/backend/core/wiki/engine.py` and `studio/backend/tests/test_wiki_rag_pipeline.py`:

- LLM rerank candidate pool and top-N now scale with context-page needs so rerank output can directly drive actual context selection.
- Deterministic tail appending after LLM rerank was removed to avoid unrelated pages leaking into `## Context Pages`.
- Added regression coverage to ensure reranked output is not padded with non-semantic tail pages.

### C) Duplicate external source summarization prevention across maintenance runs

Updated in `studio/backend/core/wiki/engine.py` and `studio/backend/tests/test_wiki_rag_pipeline.py`:

- Web gap-fill now normalizes source URLs and reuses existing source pages by `source_ref`.
- If an existing source-first summary page already exists for that source, maintenance reuses it instead of creating a new `-2` analysis page.
- Report counters now distinguish reused vs newly ingested/created external artifacts.

### D) Index analysis summary title hardening

Updated in `studio/backend/core/wiki/engine.py` and `studio/backend/tests/test_wiki_rag_pipeline.py`:

- Index summary extraction now rejects source-first template placeholder titles such as `Section A: Brief summary paragraph`.
- When placeholder titles are detected, index summary falls back to informative source title from the question (and keeps primary source annotation and fallback-resolved tags).

### Runtime/UI parity for enrichment selector controls

Updated in `studio/backend/core/wiki/runtime_env.py` and `studio/frontend/src/components/wiki-behaviour-dialog.tsx`:

- Added runtime variables for enrichment LLM selector enablement and candidate cap.
- Updated wiki behaviour dialog expected env count and category mapping.

## Changed Files
- `studio/backend/core/wiki/ingestor.py`
- `studio/backend/core/wiki/engine.py`
- `studio/backend/core/wiki/manager.py`
- `studio/backend/core/wiki/watcher.py`
- `studio/backend/core/inference/inference.py`
- `studio/backend/main.py`
- `studio/backend/run.py`
- `studio/backend/routes/inference.py`
- `studio/backend/models/inference.py`
- `studio/backend/tests/test_wiki_rag_pipeline.py`
- `updates.md`

## New/Updated Behaviors

### 1) RAG context debug endpoints
Added in `studio/backend/routes/inference.py`:
- `POST /api/inference/rag/debug/context`
  - Computes context for an arbitrary query.
  - Can optionally ingest pending files from `UNSLOTH_WIKI_VAULT/raw` first.
  - Returns selected pages, snippets, budgets, and full injected context text.
- `GET /api/inference/rag/debug/last`
  - Returns the exact last context selected by the GGUF `/chat/completions` path.
  - Useful for explaining model answers and prompt injection state.

Schemas are in `studio/backend/models/inference.py`:
- `RagContextDebugRequest`
- `RagContextDebugResponse`
- `RagContextSnippet`

### 2) Stale wiki archival endpoint
Added in `studio/backend/routes/inference.py`:
- `POST /api/inference/wiki/archive/stale`
  - Archives stale source pages under `wiki/.archive/sources`.
  - Also archives corresponding raw files under `raw/.archive` when available.
  - Supports `dry_run` mode.
  - Keeps the latest `keep_recent_per_source` source pages per canonical source and latest `keep_recent_chat` chat pages.

Schemas are in `studio/backend/models/inference.py`:
- `WikiArchiveRequest`
- `WikiArchiveResponse`

### 2b) Explicit ingest/query/lint endpoints
Added in `studio/backend/routes/inference.py`:
- `POST /api/inference/wiki/ingest`
  - Ingest a specific file path, or ingest pending files from `raw/`.
- `POST /api/inference/wiki/retry-fallback`
  - Re-runs questions for analysis pages that were previously marked as fallback.
  - Can run in `dry_run` mode to preview retry outcomes without writing.
- `POST /api/inference/wiki/query`
  - Query wiki pages and optionally save answer pages into `wiki/analysis`.
  - Requires an active loaded model for synthesis.
- `GET /api/inference/wiki/lint`
  - Runs health checks and returns:
    - orphan pages
    - stale pages
    - broken links
    - missing concept-page candidates
    - low-coverage source pages

Maintenance cadence note:
- Automatic enrichment now runs on the same cadence as automatic lint (`UNSLOTH_WIKI_AUTO_LINT_EVERY`) in both watcher-driven auto-analysis and `/wiki/query` auto-maintenance.
- Automatic fallback-analysis retry now runs in that same maintenance cycle before enrichment.
- Direct maintenance endpoints now also run fallback-analysis retry before work:
  - `POST /api/inference/wiki/enrich`
  - `GET /api/inference/wiki/lint`

Schemas are in `studio/backend/models/inference.py`:
- `WikiIngestRequest`
- `WikiIngestResponse`
- `WikiRetryFallbackRequest`
- `WikiRetryFallbackResponse`
- `WikiQueryRequest`
- `WikiQueryResponse`
- `WikiLintResponse`

### 2c) Enrichment can now fill lint gaps from web search
Updated in `studio/backend/routes/inference.py`, `studio/backend/core/wiki/manager.py`, and `studio/backend/core/wiki/engine.py`:
- `POST /api/inference/wiki/enrich` now accepts:
  - `fill_gaps_from_web: bool`
  - `max_web_gap_queries: int`
- When enabled, enrichment performs a lint pass, reads `missing_concepts`, runs web search for selected gaps, drafts concept pages, rebuilds index, then continues normal analysis-page enrichment.
- Enrichment response now includes a `web_gap_fill` report with:
  - `enabled`
  - `lint_missing_concepts`
  - `concepts_considered`
  - `queries_used`
  - `concepts_created`
  - `created_pages`
  - `failed_concepts`
- In `dry_run` mode, concept drafts are reported but not written.

Schema updates in `studio/backend/models/inference.py`:
- `WikiEnrichRequest` now includes `fill_gaps_from_web` and `max_web_gap_queries`.
- `WikiEnrichResponse` now includes `web_gap_fill`.

### 3) Request-level RAG trace capture
The GGUF `/chat/completions` code path now stores the exact RAG context selection used for the last request, including:
- query
- selected pages + scores + snippets
- applied limits
- final context payload size

This trace powers `GET /api/inference/rag/debug/last`.

### 4) Retrieval exclusion for archives
Updated `studio/backend/core/wiki/engine.py`:
- `_all_wiki_pages()` now skips any file inside `.archive` directories.

This prevents archived stale pages from re-entering ranking/retrieval.

### 5) Watcher lifecycle + duplicate ingest hardening
Updated watcher ownership and event handling:
- `studio/backend/main.py` is now the single owner of watcher startup/shutdown lifecycle.
- `studio/backend/core/inference/inference.py` no longer starts a second watcher instance.
- `studio/backend/core/wiki/watcher.py` now handles `on_modified` events so real raw-file edits are ingested.
- `studio/backend/core/wiki/watcher.py` now computes per-file content hashes and suppresses duplicate re-ingests for unchanged bytes during filesystem event bursts.

## End-to-End Wiki Workflow (Current)
1. Place files in `UNSLOTH_WIKI_VAULT/raw` (or ingest directly via route helpers).
2. Ingestion reads content (including local PDFs), extracts wiki summary/entities/concepts, and writes source/entity/concept pages.
3. For GGUF chat requests, the route:
   - ingests pending raw files
   - retrieves context snippets from wiki pages
   - injects context as system block
   - stores chat history back into wiki sources
4. Debug with:
   - `POST /api/inference/rag/debug/context`
   - `GET /api/inference/rag/debug/last`
5. Periodically prune noisy history/duplicates with:
   - `POST /api/inference/wiki/archive/stale`

## RAG Context Size Guidance
Route-level controls (environment variables):
- `UNSLOTH_WIKI_RAG_MAX_PAGES` (default: `8`)
- `UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE` (default: `1800`)
- `UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS` (default: `12000`)
- `UNSLOTH_WIKI_LOG_INJECTED_CONTEXT` (default: `true`)
- `UNSLOTH_WIKI_LOG_INJECTED_CONTEXT_MAX_CHARS` (default: `12000`, `0` = no truncation)

Practical tuning for GGUF context length `n_ctx`:
- Keep combined RAG context under roughly 20-35% of `n_ctx` to reduce truncation pressure.
- If you hit context overflow errors:
  - lower `UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS` first
  - then lower `UNSLOTH_WIKI_RAG_MAX_PAGES`
  - then lower `UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE`
- For chat-history-heavy requests, consider raising pages but lowering chars/page to preserve breadth.

## Wiki Engine Quality Knobs
Environment variables for wiki extraction/ranking page quality:
- `UNSLOTH_WIKI_ENGINE_EXTRACT_SOURCE_MAX_CHARS` (default: `20000`)
- `UNSLOTH_WIKI_ENGINE_RANKING_MAX_CHARS` (default: `24000`)
- `UNSLOTH_WIKI_ENGINE_SOURCE_EXCERPT_MAX_CHARS` (default: `8000`)
- `UNSLOTH_WIKI_ENGINE_MAX_CONTEXT_PAGES` (default: `16`)
- `UNSLOTH_WIKI_ENGINE_MAX_CHARS_PER_PAGE` (default: `3500`)
- `UNSLOTH_WIKI_ENGINE_QUERY_CONTEXT_MAX_CHARS` (default: `24000`)

Environment variables for lint-driven web gap fill in enrichment:
- `UNSLOTH_WIKI_ENGINE_ENRICH_FILL_GAPS_FROM_WEB` (default: `false`)
- `UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_QUERIES` (default: `4`)
- `UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_RESULTS` (default: `3`)
- `UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_SNIPPET_CHARS` (default: `280`)

## Debug/API Examples

### Preview context for a query
```bash
curl -s -X POST http://127.0.0.1:8000/api/inference/rag/debug/context \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is my full name from the PDF?","include_pending_raw":true}'
```

### Get last context used by chat/completions
```bash
curl -s http://127.0.0.1:8000/api/inference/rag/debug/last
```

### Dry-run archival
```bash
curl -s -X POST http://127.0.0.1:8000/api/inference/wiki/archive/stale \
  -H 'Content-Type: application/json' \
  -d '{"dry_run":true,"keep_recent_chat":16,"keep_recent_per_source":1}'
```

### Dry-run fallback-analysis retry
```bash
curl -s -X POST http://127.0.0.1:8000/api/inference/wiki/retry-fallback \
  -H 'Content-Type: application/json' \
  -d '{"dry_run":true,"max_analysis_pages":24}'
```

### Enrich with lint-driven web gap fill
```bash
curl -s -X POST http://127.0.0.1:8000/api/inference/wiki/enrich \
  -H 'Content-Type: application/json' \
  -d '{"dry_run":false,"max_analysis_pages":64,"fill_gaps_from_web":true,"max_web_gap_queries":4}'
```

## Auth/PR Safety Plan
Current auth-related modified files seen in working tree:
- `studio/backend/auth/authentication.py`
- `studio/frontend/src/features/auth/components/auth-form.tsx`

Safe rollback for PR isolation (without reclone):
```bash
git restore studio/backend/auth/authentication.py
git restore studio/frontend/src/features/auth/components/auth-form.tsx
```

If you want to preserve local experiments before reverting:
```bash
git stash push -m 'auth experiment backup' -- studio/backend/auth/authentication.py studio/frontend/src/features/auth/components/auth-form.tsx
```

## Startup/Access Notes
Auth enabled (normal mode):
```bash
cd studio/backend
python main.py
```

Auth disabled (debug/testing mode only):
```bash
cd studio/backend
AUTH_DISABLED=true python main.py
```

Wiki watcher startup controls (for `python studio/backend/run.py`):
```bash
# Explicitly enable background raw-folder watcher
python studio/backend/run.py --host 127.0.0.1 --port 8888 --wiki-watcher

# Explicitly disable background raw-folder watcher
python studio/backend/run.py --host 127.0.0.1 --port 8888 --no-wiki-watcher

# Auto-run wiki query analysis after each raw ingest (default on)
python studio/backend/run.py --host 127.0.0.1 --port 8888 --wiki-auto-query

# Disable auto query analysis
python studio/backend/run.py --host 127.0.0.1 --port 8888 --no-wiki-auto-query

# Run lint every N auto analyses/queries
python studio/backend/run.py --host 127.0.0.1 --port 8888 --wiki-lint-every 10

# Include chat_history_* in auto-analysis (default off)
python studio/backend/run.py --host 127.0.0.1 --port 8888 --wiki-auto-query-chat-history
```

Environment equivalent:
```bash
UNSLOTH_WIKI_WATCHER=true  # or false
UNSLOTH_WIKI_AUTO_QUERY_ON_INGEST=true
UNSLOTH_WIKI_AUTO_QUERY_CHAT_HISTORY=false
UNSLOTH_WIKI_AUTO_LINT_EVERY=10
UNSLOTH_WIKI_AUTO_RETRY_FALLBACK_ANALYSES_MAX_PAGES=24
UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY=false
UNSLOTH_WIKI_CHAT_HISTORY_FLUSH_SECONDS=600
UNSLOTH_WIKI_LOG_INJECTED_CONTEXT=true
UNSLOTH_WIKI_LOG_INJECTED_CONTEXT_MAX_CHARS=12000
```

`UNSLOTH_WIKI_AUTO_LINT_EVERY` now controls both:
- auto lint cadence
- auto fallback-analysis retry cadence
- auto enrichment cadence

`UNSLOTH_WIKI_AUTO_RETRY_FALLBACK_ANALYSES_MAX_PAGES` controls how many recent analysis pages are scanned for fallback retries during each maintenance run (`0` disables fallback retries).

Wiki engine quality/context knobs (affect background ingestion + retrieval scoring):
```bash
# How much source text is sent into extraction
UNSLOTH_WIKI_ENGINE_EXTRACT_SOURCE_MAX_CHARS=150000

# How much of each page is scanned when ranking matches
UNSLOTH_WIKI_ENGINE_RANKING_MAX_CHARS=120000

# How much source excerpt is written onto source pages
UNSLOTH_WIKI_ENGINE_SOURCE_EXCERPT_MAX_CHARS=160000

# Retrieval defaults used by WikiEngine.query/retrieve_context
UNSLOTH_WIKI_ENGINE_MAX_CONTEXT_PAGES=10
UNSLOTH_WIKI_ENGINE_MAX_CHARS_PER_PAGE=20000

# Include prior analysis pages when answering wiki queries (default true)
UNSLOTH_WIKI_ENGINE_INCLUDE_ANALYSIS_IN_QUERY=true
```

Why `wiki/analysis` can be empty:
- The `analysis` folder is only populated by `WikiEngine.query(..., save_answer=True)`.
- Current chat + watcher flow ingests sources and retrieves snippets directly; it does not call `query()` for each request.
- So empty `analysis` is expected unless you run a query path that saves answer pages.

Why extraction diagnostics show fallback + prompt text:
- If extraction metadata shows `reason: llm_prompt_echo`, the wiki extraction callback returned the input prompt instead of JSON.
- This typically occurs when no active model response is available for wiki extraction and the route-level stub falls back to echoing the prompt.
- In that case ingestion still proceeds via heuristic extraction, but entity/concept quality is reduced until a model is loaded.

Extraction reliability hardening:
- Structured wiki extraction prompts now run with stricter deterministic decoding settings in the route-level wiki LLM stub.
- If first-pass extraction output is malformed/non-JSON, the engine now performs a JSON-repair pass before falling back to heuristics.
- Additional extraction reasons now distinguish prompt echo vs garbled model output (`llm_prompt_echo`, `llm_garbled_output`).

Analysis quality + chat history persistence hardening:
- Wiki query ranking now excludes `index.md` and `log.md`; `analysis/*` pages are included in query retrieval by default (configurable) to improve source-following and cross-page grounding.
- Saved analysis page filenames now use `YYYY-MM-DD` plus topic terms derived from the query (and fall back to retrieved context-page names when the query is generic), with collision-safe suffixes (`-2`, `-3`, ...).
- If a query answer looks low quality/garbled, the engine writes an extractive fallback answer from cited context instead of persisting gibberish.
- Chat history snapshots are buffered in memory and flushed to disk/wiki on a cadence (default 10 minutes) rather than every request.
- Index entries for fallback-generated analysis pages are explicitly marked (for example `[fallback: repetition]`) so retrieval and maintenance flows can de-prioritize low-confidence analysis.
- When fallback retry maintenance regenerates a better analysis, the prior fallback page is marked as resolved (for example `[fallback-resolved: analysis/<new-page>]`) in index metadata.

If login requires password change and you cannot proceed:
- Use the bootstrap password generated by backend startup logs.
- Complete the forced password change once, then log in normally.
- If bootstrap flow is out of sync, restart backend and check startup output for bootstrap/password prompts.

## Regression Status
Targeted regression test added earlier and passing:
- `studio/backend/tests/test_wiki_rag_pipeline.py`

## April 2026 Addendum (Wiki + Background Analysis)

This addendum captures follow-up hardening and tuning changes made after the initial Phase 3 pass.

### Additional changed files
- `studio/backend/core/inference/tools.py`
- `studio/backend/core/wiki/engine.py`
- `studio/backend/core/wiki/manager.py`
- `studio/backend/core/wiki/watcher.py`
- `studio/backend/main.py`
- `studio/backend/routes/inference.py`
- `studio/backend/tests/test_inference_tools_workdir.py`
- `studio/backend/tests/test_wiki_rag_pipeline.py`
- `studio/backend/tests/test_wiki_watcher.py`
- `updates.md`

### Ingestion hygiene: skip metadata and hidden files
The wiki pipeline now consistently skips hidden/system metadata files such as:
- `.DS_Store`
- `._*`
- dotfiles
- `Thumbs.db`

Applied in:
- direct ingest (`WikiIngestor.ingest_file` / directory walks)
- watcher-triggered ingest events
- route-level pending-raw ingestion sweep

This prevents junk pages and noisy retrieval candidates from entering the wiki.

### Tool sandbox path fix for relative wiki commands
Tool calls (`terminal`, `python`) run in a per-session sandbox directory, which previously caused relative paths like `sources/` to fail.

Now each sandbox exposes wiki shortcuts as symlinks to the active vault:
- `wiki/`
- `sources/`
- `entities/`
- `concepts/`
- `analysis/`
- `raw/`

This keeps model tool calls path-stable without requiring absolute paths.

### Analysis quality hardening and diagnostics
To reduce low-quality persisted answers and make failures debuggable:
- Added fallback reason recording and raw-answer preview in analysis pages.
- Added retrieval diagnostics block in analysis pages, including:
  - link-depth/fanout settings
  - effective context limits
  - ranked-page count and used-page count
- Strengthened query prompt to treat instructions inside context pages as quoted source text, not executable instructions.
- Added explicit suppression guidance for chain-of-thought style tags in generated answers.
- Added optional lint-driven web gap filling in enrichment, including automatic concept draft generation from `missing_concepts`.
- Web search dependency for enrichment gap fill is dynamically imported so missing optional packages do not break baseline runtime.

### Context control updates

#### 1) Recursive retrieval expansion (enabled by default)
Wiki ranking can expand through wiki links to configurable depth/fanout:
- `UNSLOTH_WIKI_ENGINE_RANKING_LINK_DEPTH`
- `UNSLOTH_WIKI_ENGINE_RANKING_LINK_FANOUT`

Default depth is `2` and default fanout is `8`, so source-following works out-of-the-box.

#### 1b) Hybrid ranking signals (now default)
`_rank_pages` now combines multiple signals instead of relying only on lexical overlap:
- token overlap in page content
- overlap with path/slug tokens
- overlap with title/header tokens
- phrase matches (quoted phrases and compact query phrases)
- entity-intent boosts for prompts like "who is ..."

If lexical scoring produces no candidates, ranking now falls back to recency ordering instead of returning an empty list.

This improves robustness for sparse corpora, new entities, and short biography-style prompts.

#### 1c) LLM reranking stage (same backend/model path)
The reranker can reorder top retrieval candidates using the same `llm_fn` path already used by the wiki engine (so it uses the same model server/backend currently active).

Controls:
- `UNSLOTH_WIKI_ENGINE_LLM_RERANK_ENABLED` (default: `true`)
- `UNSLOTH_WIKI_ENGINE_LLM_RERANK_CANDIDATES` (default: `32`)
- `UNSLOTH_WIKI_ENGINE_LLM_RERANK_TOP_N` (default: `12`)
- `UNSLOTH_WIKI_ENGINE_LLM_RERANK_PREVIEW_CHARS` (default: `420`)

Design safeguards:
- reranker can only return pages from current candidate set
- invalid/non-parseable reranker output falls back to deterministic order
- deterministic ranker still runs first; reranker is a bounded second-stage reorder

#### 2) Engine "0 means unlimited" behavior
The wiki engine/manager now treat `0` as unlimited for:
- `UNSLOTH_WIKI_ENGINE_MAX_CONTEXT_PAGES`
- `UNSLOTH_WIKI_ENGINE_MAX_CHARS_PER_PAGE`
- `UNSLOTH_WIKI_ENGINE_QUERY_CONTEXT_MAX_CHARS`
- `UNSLOTH_WIKI_ENGINE_RANKING_MAX_CHARS`

This allows full-page/full-set analysis context when explicitly requested.

#### 3) Background auto-analysis can scale to model context (not chat RAG)
Watcher-only auto-analysis now supports deriving context budget from the currently loaded model context window using:

- `UNSLOTH_WIKI_AUTO_ANALYSIS_CONTEXT_FRACTION` (default: `0.70`)
- `UNSLOTH_WIKI_AUTO_ANALYSIS_CHARS_PER_TOKEN` (default: `4`)

Effective analysis context budget (characters) is:

`max(500, model_context_tokens * fraction * chars_per_token)`

This applies only to background watcher analysis calls and does not change route-level chat RAG limits.

Background analysis mode flags:
- `UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY` (default: `false`)
  - Force source-only analysis context from the first attempt.
- `UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY_FINAL_RETRY` (default: `true`)
  - If source-only mode is not already enabled, switch to source-only on the final fallback retry.

GGUF wiki prefill/read timeout:
- `UNSLOTH_LLAMA_CPP_PREFILL_READ_TIMEOUT_SECONDS` (default: `120`)
  - Controls the backend llama.cpp prefill/read timeout used by wiki LLM calls.
  - This is a server-side timeout and is separate from client-side curl `--max-time`.

### What `UNSLOTH_WIKI_AUTO_ANALYSIS_CHARS_PER_TOKEN` means
`UNSLOTH_WIKI_AUTO_ANALYSIS_CHARS_PER_TOKEN` is a rough conversion factor from tokens to characters when estimating how much wiki text to include in background analysis context.

- Why needed: model context is measured in tokens, but wiki snippets are counted in characters.
- Default `4` means: estimate that 1 token is about 4 characters on average.
- Example: if model context is 8192 tokens and fraction is 0.70, estimated char budget is:
  - `8192 * 0.70 * 4 = 22937` chars (rounded down)

If your corpus has longer/denser words, raise this value slightly; if answers start degrading from oversized prompts, lower it.

### Low-quality gate tuning (low_unique_ratio)
The lexical-diversity fallback gate is now configurable via env:

- `UNSLOTH_WIKI_LOW_UNIQUE_RATIO_MIN_TOKENS` (default: `40`)
- `UNSLOTH_WIKI_LOW_UNIQUE_RATIO_THRESHOLD` (default: `0.25`)

Use these to reduce false positives when valid technical summaries repeat terminology.

### Prompt update for watcher summaries
Watcher auto-analysis prompt was rewritten to be source-first and explicitly grounded to the newly ingested page (`[[sources/<slug>]]`) with structured section requirements.

### Current recommended env baseline
```bash
# Model-capacity-driven wiki char budgeting baseline
UNSLOTH_WIKI_ENGINE_MODEL_TOKEN_CAPACITY=125000
UNSLOTH_WIKI_ENGINE_MODEL_SAFE_TOKEN_RATIO=0.50
UNSLOTH_WIKI_ENGINE_MODEL_CHARS_PER_TOKEN=4.0

# Watcher + auto-analysis
UNSLOTH_WIKI_WATCHER=true
UNSLOTH_WIKI_AUTO_QUERY_ON_INGEST=true
UNSLOTH_WIKI_AUTO_ANALYSIS_CONTEXT_FRACTION=0.70
UNSLOTH_WIKI_AUTO_ANALYSIS_CHARS_PER_TOKEN=4
UNSLOTH_WIKI_AUTO_ANALYSIS_RETRY_ON_FALLBACK=true
UNSLOTH_WIKI_AUTO_ANALYSIS_MAX_RETRIES=3
UNSLOTH_WIKI_AUTO_ANALYSIS_RETRY_REDUCTION=0.5
UNSLOTH_WIKI_AUTO_ANALYSIS_MIN_CONTEXT_CHARS=8000
UNSLOTH_WIKI_AUTO_LINT_EVERY=10  # shared schedule for lint + fallback-retry + enrichment
UNSLOTH_WIKI_AUTO_RETRY_FALLBACK_ANALYSES_MAX_PAGES=24
UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY=true
UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY_FINAL_RETRY=true
UNSLOTH_LLAMA_CPP_PREFILL_READ_TIMEOUT_SECONDS=300

# Optional: unlimited engine-side context (0 = unlimited)
UNSLOTH_WIKI_ENGINE_MAX_CONTEXT_PAGES=0
UNSLOTH_WIKI_ENGINE_MAX_CHARS_PER_PAGE=0
UNSLOTH_WIKI_ENGINE_QUERY_CONTEXT_MAX_CHARS=0
UNSLOTH_WIKI_ENGINE_RANKING_MAX_CHARS=0

# Recursive link expansion defaults
UNSLOTH_WIKI_ENGINE_RANKING_LINK_DEPTH=2
UNSLOTH_WIKI_ENGINE_RANKING_LINK_FANOUT=8

# LLM reranking defaults (same wiki llm_fn backend)
UNSLOTH_WIKI_ENGINE_LLM_RERANK_ENABLED=true
UNSLOTH_WIKI_ENGINE_LLM_RERANK_CANDIDATES=32
UNSLOTH_WIKI_ENGINE_LLM_RERANK_TOP_N=12
UNSLOTH_WIKI_ENGINE_LLM_RERANK_PREVIEW_CHARS=420

# Include analysis pages in query retrieval by default
UNSLOTH_WIKI_ENGINE_INCLUDE_ANALYSIS_IN_QUERY=true

# Optional: lint-driven web gap fill during /wiki/enrich
UNSLOTH_WIKI_ENGINE_ENRICH_FILL_GAPS_FROM_WEB=false
UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_QUERIES=4
UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_RESULTS=3
UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_SNIPPET_CHARS=280
```

### Additional regression tests
Passing tests now include:
- `studio/backend/tests/test_wiki_rag_pipeline.py`
- `studio/backend/tests/test_inference_tools_workdir.py`
- `studio/backend/tests/test_wiki_watcher.py`

## April 2026 Addendum (Graphify Reuse Expansion)

This pass increases direct graphify reuse in Studio wiki maintenance flows and removes duplicate lower-feature logic where graphify already provides stronger behavior.

### Additional changed files
- `studio/backend/core/wiki/ingestor.py`
- `studio/backend/core/wiki/engine.py`
- `studio/backend/routes/inference.py`
- `studio/backend/models/inference.py`
- `studio/backend/tests/test_wiki_rag_pipeline.py`
- `updates.md`

### 1) Pending raw ingest now uses graphify-style detection + classification
Route-level pending ingestion (`/wiki/ingest` without `source_path`, plus any path that triggers pending-raw sweep) now delegates to a new ingestor service method:

- `WikiIngestor.ingest_pending_raw_files(...)`

What changed:
- Candidate detection now prefers graphify detect (`graphify.detect.detect`) when available.
- Classification now prefers graphify file typing (`classify_file`) instead of a hard-coded extension subset.
- Sensitive file candidates skipped by graphify detect are now excluded from ingestion sweeps.
- Fallback behavior still exists if graphify detect/cache modules are unavailable.

Practical impact:
- Better file-type coverage and filtering quality.
- Reduced accidental ingestion of risky/sensitive raw files.
- Less custom duplicate filtering logic in route code.

### 2) Persistent incremental ingest state for unchanged-file skipping
Pending raw ingestion now stores a persistent hash map in:

- `UNSLOTH_WIKI_VAULT/raw/.ingest_state.json`

Behavior:
- Successfully ingested files are tracked by content hash.
- Unchanged files are skipped across backend restarts.
- If a raw file changes, it is re-ingested even when a prior source page exists.

Hashing uses graphify cache hashing when available (`graphify.cache.file_hash`) with a local SHA256 fallback.

### 3) Lint now includes graphify structural insights
`WikiEngine.lint()` now returns an additional payload:

- `graphify_insights`

This is surfaced by:
- `GET /api/inference/wiki/lint`
- `WikiLintResponse.graphify_insights`

Payload includes:
- `available` (bool)
- `reason` (string status/failure reason)
- `god_nodes`
- `surprising_connections`
- `community_count`

Implementation details:
- Graphify analyze is loaded dynamically (`graphify.analyze`) with monorepo import fallback.
- If graphify/networkx is unavailable, lint remains successful and returns `available: false` with a reason.

### 4) Replaced lower-feature duplicate logic
Removed/avoided duplicated route-level pending ingest filtering logic in favor of the richer ingestor + graphify-backed path.

Net effect:
- Better feature set with fewer hard-coded filters.
- Centralized ingest candidate decisions in one service.

### 5) New graphify wiki export endpoint
Added endpoint:

- `POST /api/inference/wiki/export/graphify-wiki`

Request:
- `output_subdir` (default: `graphify-wiki`)

Behavior:
- Projects current wiki pages/links into a lightweight graph model.
- Uses graphify analysis for god-node selection.
- Uses graphify wiki exporter to write a browsable markdown wiki (`index.md` + community articles + god-node articles).
- Writes output under `wiki/<output_subdir>`.

Response shape:
- `status` (`ok`, `unavailable`, or `error`)
- `reason`
- `output_dir`
- `index_file`
- `articles_written`
- `communities`
- `god_nodes`

If graphify/networkx modules are unavailable, export returns `status: unavailable` with a reason while keeping the rest of wiki APIs functional.

### 6) Chat latency guard for pending raw ingest
To keep `/v1/chat/completions` responsive, pending raw ingestion in the GGUF chat path is now throttled and capped.

New env knobs:
- `UNSLOTH_WIKI_PENDING_INGEST_INTERVAL_SECONDS` (default: `45`)
  - Minimum time between automatic pending-ingest sweeps triggered by chat requests.
- `UNSLOTH_WIKI_PENDING_INGEST_MAX_FILES_PER_CHAT` (default: `1`)
  - Maximum pending files ingested synchronously per chat request.
  - Set to `0` to disable chat-triggered pending ingestion entirely.

Notes:
- Explicit maintenance calls (`/wiki/ingest`, debug context with `include_pending_raw=true`) bypass interval throttling so manual/diagnostic operations remain immediate.

## April 2026 Addendum (Merge Maintenance + Source Policy)

This pass adds a first-class duplicate-merge maintenance workflow and reduces retrieval/index overhead when source pages are intentionally excluded.

### Additional changed files
- `studio/backend/core/wiki/engine.py`
- `studio/backend/core/wiki/manager.py`
- `studio/backend/routes/inference.py`
- `studio/backend/models/inference.py`
- `studio/backend/tests/test_wiki_rag_pipeline.py`
- `updates.md`

### 1) Internal wiki maintenance now keeps RAG history enabled by default
Route-level internal wiki maintenance calls now keep `enable_wiki_rag_history=true` by default so maintenance runs have the same retrieval/history context behavior as normal wiki-assisted paths.

### 2) Source-page exclusion now happens earlier (retrieval path), with index compaction
Retrieval now enforces source inclusion policy before ranking/rerank work, so disabled source pages do not consume scoring budget or rerank tokens.

Related behavior updates:
- `retrieve_context(...)` now accepts `include_source_pages` through manager/route wiring.
- `_rank_pages(...)` respects effective source policy directly.
- LLM rerank prompt now uses compact candidate-only planner text (instead of a broad index dump).
- Wiki index generation can omit the large Sources listing when source inclusion is disabled.

Environment knobs:
- `UNSLOTH_WIKI_RAG_INCLUDE_SOURCE_PAGES`
- `UNSLOTH_WIKI_INDEX_INCLUDE_SOURCE_PAGES`

### 3) Lint now reports merge candidates for entities and concepts
`GET /api/inference/wiki/lint` now includes merge candidate suggestions:
- `entity_merge_candidates`
- `concept_merge_candidates`

These are advisory and non-destructive; they are intended to feed maintenance workflows.

### 4) New duplicate merge-maintenance endpoint (dry-run + apply)
Added endpoint:
- `POST /api/inference/wiki/merge-maintenance`

Request/response models:
- `WikiMergeMaintenanceRequest`
- `WikiMergeMaintenanceResponse`

Apply behavior:
- chooses canonical pages for high-confidence duplicates
- archives duplicate pages under `wiki/.archive/...`
- rewrites wiki links to canonical targets
- rebuilds wiki index after apply

Dry-run behavior:
- returns planned operations and counts without writing.

### 5) Verification status
Focused regression tests:
```bash
/Users/zohairshafi/Local\ Workspace/unsloth/.venv/bin/python -m pytest -q studio/backend/tests/test_wiki_rag_pipeline.py
# 38 passed
```

Live API smoke (auth-enabled backend):
- Dry run: `POST /api/inference/wiki/merge-maintenance` -> `HTTP 200`
- Apply mode: `POST /api/inference/wiki/merge-maintenance` -> `HTTP 200`
- Observed response field in both runs: `rewritten_links: 303`

### 6) Post-review fixes from commit `5b4f360ff0`

Addressed two correctness issues raised in automated review:

- **Stale archive source grouping now uses full source identity**
  - Prior behavior grouped by `Path(source_ref).stem`, which could collide unrelated sources such as `raw/repoA/README.md` and `raw/repoB/README.md`.
  - `_archive_stale_wiki_pages(...)` now groups by a full source identity key derived from `source_ref` (with raw-relative normalization where possible), falling back to page stem only if `source_ref` is missing.
  - This prevents accidental archival of unrelated sources when `keep_recent_per_source=1`.

- **Wiki raw watcher now monitors recursively**
  - `WikiIngestionWatcher.start()` now schedules watchdog with `recursive=True`.
  - Files added under nested folders in `raw/` are now observed by the watcher path, matching recursive ingest expectations.

Added regression tests:
- `studio/backend/tests/test_wiki_archive_stale.py`
- `studio/backend/tests/test_wiki_watcher.py::test_watcher_start_schedules_raw_dir_recursively`

## April 2026 Follow-up (P2 fixes + rewrite controls)

This follow-up addresses two review-flagged P2 issues and adds explicit rewrite/compaction controls to maintenance/enrichment APIs so wiki pages do not grow unboundedly.

### Additional changed files
- `graphify/graphify/ingest.py`
- `studio/backend/core/inference/inference.py`
- `studio/backend/core/wiki/engine.py`
- `studio/backend/core/wiki/manager.py`
- `studio/backend/routes/inference.py`
- `studio/backend/models/inference.py`
- `studio/backend/tests/test_wiki_rag_pipeline.py`
- `updates.md`

### 1) Graphify tweet oEmbed import bug fixed
Issue:
- `_fetch_tweet(...)` used `urllib.request.Request/urlopen` without importing `urllib.request`.

Fix:
- Added `import urllib.request` in `graphify/graphify/ingest.py`.

Impact:
- Tweet/oEmbed fetch path no longer silently degrades due to `AttributeError` when oEmbed is otherwise available.

### 2) Inference startup env parsing hardened
Issue:
- `InferenceBackend` parsed `UNSLOTH_WIKI_CHAT_HISTORY_FLUSH_SECONDS` with raw `int(os.getenv(...))` during backend initialization.
- Non-numeric values could crash import/startup.

Fix:
- Added `_safe_env_int(...)` helper in `studio/backend/core/inference/inference.py`.
- Replaced the direct cast with guarded parsing + fallback default (`600`) and minimum bound (`0`).

Impact:
- Invalid env values no longer prevent inference backend startup.

### 3) Rewrite/compaction controls for maintenance + enrichment
Findings:
- `analysis` enrichment/retry paths already used upsert-style section rewrites.
- Entity/concept knowledge pages could still grow over time via `## Incremental Updates` accumulation.

Implemented controls:
- New wiki env knob:
  - `UNSLOTH_WIKI_KNOWLEDGE_MAX_INCREMENTAL_UPDATES` (default `48`, max `256`)
- `_upsert_knowledge_page(...)` now trims `Incremental Updates` to the configured maximum.
- New engine maintenance helper:
  - `compact_knowledge_pages(...)`
  - Trims oversized `Incremental Updates` blocks in `entities/*` and `concepts/*`.

API additions:
- `POST /api/inference/wiki/enrich`
  - `compact_knowledge_pages: bool = false`
  - `max_incremental_updates: int`
  - response now includes `knowledge_compaction` report.
- `POST /api/inference/wiki/merge-maintenance`
  - `compact_knowledge_pages: bool = false`
  - `max_incremental_updates: int`
  - response now includes `knowledge_compaction` report.

Behavioral note:
- With compaction enabled, enrichment/merge flows can actively rewrite oversized knowledge pages (not just append), while preserving recent incremental history.

### 4) Verification
Targeted checks passed:
- `graphify/tests/test_ingest.py` -> `8 passed`
- `studio/backend/tests/test_wiki_rag_pipeline.py -k "upsert_knowledge_page_caps_incremental_updates or enrich_analysis_pages_can_compact_knowledge_updates or merge_maintenance_can_compact_knowledge_updates_without_merges"` -> `3 passed`
- `py_compile studio/backend/core/inference/inference.py` -> success

### 5) Future add-ons (deferred): chat web-search -> wiki ingestion
Status:
- Not implemented in this follow-up.
- Captured here as a potential roadmap only.

Current behavior snapshot:
- `web_search` tool output is emitted via tool events and rendered in chat UI source/tool blocks.
- It is not automatically ingested into wiki source pages today.
- Route-level chat history flush currently snapshots incoming chat messages before the GGUF tool loop, so tool outputs are typically not part of that persisted batch.

Potential phased rollout:
1. Phase A (low-risk, URL-only ingest)
  - Ingest only successful `web_search` calls that used the `url` argument (full page fetch mode).
  - Hook at GGUF tool event handling in `studio/backend/routes/inference.py` when `tool_end` is available.
  - Write a wiki source page through existing wiki manager ingest path, with `source_ref` set to the URL.

2. Phase B (safety + quality guardrails)
  - Add opt-in env flag (default off), for example: `UNSLOTH_WIKI_INGEST_WEB_SEARCH=false`.
  - Skip known non-content/error results (for example: `No results found`, `Search failed`, `Blocked`, fetch errors).
  - Add dedupe/rate controls (canonical URL + content hash; per-thread/per-request limits).

3. Phase C (provenance + retrieval tuning)
  - Persist minimal provenance (query, URL, tool_call_id, timestamp) in source metadata.
  - Optionally tag web-derived pages so ranking can down-weight them unless explicitly requested.

Suggested tests when implemented:
- Unit: tool_end URL path triggers ingest once.
- Unit: dedupe + error-skip behavior.
- Integration: ingested web page is retrievable through wiki query context.

No API/runtime changes were made for this deferred item.

### 6) Environment variable reference (consolidated)

The tables below list all `UNSLOTH_*` environment variables currently referenced in this document, with a brief description and a practical recommended value.

Runtime UI note:
- The `Edit Wiki Behaviour` dialog (`/wiki/env`) shows the runtime-editable wiki subset from `WIKI_ENV_SPECS`.
- Two variables documented below are currently not part of that runtime-editable wiki subset:
  - `UNSLOTH_LLAMA_CPP_PREFILL_READ_TIMEOUT_SECONDS`
  - `UNSLOTH_WIKI_INGEST_WEB_SEARCH` (deferred/not implemented)

#### 6.1 Core wiki path, watcher, and ingest cadence

| Env var | What it controls | Recommended value | Notes |
|---|---|---|---|
| `UNSLOTH_WIKI_VAULT` | Root wiki vault path (`raw/`, `wiki/`, etc.) | `./.unsloth_wiki` (project-local persistent path) | Avoid `/tmp` for long-lived knowledge unless intentionally ephemeral. |
| `UNSLOTH_WIKI_WATCHER` | Enables background raw-folder watcher | `true` | Set `false` only for manual/route-only ingest workflows. |
| `UNSLOTH_WIKI_PENDING_INGEST_INTERVAL_SECONDS` | Minimum delay between route-triggered pending ingest sweeps | `45` | Lower only if you need near-real-time ingest via chat path. |
| `UNSLOTH_WIKI_PENDING_INGEST_MAX_FILES_PER_CHAT` | Max pending raw files ingested per chat request | `1` | Increase cautiously to avoid latency spikes in chat completions. |
| `UNSLOTH_WIKI_AUTO_QUERY_ON_INGEST` | Auto-run wiki analysis query after ingest | `true` | Disable if ingest throughput is more important than immediate analysis pages. |
| `UNSLOTH_WIKI_AUTO_QUERY_CHAT_HISTORY` | Include `chat_history_*` files in auto-analysis | `false` | Keep off by default to reduce noisy analysis pages. |
| `UNSLOTH_WIKI_CHAT_HISTORY_FLUSH_SECONDS` | Chat-history batch flush interval | `600` | `0` forces immediate flush per request (higher IO churn). |
| `UNSLOTH_WIKI_AUTO_LINT_EVERY` | Shared cadence for lint, fallback-retry scan, enrichment | `10` | Use `5` for more aggressive maintenance, higher for lower overhead. |
| `UNSLOTH_WIKI_AUTO_RETRY_FALLBACK_ANALYSES_MAX_PAGES` | Max recent fallback analysis pages scanned per maintenance run | `24` | Set `0` to disable fallback-retry scanning. |

#### 6.2 Route-level chat RAG controls

| Env var | What it controls | Recommended value | Notes |
|---|---|---|---|
| `UNSLOTH_WIKI_RAG_MAX_PAGES` | Max pages injected by route-level RAG | `8` | Lower to `4-6` if responses start repeating or drifting. |
| `UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE` | Per-page snippet cap in route-level RAG | `1800` | Lower to `1000-1400` for small-context GGUF models. |
| `UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS` | Total route-level injected context cap | `12000` | First knob to reduce when you see context pressure or garbling. |
| `UNSLOTH_WIKI_RAG_INCLUDE_SOURCE_PAGES` | Include `wiki/sources/*` in route-level retrieval | `true` | Keep enabled unless source pages are very noisy. |
| `UNSLOTH_WIKI_INDEX_INCLUDE_SOURCE_PAGES` | Include sources in index-level retrieval helpers | `true` | Keep aligned with `UNSLOTH_WIKI_RAG_INCLUDE_SOURCE_PAGES`. |
| `UNSLOTH_WIKI_LLM_MAX_TOKENS` | Token budget for wiki-generated responses | `1200` | Lower for faster/shorter answers, raise carefully for depth if model/context allows. |
| `UNSLOTH_WIKI_LOG_INJECTED_CONTEXT` | Logs injected RAG context to backend logs | `true` during tuning, `false` for quiet prod logs | Useful for diagnosing context quality and truncation behavior. |
| `UNSLOTH_WIKI_LOG_INJECTED_CONTEXT_MAX_CHARS` | Max logged chars for injected context | `12000` | Set `0` for no log truncation (can produce very large logs). |

#### 6.3 Wiki engine retrieval, extraction, and ranking

| Env var | What it controls | Recommended value | Notes |
|---|---|---|---|
| `UNSLOTH_WIKI_ENGINE_MODEL_TOKEN_CAPACITY` | Model token capacity baseline used to derive char limits when per-knob overrides are unset | `125000` | Set `0` to disable derivation and keep explicit per-knob values only. |
| `UNSLOTH_WIKI_ENGINE_MODEL_SAFE_TOKEN_RATIO` | Safe fraction of model token capacity used for wiki budgets | `0.50` | Safe token budget = `MODEL_TOKEN_CAPACITY * MODEL_SAFE_TOKEN_RATIO`. |
| `UNSLOTH_WIKI_ENGINE_MODEL_CHARS_PER_TOKEN` | Token-to-character heuristic used for derived wiki char limits | `4.0` | Safe char budget = safe token budget × chars-per-token. |
| `UNSLOTH_WIKI_ENGINE_EXTRACT_SOURCE_MAX_CHARS` | Max source text fed into extraction | `20000` | Raise for long papers only if extraction misses key sections. |
| `UNSLOTH_WIKI_ENGINE_SOURCE_EXCERPT_MAX_CHARS` | Max excerpt persisted on source pages | `8000` | Larger excerpts improve traceability but increase page size. |
| `UNSLOTH_WIKI_ENGINE_RANKING_MAX_CHARS` | Max chars read per page for ranking | `24000` | Do not set to `0` while debugging quality; unlimited can destabilize ranking. |
| `UNSLOTH_WIKI_ENGINE_MAX_CONTEXT_PAGES` | Max pages used by engine query context | `16` | `0` means unlimited; prefer finite caps for stable analysis quality. |
| `UNSLOTH_WIKI_ENGINE_MAX_CHARS_PER_PAGE` | Per-page cap for engine query context | `3500` | `0` means unlimited; risky for repetition/degeneration. |
| `UNSLOTH_WIKI_ENGINE_QUERY_CONTEXT_MAX_CHARS` | Total engine query context cap | `24000` | `0` means unlimited; avoid during fallback-quality tuning. |
| `UNSLOTH_WIKI_ENGINE_INCLUDE_ANALYSIS_IN_QUERY` | Include prior `analysis/*` pages in retrieval | `true` | Set `false` if old analyses bias current answers too much. |
| `UNSLOTH_WIKI_ENGINE_RANKING_LINK_DEPTH` | Graph/link expansion depth in ranking | `2` | Larger values can increase recall but also add noise/latency. |
| `UNSLOTH_WIKI_ENGINE_RANKING_LINK_FANOUT` | Link fanout per hop during expansion | `8` | Keep moderate to prevent combinatorial candidate growth. |
| `UNSLOTH_WIKI_ENGINE_LLM_RERANK_ENABLED` | LLM rerank on ranking candidates | `true` | Keep enabled for better relevance when latency budget permits. |
| `UNSLOTH_WIKI_ENGINE_LLM_RERANK_CANDIDATES` | Candidate pool size before LLM rerank | `32` | Lower for speed, raise for recall on broad corpora. |
| `UNSLOTH_WIKI_ENGINE_LLM_RERANK_TOP_N` | Number of reranked candidates kept | `12` | Keep below candidates; tune with context caps together. |
| `UNSLOTH_WIKI_ENGINE_LLM_RERANK_PREVIEW_CHARS` | Per-candidate preview chars for rerank prompt | `420` | Raise only if reranker lacks enough local context. |
| `UNSLOTH_WIKI_ENGINE_LLM_RERANK_LOG_OUTPUT` | Enable reranker output logging for diagnostics | `true` during tuning, `false` for quieter logs | Keep on while validating retrieval quality; disable in noisy/production runs. |
| `UNSLOTH_WIKI_ENGINE_LLM_RERANK_LOG_MAX_CHARS` | Max chars logged per reranker output entry | `4000` | Lower if logs are too verbose; raise only when deeper rerank debugging is needed. |

Quick copy-paste preset for a 125k model-token baseline:

```bash
UNSLOTH_WIKI_ENGINE_MODEL_TOKEN_CAPACITY=125000
UNSLOTH_WIKI_ENGINE_MODEL_SAFE_TOKEN_RATIO=0.50
UNSLOTH_WIKI_ENGINE_MODEL_CHARS_PER_TOKEN=4.0
```

With defaults above (and no explicit per-knob char overrides), derived budgets are:
- Safe token budget: `62500`
- Safe char budget: `250000`
- Derived `EXTRACT_SOURCE_MAX_CHARS`, `QUERY_CONTEXT_MAX_CHARS`, `RANKING_MAX_CHARS`, `CHUNK_ANALYSIS_CONTEXT_WINDOW_CHARS`, `CHUNK_ANALYSIS_MAX_CHARS`: `250000`
- Derived `SOURCE_EXCERPT_MAX_CHARS`: `50000`
- Derived `MAX_CHARS_PER_PAGE` (with `MAX_CONTEXT_PAGES=16`): `15625`

#### 6.4 Background auto-analysis and fallback controls

| Env var | What it controls | Recommended value | Notes |
|---|---|---|---|
| `UNSLOTH_WIKI_AUTO_ANALYSIS_CONTEXT_FRACTION` | Fraction of model context budget allocated to auto-analysis context | `0.70` | Lower (`0.50-0.60`) if auto analyses are repetitive/garbled. |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_CHARS_PER_TOKEN` | Token->char conversion heuristic | `4` | Keep at `4` unless tokenizer/domain strongly differs. |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_RETRY_ON_FALLBACK` | Retry auto-analysis when fallback quality gates trigger | `true` | Disable only to minimize compute. |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_MAX_RETRIES` | Max retry attempts after fallback | `3` | `2-3` is usually enough; more can amplify cost without benefit. |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_RETRY_REDUCTION` | Context reduction factor per retry | `0.5` | Smaller factor reduces context faster and often improves stability. |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_MIN_CONTEXT_CHARS` | Floor for retry-reduced context | `8000` | Lower to `4000-6000` for very small models if still garbled. |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY` | Force source-only analysis context | `false` (normal), `true` (stability mode) | Strongly helps when analysis pages become self-referential/noisy. |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY_FINAL_RETRY` | Force source-only context on final retry | `true` | Good guardrail to recover from repeated fallback loops. |
| `UNSLOTH_LLAMA_CPP_PREFILL_READ_TIMEOUT_SECONDS` | Prefill read timeout for llama.cpp generation path | `300` for typical local runs, increase up to `18000` for very large prompts/models | Raise if long-prefill requests fail with timeout. |
| `UNSLOTH_WIKI_LOW_UNIQUE_RATIO_MIN_TOKENS` | Minimum token count before low-unique-ratio gate applies | `40` | Keep default unless gate triggers too aggressively on short outputs. |
| `UNSLOTH_WIKI_LOW_UNIQUE_RATIO_THRESHOLD` | Repetition gate threshold for low unique-token ratio | `0.25` | Raise slightly (`0.28-0.32`) for stricter anti-repetition gating. |

#### 6.5 Enrichment, compaction, and deferred knobs

| Env var | What it controls | Recommended value | Notes |
|---|---|---|---|
| `UNSLOTH_WIKI_ENGINE_ENRICH_FILL_GAPS_FROM_WEB` | Enable lint-driven web gap fill during enrichment | `false` by default | Turn on for targeted enrichment passes, not continuously. |
| `UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_QUERIES` | Max missing-concept web queries per run | `4` | Increase gradually if concept coverage remains poor. |
| `UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_RESULTS` | Max search results considered per query | `3` | Higher values can pull in noise quickly. |
| `UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_SNIPPET_CHARS` | Snippet length cap used for drafted concept pages | `280` | Keep concise to avoid verbose low-confidence drafts. |
| `UNSLOTH_WIKI_ENGINE_ENRICH_REFRESH_OLDEST_NON_FALLBACK_PAGES` | Refresh oldest non-fallback analysis pages before link enrichment | `0` (off by default), try `3-12` for periodic freshness sweeps | Useful for stale-summary refresh without touching fallback-only retry behavior. |
| `UNSLOTH_WIKI_ENGINE_ENRICH_REPAIR_ANSWER_LINKS` | Repair unresolved wiki links inside analysis Answer sections during enrichment | `false` (safe default), set `true` for aggressive cleanup | Enabled mode can remove unresolved citations from prose; keep off when preserving original answer wording is preferred. |
| `UNSLOTH_WIKI_MERGE_MAINTENANCE_MAX_MERGES` | Max concept/entity merge operations per maintenance run | `512` | Lower if maintenance spikes CPU/latency; raise only for very stale corpora. |
| `UNSLOTH_WIKI_KNOWLEDGE_MAX_INCREMENTAL_UPDATES` | Max retained incremental update blocks per entity/concept page | `48` | Lower for compact pages, raise for deeper change history retention. |
| `UNSLOTH_WIKI_INGEST_WEB_SEARCH` | Deferred future flag for chat web-search ingestion to wiki | `false` | Not implemented in current runtime; documented as future add-on. |

#### 6.6 Caveats and recommended profile for garbled analyses

If you observe repetitive, incoherent, or obviously degenerate `analysis/*` output:

1. Avoid unlimited context settings during diagnosis:
  - Do **not** use `0` for:
    - `UNSLOTH_WIKI_ENGINE_MAX_CONTEXT_PAGES`
    - `UNSLOTH_WIKI_ENGINE_MAX_CHARS_PER_PAGE`
    - `UNSLOTH_WIKI_ENGINE_QUERY_CONTEXT_MAX_CHARS`
    - `UNSLOTH_WIKI_ENGINE_RANKING_MAX_CHARS`

2. Apply this stability profile first (known-good starting point):

| Env var | Recommended value for garbled-analysis mitigation |
|---|---|
| `UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS` | `6000` |
| `UNSLOTH_WIKI_RAG_MAX_PAGES` | `4` |
| `UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE` | `1200` |
| `UNSLOTH_WIKI_ENGINE_QUERY_CONTEXT_MAX_CHARS` | `8000` |
| `UNSLOTH_WIKI_ENGINE_MAX_CONTEXT_PAGES` | `6` |
| `UNSLOTH_WIKI_ENGINE_MAX_CHARS_PER_PAGE` | `1400` |
| `UNSLOTH_WIKI_ENGINE_RANKING_MAX_CHARS` | `12000` |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_CONTEXT_FRACTION` | `0.55` |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY` | `true` |
| `UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY_FINAL_RETRY` | `true` |

3. Tuning order (to preserve answer quality while reducing degeneration):
  - Lower `UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS` first.
  - Then lower `UNSLOTH_WIKI_RAG_MAX_PAGES`.
  - Then lower `UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE`.
  - Only after that, tighten engine-level caps.

4. Caveat:
  - Over-tightening can improve fluency but hurt factual recall. Balance with your model size and domain complexity.

## April 2026 Addendum (LLM Link Expansion Selector)

This pass adds an optional semantic selector for ranking-time link expansion so traversal can prioritize links by query intent rather than lexical path overlap alone.

### Changed files
- `studio/backend/core/wiki/engine.py`
- `studio/backend/tests/test_wiki_rag_pipeline.py`
- `updates.md`

### Behavior changes
1. Link expansion now supports an LLM selector stage.
  - During `_expand_ranked_pages_by_links`, depth-0 seed pages can invoke a link-selection prompt.
  - The selector chooses from concrete outgoing wikilinks only and is hard-bounded by fanout.
2. Safety and fallback behavior are preserved.
  - If selector output is invalid/empty, expansion falls back to existing lexical sort (`ranked_map` prior + path-overlap).
  - Selector is budgeted to a small number of seed pages per query to avoid runaway LLM calls.
3. Query diagnostics now include selector settings on saved analysis pages.

### New env variables
| Env var | What it controls | Default |
|---|---|---|
| `UNSLOTH_WIKI_ENGINE_RANKING_LINK_LLM_SELECTOR_ENABLED` | Enables semantic LLM link selection in expansion traversal | `true` |
| `UNSLOTH_WIKI_ENGINE_RANKING_LINK_LLM_SELECTOR_MAX_CANDIDATES` | Max outgoing links exposed to selector prompt per source page | `24` |

### Regression coverage added
- `test_rank_pages_link_expansion_can_use_llm_selector`
- `test_rank_pages_link_expansion_llm_selector_falls_back_to_lexical`
