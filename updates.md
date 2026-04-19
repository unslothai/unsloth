# Phase 3 RAG Updates

## Scope
This document summarizes the backend RAG changes made to unblock end-to-end wiki retrieval for GGUF chat, and adds practical debugging and cleanup workflows.

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
- `UNSLOTH_WIKI_RAG_MAX_PAGES` (default: `3`)
- `UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE` (default: `900`)
- `UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS` (default: `4500`)
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

# Include prior analysis pages when answering wiki queries (default false)
UNSLOTH_WIKI_ENGINE_INCLUDE_ANALYSIS_IN_QUERY=false
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
- Wiki query ranking now excludes `index.md` and `log.md` and excludes `analysis/*` by default, which reduces recursive low-quality analysis loops.
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

#### 1) Recursive retrieval expansion (optional)
Wiki ranking can expand through wiki links to configurable depth/fanout:
- `UNSLOTH_WIKI_ENGINE_RANKING_LINK_DEPTH`
- `UNSLOTH_WIKI_ENGINE_RANKING_LINK_FANOUT`

Default depth remains `0` (disabled), so recursion is opt-in.

#### 1b) Hybrid ranking signals (now default)
`_rank_pages` now combines multiple signals instead of relying only on lexical overlap:
- token overlap in page content
- overlap with path/slug tokens
- overlap with title/header tokens
- phrase matches (quoted phrases and compact query phrases)
- entity-intent boosts for prompts like "who is ..."

If lexical scoring produces no candidates, ranking now falls back to recency ordering instead of returning an empty list.

This improves robustness for sparse corpora, new entities, and short biography-style prompts.

#### 1c) Optional LLM reranking stage (same backend/model path)
An optional reranker can now reorder top retrieval candidates using the same `llm_fn` path already used by the wiki engine (so it uses the same model server/backend currently active).

Controls:
- `UNSLOTH_WIKI_ENGINE_LLM_RERANK_ENABLED` (default: `false`)
- `UNSLOTH_WIKI_ENGINE_LLM_RERANK_CANDIDATES` (default: `24`)
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

# Optional: recursive link expansion (off by default)
UNSLOTH_WIKI_ENGINE_RANKING_LINK_DEPTH=0
UNSLOTH_WIKI_ENGINE_RANKING_LINK_FANOUT=4

# Optional: LLM reranking (same wiki llm_fn backend)
UNSLOTH_WIKI_ENGINE_LLM_RERANK_ENABLED=false
UNSLOTH_WIKI_ENGINE_LLM_RERANK_CANDIDATES=24
UNSLOTH_WIKI_ENGINE_LLM_RERANK_TOP_N=12
UNSLOTH_WIKI_ENGINE_LLM_RERANK_PREVIEW_CHARS=420

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
