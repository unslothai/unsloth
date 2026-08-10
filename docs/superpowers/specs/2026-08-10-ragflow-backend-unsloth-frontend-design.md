# RAGFlow as Sole Backend, Unsloth Studio as Sole Frontend

**Date:** 2026-08-10
**Status:** design — awaiting approval
**Scope:** full RAGFlow backend surface, Unsloth frontend shell, training-family features disabled

---

## 1. Goal

Run RAGFlow as the only backend and this repository's `studio/frontend/` (the
frontend-only Unsloth Studio fork) as the only user interface.

- RAGFlow's own web UI (`web/`, 1373 files / 222,081 lines) is **not used**.
- RAGFlow's backend is **not forked or modified**. All adaptation happens in the frontend.
- Every RAGFlow backend capability is reachable from the Unsloth shell. Screens
  RAGFlow's UI has and the Unsloth shell lacks get **written** into the Unsloth
  shell, in the Unsloth stack.
- Unsloth's model-training family has no RAGFlow counterpart and is disabled
  behind feature flags (decision (a), see §7).
- Work proceeds phase by phase; each phase leaves a working application.

### Acceptance criteria

1. `docker compose up` in `ragflow-main/docker/` plus `npm run dev` in
   `studio/frontend/` yields a usable application with no RAGFlow UI involved.
2. Every route group in RAGFlow's API is assigned to exactly one phase (§6).
3. `scripts/sync-upstream.sh` still runs with conflicts confined to the file list in §5.3.
4. Each phase passes `npm run typecheck` and `npm run lint`, plus its own smoke test.

---

## 2. Verified facts

Everything in this section was read from the repositories, not assumed.

### 2.1 The two frontends are stack-incompatible

| | Unsloth `studio/frontend/` | RAGFlow `web/` |
|---|---|---|
| React | 19.2.4 | 18.2.0 |
| Router | TanStack Router 1.169.2 (hard-pinned in `overrides`) | react-router |
| HTTP | `authFetch` wrapper | umi-request |
| Build | Vite 8.0.16 | Vite 7.2.7 |
| CSS | Tailwind 4.1.18 | Tailwind 3 |
| State | zustand 5 + dexie 4.3 | zustand 4 + `@tanstack/react-query` 5 |
| Chat UI | `@assistant-ui/react` 0.12.28 + `assistant-stream` 0.3.12 | none |
| Graph editor | none | `@xyflow/react` |
| Size | 1104 files / 248,518 lines | 1373 files / 222,081 lines |

**Consequence:** no RAGFlow screen can be copied. Each one is rewritten against
the Unsloth stack. This is the dominant cost of the project and the reason for
phasing.

### 2.2 RAGFlow serves two API schemes over the same `/api/v1` prefix

`docker/.env:193` currently sets `API_PROXY_SCHEME=python`. `docker/entrypoint.sh:188-206`
picks an nginx config from that value:

| Scheme | nginx conf | `/api/v1/*` upstream | `/api/v1/skills` | `/api/v1/admin` |
|---|---|---|---|---|
| `python` | `ragflow.conf.python` | `127.0.0.1:9380` (Flask) | not served | `:9381` |
| `go` | `ragflow.conf.golang` | `127.0.0.1:9384` (Gin) | `:9384` | `:9383` |
| `hybrid` | `ragflow.conf.hybrid` | `:9380`, with `skills`, `datasets/search`, `chat/completions`, `system/config`, `user/(login\|logout)` peeled off to `:9384` | `:9384` | `:9383` / `:9381` |

Route inventories (unique method+path pairs):

- Python (`api/apps/restful_apis/*.py`, 29 modules, Flask blueprints registered
  at `/api/v1` by `api/apps/__init__.py:363`): **270**
- Go (`internal/router/*.go`): **313**

Groups the Go scheme has and the Python scheme does not:
`/skills` (skill spaces + search config), `/pipelines`, `/settings/enable-admin`,
`/all-models`, `/folders`, `/document/*`, `/chunk/*`.

Groups both serve: `/auth`, `/users`, `/tenants`, `/datasets`, `/documents`,
`/chats`, `/openai/:chat_id/chat/completions`, `/retrieval`, `/searches`,
`/files`, `/memories`, `/messages`, `/providers`, `/models`, `/agents`, `/tasks`,
`/plugin/tools`, `/connectors`, `/mcp/servers`, `/system/*`, `/chat-channels`,
`/langfuse/api-key`, `/compilation-templates`, bot endpoints.

**Decision: `API_PROXY_SCHEME=hybrid`.** It is the only value that serves all
313 Go routes *and* the Python-only handlers, so no phase is blocked by a
missing endpoint. Ports exposed to the host (`docker/docker-compose.yml:49-55`):
`80` (nginx), `9380` (Flask), `9381` (Python admin), `9382` (MCP), `9384` (Go),
`9383` (Go admin).

**The frontend targets port 80 only.** nginx owns the scheme split; the frontend
must never address `:9380`/`:9384` directly, or changing schemes would break it.

### 2.3 Response envelope

`api/utils/api_utils.py:273`:

```json
{ "code": 0, "data": ..., "total_datasets": 47, "message": "..." }
```

`code == 0` means success. Non-zero puts the reason in `message`. Every adapter
call unwraps `.data` and raises on `code != 0`. This is uniform across both schemes.

### 2.4 Retrieval endpoint — the citation source

`POST /api/v1/retrieval` (`api/apps/restful_apis/chunk_api.py`). Request:

```
dataset_ids: string[]      (required)
question: string           (required)
document_ids?: string[]
page?, page_size?
similarity_threshold?      (default 0.2)
vector_similarity_weight?  (default 0.3)
top_k?                     (default 1024)
highlight?: boolean
use_kg?, toc_enhance?, cross_languages?, keyword?
rerank_id?, metadata_condition?
```

Response `data`: `{ total, chunks[], doc_aggs{} }`. The handler renames fields
before returning:

| wire field | source |
|---|---|
| `id` | `chunk_id` |
| `content` | `content_with_weight` |
| `document_id` | `doc_id` |
| `document_keyword` | `docnm_kwd` |
| `dataset_id` | `kb_id` |
| `important_keywords` | `important_kwd` |
| `questions` | `question_kwd` |
| `positions` | `position_int` (via `internal/service/nlp/retrieval.go:385`) |
| `similarity` | scorer output |

`vector` is stripped. Guard: all `dataset_ids` must share one embedding model, or
the call returns `code = DATA_ERROR`, `"Datasets use different embedding models."`

### 2.5 What the frontend's RAG layer requires

From `src/features/rag/types/rag.ts` (verbatim shapes the adapter must emit):

```ts
KnowledgeBase   { id, name, description?, createdAt?, documentCount? }
DocumentStatus  = "pending" | "running" | "completed" | "failed"
RagDocument     { id, filename, status, error?, numChunks?, kbId?, threadId?, projectId?, createdAt? }
UploadedDocument extends RagDocument { sizeBytes?, kbName?, projectName? }
DocumentUploadResult { documentId, jobId, filename }
IndexJob        { id, documentId, status, stage?, progress?, error?, numChunks? }
JobEvent        { type: "progress"|"complete"|"error", stage?, progress?, error?, num_chunks? }
PdfRegion       { pageIndex, pageNumber, x, y, width, height }   // 0..1, top-left origin
PreviewTarget   { documentId, filename, mediaKind: "pdf"|"text", targetPage?, pdfRegions, text? }
RAG_UPLOAD_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.html,.htm"
```

Availability protocol, from `src/features/rag/api/rag-availability.ts` (read in full):

- Store fields: `available` (optimistic seed — **never gate on it**), `reason`, `answered`.
- `isUnavailable() => answered && !available`; `availabilityUnknown() => !answered`.
- Flips to unavailable **only** on a 503 whose `detail` contains the literal
  string `sqlite-vec` (`RAG_UNAVAILABLE_MARKERS = ["sqlite-vec"]`). A bare proxy
  503 is deliberately ignored.
- The KB list is expected to carry `ragAvailable: boolean` / `ragUnavailableReason`.

RAGFlow emits neither the marker nor a `sqlite-vec` 503. **The adapter injects
`ragAvailable: true` into the mapped KB list, and this file is never edited.**

### 2.6 The chat seam is 40 lines wide

`src/features/chat/api/chat-adapter.ts` is 6151 lines. RAG appears in exactly
two places, both inside `buildLocalTokenCountExtras` (lines 1540-1620):

```ts
enabled_tools: [ ...(ragOn ? ["search_knowledge_base"] : []), ... ],
...(ragOn ? { rag_scope: {
  ...(ragEnabled && ragSource.type === "kb" ? { kb_id: ragSource.kbId }
    : { ...(ragEnabled && threadId ? { thread_id: threadId } : {}),
        ...(projectRagEnabled && ragProjectId ? { project_id: ragProjectId } : {}) }),
  default_top_k: ragTopK,
  mode: ragMode,
} } : {}),
```

Everything else in the file — model selection, streaming, tool-call parsing, MCP,
sandbox, artifacts, audio, multi-model compare, thread persistence, token
counting, auto-heal — is RAG-independent.

`src/features/chat/api/chat-api.ts:1210`, `streamChatCompletions`, POSTs to
`/v1/chat/completions`: the standard OpenAI path. External-provider support
already exists client-side (`externalApiKey`, `baseUrl`, gemini/openai/custom).

**Design decision — client-side retrieval.** The frontend calls
`POST /api/v1/retrieval` itself before the completion request, injects the hits
into the prompt, and streams the completion from the configured LLM provider.

Rejected: routing chat through `POST /api/v1/openai/:chat_id/chat/completions`.
That endpoint requires a pre-created chat assistant (`internal/handler/openai_chat.go`
errors with `CodeDataError` + `"You don't own the chat "` when `chat_id` is empty)
and its server-side assembly bypasses the frontend's tool loop, killing
`web_search`, `python`, `terminal`, `render_html`, MCP, auto-heal, and token
counting — roughly 6000 of `chat-adapter.ts`'s 6151 lines. Client-side retrieval
keeps all of it and touches the two fields above.

### 2.7 Citations

`src/components/assistant-ui/rag-sources.tsx` (read in full) scans
`message.content` for `part.type === "tool-call" && part.toolName === "search_knowledge_base"`,
runs `parseCitations(part.result)`, dedups to the best-scoring chunk per
`documentId ?? filename`, and renders `CitationBadge`. `parseCitations` reads a
`__RAG_SOURCES__:` sentinel followed by JSON rows into
`Citation { id, filename, page, score, text, documentId, chunkId }`.

**The adapter synthesizes that tool-call part** from the `/retrieval` response.
`rag-sources.tsx`, `citation-utils.ts`, `CitationBadge`, and the PDF highlight
path therefore need **zero changes**.

Mapping:

| `Citation` | from `/retrieval` chunk |
|---|---|
| `id` | `id` |
| `chunkId` | `id` |
| `documentId` | `document_id` |
| `filename` | `document_keyword` |
| `text` | `content` |
| `score` | `similarity` |
| `page` | first `positions[*][0]` |

### 2.8 No SSE for document jobs

RAGFlow has no server-sent-events endpoint for parse progress; SSE exists only
for agent runs. Document state arrives from `GET /api/v1/datasets/:id/documents`
as `run` / `progress` / `progress_msg`.

`streamJobEvents`'s signature is preserved and reimplemented as a 2-second poll
that synthesizes `JobEvent`s, so callers do not change:

| RAGFlow `run` | `DocumentStatus` | `JobEvent.type` |
|---|---|---|
| `UNSTART` / `0` | `pending` | `progress` |
| `RUNNING` / `1` | `running` | `progress` |
| `DONE` / `3` | `completed` | `complete` |
| `FAIL` / `4` | `failed` | `error` |

`progress` (0..1) maps to `JobEvent.progress`; `progress_msg` to `stage`.
Polling stops on `complete` or `error`.

### 2.9 Thread and project scoping do not exist in RAGFlow

RAGFlow has datasets only. The frontend scopes RAG by KB **or** by
`(thread_id, project_id)`. Resolution: a dexie table maps
`threadId -> datasetId` and `projectId -> datasetId`, auto-creating a dataset
named `thread-<id>` / `project-<id>` on first upload. `rag_scope` in the payload
keeps its shape; the adapter resolves it to `dataset_ids` before calling `/retrieval`.

### 2.10 Auth

`POST /api/v1/auth/login` (`api/apps/restful_apis/user_api.py:61`) returns
`user.to_safe_dict(for_self=True)` and sets the token via
`construct_response(..., auth=user.get_id())` — an `Authorization` response
header. Also present: `/auth/logout`, `/auth/login/channels`,
`/auth/login/:channel`, `/auth/oauth/:channel/callback`, `/auth/password/forgot/*`,
`/auth/password/reset`, `/users/me` (GET/PATCH), `POST /users`,
`/users/me/models` (GET/PATCH).

`src/features/auth/api.ts` currently has `redirectToAuth()` beginning with a bare
`return;` under the comment *"TEMP (local dev, backend not attached): a 401/403 no
longer bounces the app to /login."* **Phase 1 restores it.**

### 2.11 Bot endpoints authenticate differently

`internal/router/router.go:202-230` defines `apiBetaAuth`, holding `/searchbots`,
`/chatbots`, `/agentbots`, and `/searchbots/detail`. These take an **SDK beta
token**, not a session cookie. Python parity exists
(`api/apps/restful_apis/bot_api.py`: `/chatbots/:dialog_id/completions`,
`/chatbots/:dialog_id/info`, `/agentbots/:agent_id/completions`,
`/agentbots/:agent_id/inputs`, `/agentbots/:shared_id/logs/:message_id`,
`/searchbots/ask`). Phase 13 handles them with a separate token-bearing client,
never the session `authFetch`.

---

## 3. Architecture

```
┌─────────────────────────────────────────────┐
│  studio/frontend  (Unsloth shell, Vite dev) │
│                                              │
│  features/chat ......... unchanged core      │
│  features/rag .......... retargeted          │
│  features/ragflow/ ..... NEW: client+mapping │
│  features/datasets/ .... NEW screens         │
│  features/agent-canvas/  NEW screens         │
│  ... (see §5.2)                              │
└───────────────┬─────────────────────────────┘
                │  /api/v1/*  and  /v1/*
                ▼
┌─────────────────────────────────────────────┐
│  nginx :80   (ragflow.conf.hybrid)          │
│    /api/v1/skills          -> :9384 (Go)    │
│    /api/v1/datasets/search -> :9384         │
│    /api/v1/chat/completions-> :9384         │
│    /api/v1/admin/*         -> :9383 / :9381 │
│    /(v1|api)               -> :9380 (Flask) │
└───────────────┬─────────────────────────────┘
                ▼
   Flask :9380 · Go :9384 · admin :9381/:9383 · MCP :9382
                ▼
   MySQL · Redis · MinIO · Elasticsearch · NATS · TEI
```

### 3.1 The single adapter boundary

`src/features/ragflow/client.ts` is the only module that knows RAGFlow's envelope:

```ts
// Unwraps { code, data, message }. Raises RagflowError on code !== 0.
async function ragflowFetch<T>(path: string, init?: RequestInit): Promise<T>
```

`src/features/ragflow/mapping.ts` holds every field translation in §2.4, §2.7,
§2.8. `src/features/ragflow/types.ts` holds RAGFlow's wire types, kept separate
from the frontend's own types so the two never blur.

Rule: no other file constructs a RAGFlow URL or reads `.code`. Violations are
what make the next upstream sync painful.

---

## 4. LLM provider

Chat completions do **not** go through RAGFlow. The user configures an LLM
provider in settings, exactly as the existing external-provider support already
allows (`chat-adapter.ts:3863` `isExternalRequest`, `externalApiKey`, `baseUrl`,
gemini/openai/custom provider types).

Rationale: RAGFlow's own completion endpoints do their own prompt assembly and
require server-side state — `POST /api/v1/openai/:chat_id/chat/completions` needs
a pre-created chat assistant (§2.6), and `POST /api/v1/chat/completions`
(`api/apps/restful_apis/chat_api.py:1223`, Go `chatSessionHandler.ChatCompletions`)
is session-scoped the same way. Either one bypasses the frontend's tool loop,
killing `web_search`, `python`, `terminal`, `render_html`, MCP, auto-heal, and
token counting — roughly 6000 of `chat-adapter.ts`'s 6151 lines.

### 4.1 Path collision — must be resolved in Phase 1

`streamChatCompletions` (`chat-api.ts:1210`) currently posts to
`/v1/chat/completions`. nginx matches `location ~ ^/(v1|api)` and forwards
**everything** under `/v1` to RAGFlow, which serves `/api/v1/chat/completions`
but not `/v1/chat/completions`. Left alone, every chat request 404s.

Resolution: the Vite dev proxy forwards only `/api` to nginx. `/v1` is **not**
proxied; `streamChatCompletions` is pointed at the configured provider's
`baseUrl` instead. This is the one edit `chat-api.ts` needs, contrary to what a
first reading of §5.3 suggests.

RAGFlow's `/providers` and `/models` surfaces are still used — for the
**embedding and rerank** models that datasets require, and to populate the model
picker's list. Phase 5 wires this.

---

## 5. Change surface

### 5.1 Sync constraint

`scripts/sync-upstream.sh:43` sets `KEEP_PREFIX="studio/frontend/"`. Everything
outside it that upstream sends is discarded. Two consequences:

- Files this project **adds** under `studio/frontend/` are sync-safe: upstream
  does not know them, so it never conflicts on them.
- Files this project **edits** that upstream also edits are the only conflict risk.

The design therefore puts all new logic in new directories and keeps edits to
existing files as small as possible.

### 5.2 New directories (no upstream counterpart)

```
src/features/ragflow/          client.ts · mapping.ts · types.ts
src/features/datasets/         dataset CRUD, parser config, chunk editing
src/features/knowledge-graph/  graph view, structure graph, mindmap
src/features/agent-canvas/     agent flow editor, versions, sessions, debug
src/features/connectors/       connector CRUD, logs, rebuild
src/features/memory/           memories + messages
src/features/skills/           skill spaces, skill search config
src/features/search-app/       search apps and completions
src/features/files/            file manager, folders, commits
src/features/providers/        provider + model administration
src/features/admin/            admin console, tenants, system status
src/features/bots/             bot embed config (SDK beta token client)
```

### 5.3 Existing files edited — the complete list

| File | Edit |
|---|---|
| `src/features/rag/api/rag-api.ts` | retarget to `features/ragflow/client.ts` |
| `src/features/chat/api/chat-api.ts` | point `streamChatCompletions` at the provider `baseUrl` instead of `/v1/chat/completions` (§4.1) |
| `src/features/chat/api/chat-adapter.ts` | resolve `rag_scope` to `dataset_ids`; call `/retrieval`; synthesize the `search_knowledge_base` part |
| `src/config/env.ts` | RAGFlow base URL |
| `src/features/auth/api.ts` | remove the TEMP `return;` in `redirectToAuth()` |
| `src/config/disabled-features.ts` | add the flags in §7 |
| `src/components/app-sidebar.tsx` | add nav entries for new screens; extend the existing flag gate at `:980` and `:1208` — **highest conflict risk** |
| `src/features/settings/tabs/data-tab.tsx` | point at RAGFlow datasets |
| `src/features/settings/components/sidebar-nav-customizer.tsx` | extend the flag gate at `:123` |
| `vite.config.ts` | proxy `/api` to `127.0.0.1:80`; **do not** proxy `/v1` (§4.1) |
| `src/app/routes/hub.tsx` `export.tsx` `video.tsx` `data-recipes.tsx` `data-recipes.$recipeId.tsx` | add the redirect gate — these have **no** `FEATURE_*` check today |
| `src/app/routes/*.tsx` (new files) | routes for the §5.2 screens (added, not edited) |

Explicitly **not** edited: `rag-availability.ts`, `rag-sources.tsx`,
`citation-utils.ts`, `tool-ui-knowledge-base.tsx`, `features/rag/types/rag.ts`.

### 5.4 Backend

RAGFlow is not modified. Only `docker/.env` changes:

```
API_PROXY_SCHEME=hybrid      # was: python
DOC_ENGINE=elasticsearch     # unchanged default
```

---

## 6. Phases

Each phase ends green: `npm run typecheck`, `npm run lint`, and its smoke test.

### Phase 1 — Boot, auth, feature gating
`docker/.env` to `hybrid`; `vite.config.ts` proxies `/api` to `:80` and drops the
`/v1` proxy (§4.1); `features/ragflow/client.ts`; restore `redirectToAuth()`;
point `streamChatCompletions` at the provider `baseUrl`; wire `/auth/login`,
`/auth/logout`, `/users/me`, `/system/healthz`, `/system/version`,
`/system/config`; add the §7 flags and the five missing route gates.
**Smoke:** log in, reload, session survives, logout returns to `/login`; a chat
message streams from the configured provider; navigating directly to `/hub`,
`/export`, `/video`, `/data-recipes` redirects to `/chat`.

### Phase 2 — Knowledge bases and documents
`/datasets` CRUD → `KnowledgeBase`; `/datasets/:id/documents` list/upload/delete
→ `RagDocument`; `/documents/upload`; poll-based `streamJobEvents` (§2.8); inject
`ragAvailable: true`; thread/project dataset mapping (§2.9).
**Smoke:** create a KB, upload a PDF, watch progress reach `completed`, see the
document listed with its chunk count.

### Phase 3 — Retrieval, citations, preview
`POST /api/v1/retrieval` from the adapter; synthesize the
`search_knowledge_base` tool-call part with `__RAG_SOURCES__:`; map `positions`
to `PdfRegion` (0..1). Preview: `rag-api.ts:312` calls
`/documents/:id/preview-target` and `:321` calls `/documents/:id/file-url`;
both are mapped onto RAGFlow's `GET /api/v1/documents/:doc_id/preview`, with
`/documents/images/:image_id` and `/thumbnails` for figure and thumbnail assets.
`preview-target` has no RAGFlow counterpart and is assembled client-side from the
retrieval hit already in hand.
**Smoke:** ask a question about the uploaded PDF; a citation badge appears;
clicking it opens the PDF at the highlighted region.

**End of Phase 3 the system is usable.** Later phases add breadth.

### Phase 4 — Dataset management
Parser config, chunk method, `/datasets/:id/chunks` CRUD,
`/datasets/:id/documents/:id/chunks` CRUD/PATCH, `/datasets/tags/aggregation`,
`/datasets/metadata/flattened`, `/datasets/:id/metadata/*`,
`/datasets/:id/ingestions/summary`, `/tasks/:id` + `/tasks/:id/cancel`.

### Phase 5 — Providers and models
`/providers` CRUD, `/providers/:id/models`, `/providers/:id/instances`,
`/providers/:id/connection`, `/models`, `/models/default` GET/PATCH,
`/all-models`, `/users/me/models`. Feeds the model picker and the embedding/rerank
selectors datasets need.

### Phase 6 — Files
`/files` CRUD, `/files/move`, `/files/:id`, `/files/:id/parent`,
`/files/:id/ancestors`, `/folders`, `/workspace`, `/v1/file`, file commits,
`/documents/artifact/:filename`, `/document/list`, `/document/metadata/summary`,
`/document/set_meta`, `/chunk/list`.

### Phase 7 — Memory
`/memories` CRUD, `/memories/:id/config`, `/messages` POST/DELETE.

### Phase 8 — Search applications
`/searches` CRUD, `/searches/:id/completion`, `/searches/:id/completions`,
`/datasets/search`.

### Phase 9 — Knowledge graph and structure
`/datasets/:id/documents/:id/structure/graph` GET/DELETE, mindmap,
`/compilation-templates/builtins`, `/compilation-templates/wiki-presets`,
compilation template groups, `/retrieval` with `use_kg: true`.

### Phase 10 — Skills (Go scheme only)
`/skills/spaces` CRUD, `/skills/space/by-folder`, `/skills/config` GET/POST.
Depends on `hybrid` routing `skills` to `:9384`.

### Phase 11 — Connectors, MCP, plugins
`/connectors` CRUD, `/connectors/:id/logs`, `/connectors/:id/rebuild`,
`/mcp/servers` CRUD + `/import`, `/plugin/tools`, `/langfuse/api-key` CRUD.

### Phase 12 — Agent canvas
Full surface from `internal/router/agent_routes.go` and
`api/apps/restful_apis/agent_api.py`:
`/agents` list/create, `/agents/templates`, `/agents/prompts`, `/agents/tags`,
`/agents/download`, `/agents/:id` GET/PUT/DELETE, `/agents/:id/run`,
`/agents/:id/publish`, `/agents/:id/reset`, `/agents/:id/upload`,
`/agents/:id/tags` PUT, `/agents/:id/versions` + `/versions/:version_id`
GET/DELETE, `/agents/:id/sessions` list/create/get/delete,
`/agents/:id/components/:cid/input-form`, `/agents/:id/components/:cid/debug`,
`/agents/attachments/:id/download`, `/agents/attachments/:id/preview`,
`/components`, `/pipelines` + `/pipelines/:id` (Go scheme, unauthenticated).
**Largest phase.** RAGFlow's own agent UI is 39,021 lines and the Unsloth stack
has no node editor; a graph library must be chosen here (see risk R1).

### Phase 13 — Admin, tenants, bots, channels
`/admin/*` (nginx sends these to `:9383`/`:9381`), `/tenants` CRUD +
`/tenants/:id/users`, `/tenant`, `/settings/enable-admin`, `/system/status`,
`/system/stats`, `/system/tokens`, `/chat-channels` CRUD +
`/chat-channels/:id/runtime`, `/chats` + `/chats/:id/sessions`,
`/chat/*` (audio, recommendation), `/dify/*`, `/aimlapi/*`.
Bot embeds via a separate SDK-beta-token client (§2.11): `/chatbots/:id/completions`,
`/chatbots/:id/info`, `/agentbots/:id/completions`, `/agentbots/:id/inputs`,
`/agentbots/:shared_id/logs/:message_id`, `/searchbots/ask`, `/searchbots/detail`.

---

## 7. Disabled Unsloth features

Decision (a): the model-training family is disabled behind flags. RAGFlow is a
retrieval platform; it does not train LoRAs, download HuggingFace checkpoints,
export GGUF, generate images or video, or probe local hardware. There is no
endpoint to point these screens at.

`src/config/disabled-features.ts` today exports exactly two flags:

```ts
export const FEATURE_IMAGES = false;
export const FEATURE_TRAIN = false;
```

`FEATURE_TRAIN` already gates `src/app/routes/studio.tsx:22`,
`data-tab.tsx:107,496`, `sidebar-nav-customizer.tsx:123`, and
`app-sidebar.tsx:980,1208` — where `:980` covers the studio, recipes, **and**
export routes together. It is reused, not replaced.

Flags after this change:

| Flag | Feature | Lines | Status |
|---|---|---|---|
| `FEATURE_TRAIN` | training UI, recipes, export (existing flag, widened) | 11,725 + 29,364 + 4,485 | exists, `false` |
| `FEATURE_IMAGES` | image generation | 8,426 | exists, `false` |
| `FEATURE_HUB` | HuggingFace model browsing/download | 31,372 | **new** |
| `FEATURE_TRAINING` | training runs | 10,496 | **new** |
| `FEATURE_VIDEO` | video | 4,010 | **new** |
| `FEATURE_LOCAL_MODELS` | local model load/unload | 1,950 | **new** |
| `FEATURE_TRAIN_MODEL_PICKER` | training model picker | 1,470 | **new** |
| `FEATURE_TRANSFORMERS_UPGRADE` | transformers upgrade prompt | 406 | **new** |
| **Total gated** | | **~103,700** | |

Gating copies the existing pattern: route-level
`throw redirect({ to: "/chat" })` as at `src/app/routes/studio.tsx:22` and
`images.tsx:25`, plus sidebar suppression at `app-sidebar.tsx:1208` and
`sidebar-nav-customizer.tsx:123`.

**Verified gap:** `hub.tsx`, `export.tsx`, `video.tsx`, `data-recipes.tsx`, and
`data-recipes.$recipeId.tsx` contain **no** `FEATURE_*` reference today — they are
reachable by direct URL even with the sidebar row hidden. Phase 1 adds the
redirect gate to each. Only `__root.tsx`, `images.tsx`, and `studio.tsx` are
gated as things stand.

Code is **not deleted** — flags keep upstream sync clean and let any of these
return if a training backend is ever added.

`features/model-picker` (14,633 lines) is **kept**: model *selection* is fed by
RAGFlow's `/models`. Only its local-loading paths are gated by
`FEATURE_LOCAL_MODELS`.

### Retained Unsloth features

`features/chat` (49,976 — in full), `features/settings` (18,073 — retargeted),
`features/model-picker` (selection), `features/rag` (3,459 — retargeted),
`features/profile`, `features/api-monitor`, `features/onboarding`,
`features/auth`, `features/security`, `features/dataset-picker`,
`features/tour`, `features/deep-links`, `features/native-intents`,
`features/hf-auth`.

---

## 8. Risks

**R1 — Agent canvas size (high).** Phase 12 replaces 39,021 lines of
`@xyflow/react`-based UI. The Unsloth stack has no node editor, so a library
must be added — the only new runtime dependency this design anticipates. Phase 12
begins with a spike: render one read-only agent graph before committing to the
full editor.

**R2 — `positions` normalization (medium).** `position_int` is a pixel-space
tuple; `PdfRegion` is 0..1. Page dimensions may not be in the retrieval response,
in which case they must come from `pdf.js` at render time. Phase 3 verifies this
against a real PDF before the mapping is considered done.

**R3 — `app-sidebar.tsx` conflicts (medium).** Every phase adds nav entries to a
file upstream also edits. Mitigation: all new entries go in one contiguous block
behind a single marker comment, so a conflict is one hunk rather than a dozen.

**R4 — Embedding-model uniformity (medium).** `/retrieval` rejects
`dataset_ids` spanning different embedding models (§2.4). Multi-KB retrieval must
either enforce one embedding model per workspace or group calls by model and
merge results. Phase 3 picks one; Phase 4 surfaces the constraint in the UI.

**R5 — Tenant model (medium).** RAGFlow scopes everything by tenant. The Unsloth
shell has no tenant concept. Phase 1 assumes single-tenant (the logged-in user's
own tenant) and Phase 13 revisits it when the admin console lands.

**R6 — Scheme drift (low).** `hybrid` moves five path prefixes to the Go server.
If a future RAGFlow release changes that split, endpoints move. Mitigation:
the frontend addresses port 80 only and never a scheme-specific port.

---

## 9. Verification

Per phase: `npm run typecheck`, `npm run lint`, and the phase's smoke test.

End-to-end gate after Phase 3, run manually:

1. Log in.
2. Create a knowledge base.
3. Upload a PDF; progress reaches `completed` with a chunk count.
4. Ask a question whose answer is in that PDF.
5. A citation badge appears under the answer.
6. Clicking the badge opens the PDF at the highlighted region.

If any step fails, that phase is not done.

Sync gate, run once after Phase 3 and again after Phase 13:
`scripts/sync-upstream.sh --dry-run` — conflicts must stay inside the §5.3 list.
