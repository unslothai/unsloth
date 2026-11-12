# Subagents

Subagents for Studio chats that run on an **external provider**
(OpenRouter and every other `ExternalProviderClient` connection). The main chat's model gets
a `spawn_subagent` tool. Each call starts a child agent that runs the same Studio tool loop
(`core/inference/studio_tool_loop.py`) with its own short conversation, then returns its final
answer to the main agent as the tool result. Several `spawn_subagent` calls in one assistant
turn run **in parallel**. Every subagent is listed live in a side panel of the parent chat and
is saved as its own (archived) chat thread.

## Behaviour

- Depth is 1: a subagent never gets `spawn_subagent`.
- A subagent inherits the parent's permission mode, sandbox level, website policy, RAG scope,
  tool timeout and tool budget. It can only use a subset of the parent's enabled tools.
- It shares the parent loop's live permissions: a Full access grant made in the chat or on any
  subagent's approval covers the parent and all of its subagents for the rest of that run.
- It shares the parent's `session_id`, so it works in the same sandbox / project folder and
  its approval prompts resolve through the existing `POST /api/inference/tool-confirm`.
- Optional per-subagent `model` override on the same provider connection (e.g. any OpenRouter
  model id). Default: the parent's model.
- Parallel limit: `UNSLOTH_SUBAGENT_MAX_PARALLEL` (default 4).
- Stopping the parent chat stops its running subagents; a single subagent can be stopped from
  the panel.

## Tool (model-facing)

`spawn_subagent(description: str, prompt: str, tools?: string[], model?: str)`

- `description`: 3–6 word label shown in the panel.
- `prompt`: self-contained brief; the subagent sees nothing else from the parent chat.
- `tools`: optional subset of the parent's enabled tool names; default = all of them minus
  `spawn_subagent`.
- `model`: optional model id on the same connection.

Enabled by the client sending `"spawn_subagent"` in `enabled_tools`; the chat UI does this when
Settings → Chat → Thinking and tools → Subagents is on (off by default) and the chat runs Studio's tool loop on an
external provider. Local GGUF/MLX models and managed engines are not covered yet.

## Backend modules

| File | Purpose |
| --- | --- |
| `backend/core/inference/subagents.py` | Tool schema, in-memory registry, child runner, persistence calls |
| `backend/core/inference/studio_tool_loop.py` | Parallel start + execution branch for `spawn_subagent` |
| `backend/routes/inference.py` | Builds the child transport factory for external providers |
| `backend/storage/subagents_db.py` | Persisted subagent records |
| `backend/routes/subagents.py` | REST API below |

### Registry API (`core/inference/subagents.py`, for routes)

```python
def live_subagent(subagent_id: str) -> dict | None
    # Full record (see "Record") of a run still held in memory, for the current account; else None.
def live_subagents_for_parent(parent_thread_id: str) -> list[dict]
    # Summary records (no "transcript") of in-memory runs for that parent, current account only.
def cancel_subagent(subagent_id: str) -> bool
    # Sets the run's cancel event. False when unknown / not running / another account's.
```

### Storage API (`storage/subagents_db.py`)

Per-account like the other Studio stores (mirror `storage/scheduled_tasks_db.py` /
`storage/studio_db.py` for connection, account scoping and table creation).

```python
def upsert_subagent(record: dict) -> None      # insert or replace by record["id"]
def get_subagent(subagent_id: str) -> dict | None   # full record incl. "transcript"
def list_subagents(parent_thread_id: str) -> list[dict]  # summaries (no "transcript"), oldest first
def delete_subagents_for_parent(parent_thread_id: str) -> int
```

### Record

```jsonc
{
  "id": "sa_…",                     // str
  "parentThreadId": "…",            // str | null (chat thread that spawned it)
  "childThreadId": "…",             // str (the saved, archived chat thread of the subagent)
  "sessionId": "…",                 // str | null
  "description": "Inspect backend", // str
  "prompt": "…",                    // str
  "model": "openai/gpt-5.2",        // str | null
  "tools": ["web_search", "python"],// list[str]
  "status": "running",              // "running" | "completed" | "failed" | "cancelled"
  "error": null,                    // str | null
  "resultPreview": "…",             // str | null (first ~600 chars of the final answer)
  "toolCallCount": 3,               // int
  "pendingApproval": null,          // {"approvalId","toolName","toolCallId","arguments"} | null (live only)
  "startedAt": 1731300000000,       // int ms
  "finishedAt": null,               // int ms | null
  "transcript": [ /* TranscriptItem, only on detail reads */ ]
}
```

`TranscriptItem` (ordered):

```jsonc
{"kind": "text", "text": "…"}                         // assistant prose (coalesced)
{"kind": "reasoning", "text": "…"}                    // thinking (coalesced)
{"kind": "tool", "toolCallId": "…", "toolName": "…", "arguments": {…},
 "status": "running" | "awaiting_approval" | "done", "approvalId": "…" | null,
 "result": "…" | null}                                 // result truncated to ~4000 chars
{"kind": "status", "text": "…"}                        // transient tool badge text
{"kind": "error", "text": "…"}
```

### REST API (`routes/subagents.py`, mounted at `/api/subagents`, authenticated)

| Method | Path | Response |
| --- | --- | --- |
| GET | `/api/subagents?parent_thread_id=X` | `{"subagents": [Record without transcript]}` — live runs merged over stored rows (live wins), oldest first |
| GET | `/api/subagents/{id}` | `Record` with `transcript` (live first, else stored); 404 if unknown |
| POST | `/api/subagents/{id}/cancel` | `{"cancelled": bool}` |

Approvals for a subagent's tool call: `POST /api/inference/tool-confirm` with
`{"session_id": record.sessionId, "approval_id": item.approvalId, "decision": "allow"|"deny"}`.
