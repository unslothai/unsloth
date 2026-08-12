# ADR 0003 — A Project is a platform Chat, a Thread is a platform Session, and the platform owns message history; a per-message write does not exist and a per-message delete removes a turn pair

* Status: Accepted
* Date: 2026-08-12
* Scope: Faz 0B fixes the mapping; Faz 7 implements it. `features/chat`, `features/rag`, `src/integrations/platform-backend`
* Supersedes: nothing. Superseded by: nothing.

## Context

Plan §3.2 proposes an entity mapping and then requires it to be settled here:
"Bu eşleme Faz 0 sonunda ADR olarak kesinleştirilmelidir. Özellikle Project →
Chat ve Thread → Session kararı daha sonra sessizce değiştirilmemelidir."

The proposal is sound, but it is written at the level of entity names. The
backend's actual storage shape imposes constraints that the name-level mapping
hides, and three of them invalidate frontend operations that currently work.
Everything below was read from `/Users/baran/Desktop/rag-backend`.

### The two platform entities

`api/db/db_models.py` `class Dialog` — the platform's word for a chat
assistant. Fields that matter to the mapping: `tenant_id`, `name`,
`description`, `icon`, `language`, `llm_id`, `tenant_llm_id`, `llm_setting`,
`prompt_type`, `prompt_config`, `kb_ids`, `similarity_threshold`,
`vector_similarity_weight`, `top_n`, `top_k`, `rerank_id`, `meta_data_filter`,
`do_refer`, `status`.

`api/db/db_models.py` `class Conversation` — the platform's word for a chat
session. Its entire field set is `id`, `dialog_id`, `name`, `message`
(JSONField), `reference` (JSONField, default `[]`), `user_id`.

The REST layer renames both on the way out (`chat_api.py:182-186`):

```python
def _build_session_response(conv: dict) -> dict:
    conv = dict(conv)
    conv["chat_id"] = conv.pop("dialog_id", conv.get("chat_id"))
    conv["messages"] = conv.pop("message", conv.get("messages", []))
    return conv
```

and `_build_chat_response` (`:125-131`) renames `kb_ids` → `dataset_ids` and
adds a resolved `kb_names`. So the wire vocabulary is `chat` / `session` /
`dataset_ids`, and the storage vocabulary is `dialog` / `conversation` /
`kb_ids`. The adapter layer speaks the wire names; nothing in the frontend
should ever see `dialog_id`.

### The frontend entities

`features/chat/types.ts:16-25` `ProjectRecord`: `id`, `name`, `instructions`,
`rootPath`, `sandboxPath`, `archived`, `createdAt`, `updatedAt`.

`:27-64` `ThreadRecord`: `id`, `title`, `modelType`, `modelId`, `pairId`,
`projectId`, `archived`, `createdAt`, `updatedAt`,
`openaiCodeExecContainerId`, `anthropicCodeExecContainerId`,
`forkedFromThreadId`, `forkedFromMessageId`.

`:66-75` `MessageRecord`: `id`, `threadId`, `parentId`, `role`, `content`,
`attachments`, `metadata`, `createdAt`.

`ChatView` (`:3-14`) already has a `mode: "project"` variant and carries
`projectId` on `single` and `compare`, so project scope is a first-class UI
concept, not an add-on.

### Constraint 1 — messages are strictly paired, and one delete removes two

`delete_session_message` (`chat_api.py:990-1012`):

```python
for i, msg in enumerate(conv["message"]):
    if msg_id != msg.get("id", ""): continue
    assert conv["message"][i + 1]["id"] == msg_id
    conv["message"].pop(i)
    conv["message"].pop(i)
    ref_index = (i - 1) // 2
    conv["reference"].pop(ref_index)
    break
```

The user message and the assistant message that answered it **share one id**,
sit adjacent, and are deleted together. `reference` is indexed at `(i-1)//2`,
i.e. one entry per assistant turn, not one per message.

`MessageRecord` is a flat list with `id` and `parentId` — a tree. A tree cannot
round-trip through a paired array: any structure where a user message has two
assistant replies, or an assistant message has no user message before it, is
unrepresentable.

### Constraint 2 — there is no per-message write

The platform's only writes to `Conversation.message` are:

| Route | Effect |
|---|---|
| `POST /api/v1/chats/<chat_id>/sessions` | seeds the opener from the assistant's prologue |
| `POST /api/v1/chat/completions` | appends the user turn, then the assistant turn |
| `DELETE …/messages/<msg_id>` | removes the pair |
| `PUT …/messages/<msg_id>/feedback` | sets `thumbup` / `feedback` on an assistant message |

And `update_session` (`chat_api.py:907-934`) refuses history writes outright:

```python
if "message" in req or "messages" in req:
    return get_data_error_result(message="`messages` cannot be changed")
if "reference" in req:
    return get_data_error_result(message="`reference` cannot be changed")
```

So `saveChatMessage` (`PUT /api/chat/threads/{t}/messages/{m}`) and
`syncChatMessages` (`PUT /api/chat/threads/{t}/messages`) have **no platform
path at all**. History is append-only through generation.

### Constraint 3 — the assistant history is written by the server during streaming

`structure_answer` (`api/db/services/conversation_service.py:188-228`) appends
and then mutates the assistant message on every chunk:

```python
if not conv.message or conv.message[-1].get("role", "") != "assistant":
    conv.message.append({"role": "assistant", "content": content, "created_at": time.time(), "id": message_id})
```

and `session_completion` pre-appends the reference slot for the turn
(`chat_api.py:1294-1297`):

```python
conv.reference = [r for r in conv.reference if r]
conv.reference.append({"chunks": [], "doc_aggs": []})
```

The server is therefore the writer of record during a stream. A client that
also persists the streamed text produces two histories that can disagree.

### Constraint 4 — chat creation demands a model; a Studio project never had one

`create` (`chat_api.py:408-495`) requires a resolvable chat model, falling back
to `tenant.tenant_llm_id`, then validating the pair. `session_completion` raises
`LookupError("No default chat model for tenant.")` when neither the chat nor the
tenant has one — which is exactly what `fixtures/stream.json` recorded.

`create` also rejects duplicate names per tenant:

```python
if DialogService.query(name=req["name"], tenant_id=current_user.id, status=StatusEnum.VALID.value):
    return get_data_error_result(message="duplicated chat name in creating chat")
```

and `_validate_name` (`:148-161`) caps the name at 255 **UTF-8 bytes**.
`ProjectRecord.name` has neither constraint today.

### Constraint 5 — thread-scoped retrieval is the frontend's default and the platform has no such scope

`features/chat/stores/chat-runtime-store.ts:102-106`:

```js
export type RagSource = { type: "thread" } | { type: "kb"; kbId: string };
export const DEFAULT_RAG_SOURCE: RagSource = { type: "thread" };
```

The platform's retrieval scope is always a dataset list.
`POST /api/v1/retrieval` (`chunk_api.py:331-337`) opens with

```python
if not req.get("dataset_ids"):
    return get_error_data_result("`dataset_ids` is required.")
```

and a chat's scope is `kb_ids` / `dataset_ids`. There is no session-scoped or
thread-scoped corpus. So the frontend's *default* retrieval source is the one
scope the platform does not model. Resolved in ADR 0004 decision D4.

### Constraint 6 — no job entity, and the status vocabulary is wider than the frontend's

`features/rag/api/rag-api.ts:242-252` reads `GET /jobs/{id}` and streams
`GET /jobs/{id}/events` as SSE, into `JobEvent` with
`type: "progress" | "complete" | "error"`.

The platform has no `/jobs` route in `document_api.py` — parse progress lives on
the document row itself (`db_models.py:909-917`): `progress` (float),
`progress_msg` (text), `run` (char). `run` takes six values
(`common/constants.py:106-112`):

```python
UNSTART = "0"; RUNNING = "1"; CANCEL = "2"; DONE = "3"; FAIL = "4"; SCHEDULE = "5"
```

against the frontend's four-valued `DocumentStatus`
(`features/rag/types/rag.ts:13`): `pending | running | completed | failed`.
`CANCEL` and `SCHEDULE` have no frontend member.

### What the platform gives back for citations

`get_session` (`chat_api.py:883-905`) formats references on read:

```python
for ref in conv.reference:
    if isinstance(ref, list): continue
    ref["chunks"] = chunks_format(ref)
```

`chunks_format` (`rag/prompts/generator.py:41-64`) normalises each chunk to
`id`, `content`, `document_id`, `document_name`, `dataset_id`, `image_id`,
`positions`, `url`, `similarity`, `vector_similarity`, `term_similarity`,
`row_id`, `doc_type`, `document_metadata`. `get_session` also injects `avatar`
from the owning chat's `icon`.

This is a richer citation record than the frontend currently renders, and it is
per assistant turn — which is the shape the source UI needs anyway.

## Decision

**1. `Project` → platform Chat (`Dialog`). Frozen.** A Studio project becomes a
platform chat assistant. `ProjectRecord.name` → `name`,
`ProjectRecord.instructions` → `prompt_config.system`, project dataset scope →
`dataset_ids`. The platform chat id **is** the project id; no local id is kept
alongside it.

**2. `Thread` → platform Session (`Conversation`). Frozen.** Per plan §3.2,
"Thread ID olarak Rag Platform backend session ID kullanılır": the session id
**is** the thread id. `ThreadRecord.title` → session `name`.

**3. Every session lives under a chat, so a thread cannot exist without a
project.** The platform addresses sessions as
`/chats/<chat_id>/sessions/<session_id>` and `session_completion` enforces it:

```python
if session_id and not chat_id:
    return get_data_error_result(message="`chat_id` is required when `session_id` is provided.")
```

A chat with no project selected therefore resolves to a **default assistant**,
created once per tenant and reused, rather than to a session with no chat. The
default assistant is a real, listable chat — not a hidden record — so the user
can see and configure what their unscoped chats are using.

**4. The platform is the source of truth for message history.** Per plan §3.2,
"Rag Platform backend session history source of truth olur". The client does not
write history: it reads sessions, and it appends only by calling
`/chat/completions`. Local storage may cache for offline reads, but a
disagreement is resolved in the server's favour, never merged.

**5. `MessageRecord.parentId` is not sent to the platform, and message trees are
not synthesised.** History is a linear paired array. The adapter maps
`messages[]` → `MessageRecord[]` with `parentId` set to the previous message in
the array, so the existing UI keeps working, and maps nothing back. A UI feature
that needs true branching (fork, edit-and-resend) is resolved in ADR 0004
(decisions D2 and C) — not by inventing a tree the backend cannot store.

**6. "Delete message" is presented as what it is: deleting a turn.** Because the
backend pops the pair and drops the turn's reference entry, the UI must not offer
a control that appears to delete one side of an exchange. The action is named and
confirmed at turn granularity.

**7. Per-message editing and history replace have no platform path and are not
faked.** `saveChatMessage` and `syncChatMessages` are dropped rather than
re-pointed. Per ADR 0001 decision 5, a control that cannot be served is removed
or stated, never left silently inert.

**8. Retrieval settings are normalised field-by-field, not passed through.** The
assistant carries `similarity_threshold`, `vector_similarity_weight`, `top_n`,
`top_k`, `rerank_id`; the retrieval call carries `dataset_ids`, `question`,
`document_ids`, `page`, `page_size`, `similarity_threshold`,
`vector_similarity_weight`, `top_k`, `rerank_id`, `keyword`, `highlight`,
`use_kg`, `toc_enhance`, `cross_languages`, `metadata_condition`. The frontend's
`RagMode` (`hybrid | lexical | dense`) maps onto `vector_similarity_weight`
rather than onto a mode field, because the platform has no mode field:
`dense` and `lexical` are the endpoints of that weight, and the mapping is
recorded once in the adapter, not guessed per call site. Note that the
frontend's `DEFAULT_RAG_TOP_K = 5` is a *result count* while the platform's
`top_k` default of 1024 is a *candidate-pool size* — the frontend value maps to
`page_size`, never to `top_k`.

**9. Citations map from `reference[turnIndex]`, addressed by turn, not by
message.** `chunks` and `doc_aggs` from `chunks_format`'s normalised shape feed
the source UI. The turn index is `(messageIndex - 1) // 2` — the backend's own
arithmetic, used verbatim rather than re-derived.

**10. Name constraints are enforced client-side before the call.** 255 UTF-8
bytes, non-empty after trim, and unique per tenant for chats. A duplicate name
is surfaced as a validation error on the field, not as an error envelope after
the fact.

**11. Assistant model configuration is part of creating a project, not a hidden
default.** Because chat creation requires a resolvable `llm_id`, the project
creation flow either uses the tenant default (when one exists) or asks. It never
creates a chat that cannot complete — the `LookupError("No default chat model for
tenant.")` in `fixtures/stream.json` is precisely the failure this avoids.

**12. Job SSE becomes document-status polling.** Per plan §3.2, "Polling state
machine ile değiştirilir". `streamJobEvents` is replaced by polling the
document's `run` / `progress` / `progress_msg`. The status normaliser is
**total over all six** `TaskStatus` values: `UNSTART`→`pending`,
`SCHEDULE`→`pending`, `RUNNING`→`running`, `DONE`→`completed`, `FAIL`→`failed`,
`CANCEL`→ a new `cancelled` member. `DocumentStatus` gains that member rather
than folding cancellation into `failed`, because a user who cancelled a parse
has not suffered an error.

**13. `archived`, `rootPath`, `sandboxPath`, `pairId`, `modelType`, the two
code-exec container ids and `forkedFrom*` have no platform column.** They are
enumerated here so the mapping is total, and each is resolved rather than silently
dropped at the adapter boundary: `archived` in ADR 0004 decision D3,
`forkedFrom*` in D2. The remaining six — `rootPath`, `sandboxPath`, `pairId`,
`modelType`, and the two code-exec container ids — belong to capabilities that
stay Studio-local per ADR 0001 decision 2 (local inference, model selection,
compare-mode pairing, code execution). They stay client-local frontend state, are
never sent to the platform, and their absence from `Conversation` is therefore
expected rather than lossy.

## Alternatives rejected

* **`Project` → Dataset, `Thread` → Chat** — a dataset is a corpus, with no
  prompt, no model and no message history; and it would leave `Conversation`
  unused while forcing one assistant per conversation. Every project would need
  a dataset even when it has no documents.
* **`Thread` → Chat, with sessions hidden** — creates one `Dialog` per
  conversation. Each carries model config, prompt config and dataset bindings,
  so a change to the project's instructions would have to fan out across every
  thread, and `GET /chats` would return the user's entire chat history as
  assistants.
* **Keep local `ProjectRecord`/`ThreadRecord` as the source of truth and sync to
  the platform** — two writers for one history, with the server mutating the
  assistant message on every streamed chunk (`structure_answer`). The
  reconciliation has no correct answer.
* **Map `MessageRecord.parentId` onto the paired array by convention** — the
  backend `assert`s the pairing when deleting. A tree flattened into pairs either
  loses branches or trips that assertion, which surfaces as a 500.
* **Emulate per-message editing by deleting the turn and re-completing** — a
  different assistant answer comes back, the reference entry is regenerated, and
  the operation is not an edit. Presenting it as one reports success it did not
  achieve.
* **Fold `CANCEL` into `failed`** — a cancelled parse would show as an error,
  and the user would be told something broke when they stopped it.
* **Keep the `/jobs` SSE shape and synthesise frames from polling** — the frame
  type set (`progress`/`complete`/`error`) cannot express `cancelled` or
  `scheduled`, so the synthetic stream would be lossier than the polled state it
  was built from.
* **Send `RagSource: {type:"thread"}` as an empty `dataset_ids`** — the route
  rejects it (`"dataset_ids is required."`), so thread-scoped retrieval would
  fail on every call. Handled as an unsupported scope with a real decision, not
  as a degenerate request.

## Consequences

* Faz 7 must create or reuse a chat before the first thread. `saveChatThread`
  cannot be a single call any more; the contract matrix already records this on
  its row.
* A flat thread list is no longer one request. `listChatThreads` becomes one
  `GET /chats/<chat_id>/sessions` per chat, and the sidebar needs a chat id in
  scope. Ordering is limited to `create_time | update_time | name`
  (`chat_api.py:868-870`), so any other sort is client-side over a fetched page.
* `getChatMessage` and `listChatMessages` fold into the session read: messages
  arrive inside the session object, so a single-message read is a client-side
  lookup.
* Two frontend functions lose their backend entirely — `saveChatMessage` and
  `syncChatMessages`. The features that call them are resolved in ADR 0004.
* `DocumentStatus` grows a `cancelled` member, which touches every exhaustive
  `switch` over it. That is a real, bounded refactor in the phase that lands
  polling, and it is preferable to a lossy normaliser.
* The default assistant is a visible chat in the user's list. Users will see a
  chat they did not create by name; the UI must label it as the default so it
  does not read as stray data.
* Project names become unique per tenant and byte-capped. Existing local
  projects with duplicate names cannot all migrate as-is — the migration phase
  must disambiguate rather than fail silently.
* All datasets bound to one assistant must share an embedding model
  (`chunk_api.py`: "Datasets use different embedding models."). A project cannot
  aggregate corpora built with different embedders, which constrains how project
  sources are offered in the UI.
