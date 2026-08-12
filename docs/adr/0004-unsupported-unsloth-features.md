# ADR 0004 — Twenty-eight frontend calls have no platform equivalent; each is composed, kept Studio-local, or removed outright, and none is left as a control that silently does nothing

* Status: Accepted
* Date: 2026-08-12
* Scope: Faz 0B decides; phases 1–8 execute. `features/chat`, `features/rag`, `features/auth`, `features/training`
* Supersedes: nothing. Superseded by: nothing.

## Context

`docs/rag-platform/contract-matrix.md` classifies all 272 method+path literals in
`studio/frontend/src`. Twenty-eight are **mapped-but-absent**: the platform is the
declared owner of the capability's family (per ADR 0001), yet no platform route
serves the specific call. These are the hard cases — a feature the platform
*nearly* supports.

The plan constrains the answer twice, and the two constraints pull in opposite
directions:

* line 193 — "Her unsupported özellik feature flag ile görünmez veya disabled
  yapılmalıdır; sahte başarı ve boş ekran üretilmemelidir."
* line 200 — "Kullanıcıya anlamlı public bir yetenek `unsupported` bırakılamaz;
  release öncesinde ekran/aksiyon karşılığı eklenmeli veya ürün sahibi tarafından
  gerekçeli ADR ile `api-only` kararı verilmelidir."

So "unsupported" is not a bucket to sweep into. It is a decision that has to be
argued per call, and the only permitted end states are: composed from platform
primitives, kept Studio-local with a reason, or removed from the product. A
control that stays visible and does nothing is excluded by line 193; a
user-meaningful capability quietly dropped is excluded by line 200.

The ADR filename is the one the plan fixes at line 418. What it actually records
is the resolution of each gap, and in eight cases the resolution is *not*
"unsupported".

### The 28, grouped by what they actually are

| Group | Calls | Phase |
|---|---|---|
| Deep research | 6 | 4 |
| Chat attachments (Studio blob store) | 3 | 7 |
| Chat UI settings | 2 | 7 |
| Import ledger | 2 | 7 |
| Message write / replace | 2 | 7 |
| Fork | 2 | 7 |
| Thread-scoped documents | 2 | 5 |
| Cross-dataset document list | 1 | 5 |
| Job SSE | 1 | 5 |
| Whole-history export | 1 | 7 |
| Batch message read | 1 | 7 |
| Auth refresh | 1 | 2 |
| Provider public key | 1 | 3 |
| Host hardware probe | 1 | 1 |
| OpenAPI route sniff | 1 | 1 |
| Server-side generation cancel | 1 | 8 |

### Deep research is not absent from the platform — it is a different shape

The contract matrix currently says "Deep-research orchestration is Studio-only
with no platform counterpart". That is **too strong**, and reading the source
corrects it.

`rag/advanced_rag/harness/config.py` defines four `THINKING_MODES` — `low`,
`medium`, `high`, `ultra` — with `ultra` carrying
`execution_strategy="deep_research"`, `requires_decomposition=True`,
`allows_dynamic_claims=True`, `allows_replan=True`,
`max_orchestrator_cycles=4`, `max_parallel_agents=3`.

One call site is dead (`api/db/services/dialog_service.py:725`):

```python
if False:  # prompt_config.get("reasoning", False) or kwargs.get("reasoning"):
    reasoner = DeepResearcher(chat_mdl, prompt_config, partial(retriever.retrieval, …), internet_enabled=use_web_search)
```

But `rag_agent` is live (`dialog_service.py:1875-1903`):

```python
async def rag_agent(dialog, messages, stream=True, **kwargs):
    prompt_config = dialog.prompt_config or {}
    if not prompt_config.get("reasoning", 0) and not kwargs.get("reasoning"):
        async for ans in async_chat(dialog, messages, stream, **kwargs):
            yield ans
        return
    …
    from rag.advanced_rag.harness.config import THINKING_MODES
    _mode_labels = list(THINKING_MODES.keys())
    _n = int(str(kwargs.get("reasoning")).strip())
    thinking_mode = _mode_labels[_n - 1] if 1 <= _n <= len(_mode_labels) else "medium"
    …
    rag_tools = RAGTools(…, thinking_mode=thinking_mode, …)
```

and it is reachable from the wire: `chat_api.py:1338`, `:1375` and `:1395` call

```python
async for ans in rag_agent(dia, msg, True, session_id=session_id, **req):
```

splatting the whole request body, so `reasoning: "1".."4"` in a
`POST /api/v1/chat/completions` body selects a thinking mode.
`prompt_config.reasoning` on the assistant does the same persistently.

What the platform therefore has: an **inline, per-completion agentic retrieval
mode**, four levels, whose output arrives in the same completion stream.

What Studio has (`features/chat/api/research-api.ts`, 359 lines): a **durable,
resumable, user-approved run object**. `ResearchRunStatus`
(`features/chat/types/research.ts`) has nine states — `planning`,
`awaiting_approval`, `queued`, `running`, `paused`, `cancelling`, `cancelled`,
`completed`, `failed`. `updateResearchPlan` takes an `expectedRevision` for
optimistic concurrency; `approveResearchRun` takes `planRevision` + `planHash`;
`streamResearchEvents(id, after)` resumes an event log from an offset;
`getResearchThreadState(threadId)` recovers the active run after a reload.

These are not two implementations of one feature. The platform has no run entity,
no plan revision, no approval gate, no resumable event log and no
`GET .../active` recovery. Six frontend calls address exactly those five things.

### Attachments, settings and the ledger are Studio's own stores

`listChatAttachments` / `fetchChatAttachmentBlob` / `deleteChatAttachment` read a
Studio blob store holding pasted images and extracted text per message. Platform
attachments are dataset documents — a different lifecycle (parsed, chunked,
embedded, retrievable) with a different scope (dataset, not message).

`getChatSettings` / `saveChatSettingsPatch` persist per-user UI state: model
pick, sampling, panel layout. The platform stores *assistant* configuration on
the `Dialog`, which is shared and semantic, not per-user and cosmetic.

`listChatImportLedger` / `recordChatImportLedger` are Studio's bookkeeping for
the one-shot Dexie import. The platform has no reason to know an import happened.

### Two calls are probes, not features

`selectTrainingMethodForHardware` (`GET /api/system/hardware`) reads host GPU/RAM
to pick a training method. Training is Studio-local per ADR 0001, and a remote
platform cannot see the user's hardware.

`backendEnforcesTitleGuard` (`GET /openapi.json`) sniffs whether a route exists,
from `repair-legacy-chat-titles.ts:49`. It is a migration-era existence check
against Studio's own schema.

### Auth refresh and provider public key are settled elsewhere

`POST /api/auth/refresh` — no platform refresh route exists; ADR 0002 decision 1.
`GET /api/providers/public-key` — the platform accepts provider credentials over
TLS with no client-side envelope; the login-path analogue is ADR 0002 decision 5.
Both are listed here so the 28 are complete, and decided there.

### Archive does not exist in the platform at all

`grep -rn "archive\|archived" api/apps/restful_apis/chat_api.py api/db/db_models.py`
returns **nothing**. `ProjectRecord.archived` and `ThreadRecord.archived`
(`features/chat/types.ts:22,34`) have no column, and `chat-history-storage.ts`
threads an `includeArchived` flag through five call sites (`:61`, `:118`, `:345`,
`:627`, `:682`) plus a dedicated `archived-chat-export.ts`.

### Server-side cancellation exists, but not for generation

`onAbortCancel` (`chat-adapter.ts:4584`) posts `/api/inference/cancel`. The
platform's only cancel route is document-parse scoped
(`api/apps/restful_apis/task_api.py:30,59`):

```python
@manager.route("/tasks/<task_id>/cancel", methods=["POST"])
…
REDIS_CONN.set(f"{task_id}-cancel", "x")
```

That is a parse task, not a completion stream. Aborting the `fetch` closes the
client's reader; the server-side generation keeps running and its assistant turn
is still persisted by `structure_answer` (ADR 0003 constraint 3). So an aborted
generation **still costs tokens and still lands in history** — the opposite of
what the Studio button implies.

## Decision

Each of the 28 gets exactly one of four dispositions. No fifth option, and in
particular no "flag it off and revisit".

### A. Composed from platform primitives — 5 calls

| Call | Composition |
|---|---|
| `listAllDocuments` | Iterate the user's datasets, fan out `GET /datasets/<id>/documents`, merge client-side, paginate over the merged set. |
| `getJob` (and its SSE sibling) | Poll the document's `run` / `progress` / `progress_msg` per ADR 0003 decision 12. |
| `batchListChatMessages` | Disappears: the session read returns the whole history, so the batch is a client-side slice. |
| `buildBackendChatExport` | Compose from session reads across the user's chats. Export stays a real, working feature. |
| `listChatAttachments`' text half | Extracted document text comes from the dataset chunk read, not from a blob store. |

Composition is permitted only where the result is **equivalent**, not merely
similar. `listAllDocuments` over N datasets returns the same document set the
Studio route returned; the difference is request count, which is a performance
note, not a behaviour change. The UI must show a determinate loading state across
the fan-out and must surface a partial failure (one dataset 403s) rather than
presenting a silently short list.

### B. Kept Studio-local with a stated reason — 8 calls

`listChatAttachments`, `fetchChatAttachmentBlob`, `deleteChatAttachment`,
`getChatSettings`, `saveChatSettingsPatch`, `listChatImportLedger`,
`recordChatImportLedger`, `selectTrainingMethodForHardware`.

Per ADR 0001 decision 2 the test is "no platform route can serve it", and each of
these fails that test for a reason of *kind*, not of coverage: a message-scoped
blob store, per-user cosmetic state, local migration bookkeeping, and host
hardware. They stay on the Studio backend and are recorded `api-only` in the
coverage matrix with these reasons.

This is not a deferral. Nothing about them changes in any later phase.

### C. Removed outright — 4 dispositions over 5 functions

| Call | Why removal is correct |
|---|---|
| `backendEnforcesTitleGuard` | A route-existence probe for a Studio-era migration. Delete the probe with the migration. |
| `POST /api/auth/refresh` (`promise`) | ADR 0002: a 401 means re-authenticate. The call, the `logoutGeneration` counter and the in-flight dedup go with it. |
| `importProviderPublicKey` | No platform route; keeping a key fetch would imply an envelope the platform does not apply. |
| `saveChatMessage` + `syncChatMessages` | ADR 0003 decisions 5 and 7: history is append-only through completions. `update_session` rejects `messages` and `reference` outright, so there is nothing to re-point. |

Removal means the **call and its UI affordance** go together. Per plan line 193 a
control may not remain visible and inert: concretely, the message-edit affordance
is removed from the message actions menu, not disabled with a tooltip.

### D. Feature-level decisions — the remaining calls

**D1. Deep research is re-scoped to the platform's `reasoning` mode, and the run
object is dropped.** The six research calls are removed. In their place the
completion request carries `reasoning` (1–4), and the assistant's
`prompt_config.reasoning` sets a default. What the user gains: four thinking
levels with real agentic retrieval, in the normal chat stream. What the user
loses, stated plainly in the UI and here: plan review and approval before the run
starts, pause/resume, retry, cancellation of an in-flight run, and recovery of a
run in progress after a reload.

This is a **product reduction**, not a deferral, and it is the item in this ADR
most likely to be revisited — a run object could be built on the Studio backend
later. That would be a new decision with its own record, not an implicit
continuation of this one.

**D2. Fork is composed, and its lossiness is named.** `forkChatThread` becomes:
create a new session under the same chat, then replay the messages up to the fork
point through completions. `getForkCount` is removed — with no fork primitive
there is no sibling index to count, and a client-maintained count would be wrong
the moment another client branches.

The replay is not a copy: re-completing produces *new* assistant answers with new
references. The composed operation is therefore presented as **"Branch from
here"** — a new conversation seeded with this history — and never as "fork", which
implies identity with the original. `forkedFromThreadId` / `forkedFromMessageId`
stay client-side as provenance for the UI and are never sent to the platform.

**D3. Archive becomes a client-side view state, explicitly not a server state.**
The platform has no archive column, so `archived` moves to local persisted UI
state keyed by chat/session id: an archived conversation is hidden from the
default list and reachable from an "Archived" view. Because it is local,
archiving does not follow the user to another device, and the UI says so at the
point of archiving rather than implying a server-side property.

Deleting on archive is rejected: it would destroy data on an action users
reasonably expect to be reversible.

**D4. Thread-scoped documents become an explicit per-session dataset.**
`listThreadDocuments` and `uploadThreadDocument` are re-expressed as: on the first
upload in a conversation, create a dataset named after the conversation, bind it
to the assistant's `dataset_ids`, and upload there. Listing reads that dataset.

Two consequences are accepted and must be surfaced: the conversation's documents
are visible in the datasets UI (they *are* a real dataset), and because all
datasets bound to one assistant must share an embedding model
(`chunk_api.py`: "Datasets use different embedding models."), the per-session
dataset inherits the assistant's embedder rather than choosing its own.

The frontend's `DEFAULT_RAG_SOURCE = { type: "thread" }`
(`chat-runtime-store.ts:104`) therefore keeps working, because the thread now has
a dataset. Sending an empty `dataset_ids` is rejected by the route
(`"dataset_ids is required."`) and is not attempted.

**D5. Generation cancellation is client-side abort only, and the UI stops
claiming otherwise.** `onAbortCancel`'s POST is removed. The stop button aborts
the fetch and stops rendering. Because the server keeps generating and
`structure_answer` still persists the assistant turn, the UI must not report
"cancelled" as though the work stopped: the stopped message reads **"stopped — the
response may still be completing on the server"**, and once the session is
re-read the full assistant turn appears.

This is the least satisfying item here. It is recorded as a **known limitation
with a named backend fix** — a completion-scoped cancel route analogous to
`task_api.py`'s Redis flag — which is a backend change and therefore out of Faz 0
scope per the standing instruction not to modify backend source.

### E. Nothing is deferred behind a flag

Per ADR 0001 decision 6 and the standing instruction, no new feature flag is
introduced by any disposition above. Where a capability is reduced (D1, D3, D5)
the reduced version ships; where it is removed (C) the affordance is removed with
it. The pre-existing kill switches in `config/disabled-features.ts` are untouched.

### F. Every disposition is recorded per endpoint, not only here

The coverage matrix carries, for each of the 28: class, reason, disposition letter
and target phase. A reader who lands on one row does not have to find this file to
learn why the row says what it says. In particular the six deep-research rows'
reason strings — currently "no platform counterpart"
(`scripts/rag-platform/contract-matrix.mjs:752,757,762,767,772,777`) — are
corrected to name the real distinction: the platform has an inline `reasoning`
mode and no durable run object.

## Alternatives rejected

* **Mark all 28 `unsupported` and revisit per phase** — plan line 200 forbids
  leaving a user-meaningful capability unsupported, and 15 of the 28 are
  user-meaningful. It also converts a decision into a backlog.
* **Map Studio's research run onto the platform's `reasoning` mode 1:1** — the
  platform has no run id, no plan revision, no approval gate and no resumable
  event log. The nine-state `ResearchRunStatus` would have to be synthesised
  client-side over a single streaming call, and `awaiting_approval` in particular
  cannot exist: the platform starts retrieving immediately.
* **Re-implement the research run object on the Studio backend against platform
  retrieval** — a real option, possibly the right one later, but it means
  building an orchestrator that calls the platform per step. Out of Faz 0 scope
  and against the migration's stated shape (ADR 0001: neither backend proxies the
  other).
* **Keep the disabled research UI visible with a "coming soon" state** — plan
  line 193 forbids empty screens and fake success.
* **Implement archive by deleting the session** — irreversible, and users expect
  archive to be reversible. Data loss on a non-destructive-sounding action.
* **Implement archive as a name prefix or a tag on the session** — the platform
  has no tag field on `Conversation`, and encoding state in `name` corrupts the
  user's own title and breaks `orderby=name`.
* **Keep the fork API and replay silently, presenting the result as identical** —
  re-completion yields different answers. Calling that a fork reports an outcome
  it did not achieve.
* **Thread documents as a `document_ids` filter on retrieval instead of a
  dataset** — `document_ids` narrows *within* datasets the assistant already has;
  it cannot introduce documents that were never uploaded to a dataset. Upload
  still needs a dataset, so this only moves the problem.
* **Keep `POST /api/inference/cancel` pointed at Studio while chat runs on the
  platform** — Studio cannot cancel a generation it is not running. The call
  would succeed and do nothing, which is the exact failure mode plan line 193
  names.
* **Fabricate SSE frames for document parse from polled state** — the frame type
  set (`progress`/`complete`/`error`) cannot express `cancelled` or `scheduled`
  (ADR 0003 decision 12), so the synthetic stream is lossier than the state it
  came from.
* **Migrate Studio's attachment blob store into datasets** — a pasted screenshot
  would become a parsed, chunked, embedded, retrievable document in the user's
  corpus. That changes retrieval results as a side effect of pasting an image.

## Consequences

* Five capabilities become multi-request compositions. `listAllDocuments` is
  O(datasets) requests, so the all-documents view needs pagination over merged
  results and a partial-failure state.
* Deep research loses plan approval, pause, resume, retry, in-flight cancel and
  reload recovery. Six functions plus `features/chat/types/research.ts`,
  `stores/research-run-store.ts`, `components/research-message.tsx` and
  `components/research-activity-panel.tsx` are removed or reduced in Faz 4, and
  the UI must explain the new model rather than presenting fewer buttons with no
  explanation.
* Fork becomes "branch from here" and re-runs the model, which costs tokens the
  old fork did not. The confirmation dialog must say so.
* Archive becomes device-local. A user with two devices sees different archive
  state on each. Stated as a limitation at the point of use.
* Every conversation with an upload creates a dataset. Dataset counts grow with
  conversation count, and the datasets UI shows conversation-named datasets. That
  must be visible in the UI's own vocabulary, not a surprise.
* Stopping a generation no longer stops the backend. Tokens are consumed and the
  answer is persisted. This is the one accepted regression whose fix is a backend
  route we have chosen not to write in Faz 0.
* Eight capabilities stay Studio-local permanently, so a browser-only deployment
  without the Studio backend loses message attachments, chat UI settings
  persistence, the import ledger and hardware-aware training defaults.
* Five functions are deleted along with their UI affordances. The branding and
  coverage gates do not catch a dead affordance, so removal is verified by the
  implementing phase's own tests, not by a scan.
* The contract-matrix generator's deep-research reason strings must be corrected
  in the same phase, or the generated document will contradict decision D1.
