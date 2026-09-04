# Phase 1 — Memory Wheels bones

First implementation slice after the substrate evaluation. Incorporates: internally managed agents behind the existing OpenAI API; world = project sandbox, sim = cloned sandbox; SQLite store; Grok Build as reference only; Apache 2 for the new package; AGPL only for thin Studio touchpoints.

This phase puts **every architectural piece in place**, maps it to what already exists, implements the basics that belong in the skeleton, and leaves later brains as real interfaces with stub or trivial bodies.

---

## Decisions locked by feedback

| Topic | Decision |
|-------|----------|
| Unsloth | Engine room: local serve + later C. Not the memory product. |
| Grok Build | Reference only. Lift lifecycle ideas (remember / search / first-turn inject / end-of-episode flush). Do not fork, vendor, or run as a client. |
| Face | Existing Studio OpenAI API. A virtual model id looks like a normal LLM. |
| World | One main codebase: the chat **project** sandbox (`session_id = project-<id>`). |
| Sim | Auto-cloned copy of that tree in a **new** session sandbox. Not a second project row. |
| Store | New SQLite file. Reuse Studio’s WAL / per-call `get_connection` / lazy schema **pattern**. Do not use `rag.db` `documents`/`chunks`. |
| License | `unforgettable/` = Apache 2.0. Studio edits = AGPL, as small as possible, **Studio imports the Apache package** (never the reverse). |
| Agents | Internal modules (retriever, extractor, admissions, act/sim controller) that use tools locally to maintain B. Not user-visible extra APIs. |
| Phase 1 scope | Structure + touchpoints + basic working features. Not full extract, dream, C, or fancy eyes. |

---

## Target shape (what “done” means for Phase 1)

A client can:

```http
POST /v1/chat/completions
{ "model": "unforgettable", "messages": [...], "stream": true }
```

and get a normal OpenAI chat completion, while behind that request:

1. **Retriever** pulls relevant B records into working context (A).
2. **Inner wheel** is Studio’s existing tool loop (python / terminal / web_search / render_html) plus **memory tools**.
3. Tools that touch the codebase run in the **world** sandbox by default.
4. On **recognized failure** (Phase 1: non-zero terminal exit, python exception, or explicit `rims.enter_sim`), the controller clones world → sim, flips contact mode, and retries tools against the clone.
5. **Extractor** at episode end proposes records; **admissions** auto-allows low-risk explicit writes (and logs everything).
6. The client never has to know about B, rims, or agents.

Desktop chat works if the model picker lists `unforgettable` (catalog hook). No new UI in this phase.

---

## Package layout (Apache 2.0)

New top-level package. Does **not** import `studio`, `studio.backend`, or `unsloth_cli`.

```
unforgettable/                     # Apache-2.0
  LICENSE
  __init__.py                      # version, public façade
  host.py                          # Host protocol (the only Studio contract)
  store/
    db.py                          # memory.db connection (Studio-pattern, own file)
    schema.py                      # CREATE TABLE + migrations
    records.py                     # CRUD, supersede, deprecate
    search.py                      # FTS5 (vec reserved, not required)
  tools/
    specs.py                       # OpenAI function schemas
    handlers.py                    # dispatch used by Host and tests
  agents/
    retriever.py                   # first-turn inject
    extractor.py                   # explicit + naive end-of-episode
    admissions.py                  # gate
  rims/
    types.py                       # ContactMode = world | sim
    clone.py                       # copytree world → sim (pure fs)
  throne/
    policy.py                      # §6 defaults as data + decide()
  eyes/
    protocols.py                   # WorldEyes / SimEyes / GateEyes
    basic.py                       # tool-exit / exception → failure
  loop/
    episode.py                     # middle wheel
    context.py                     # A: mode, traces, active rim ids
  tests/                           # no GPU; tmp SQLite + tmp dirs
```

Root `pyproject.toml`: add `"unforgettable*"` to `setuptools.packages.find.include`. License metadata stays Apache-2.0 for the project; Studio files keep their AGPL SPDX headers. Add `unforgettable/LICENSE` (Apache 2.0 text) so the package is relocatable.

Path for the db, when a Host is present: `Host.memory_db_path()`. Fallback for tests / standalone: `$UNFORGETTABLE_HOME/memory.db` or `~/.unforgettable/memory.db`. Studio Host implementation will put it at `$STUDIO_HOME/memory/memory.db` (sibling of `rag/`, not inside `rag.db`).

---

## Architecture map — existing vs gap vs Phase 1 work

### Timescale stack

| Piece | Role | Existing | Phase 1 |
|-------|------|----------|---------|
| **Throne** | act/sim policy, admission, what may stick | Fragments: tool approval, `permission_mode` (action risk, not memory) | New `throne.policy` with MemoryWheels §6 defaults. `decide(event) → WORLD_ACT \| ENTER_SIM \| RETRY_WORLD \| ESCALATE`. Admission policy: auto-allow `human` + explicit tool writes in default namespace; everything else logged as `proposed`. |
| **Outer B** | durable structured store | Studio RAG = documents/chunks. Chat history = A on disk. | New schema + CRUD + FTS search + memory tools. Compaction/`dream` = interface stub. |
| **Outer C** | PEFT | Unsloth training/export | Interface stub only (`sidecar.py` placeholder). Do not touch training. |
| **Middle** | episode, extract, act↔sim | One HTTP request = one Studio tool-loop episode. No extract, no rim switch. | New `loop.episode.run()`. Wraps Host.generate. Owns mode + extract. |
| **Inner** | model + tools + A | Studio tool loop (`studio_tool_loop` / GGUF / safetensors) | Reuse as-is via Host. Add memory tools to the catalog. |
| **A (working)** | episode state | Request `messages` + sandbox workdir | `loop.context.EpisodeState`: contact mode, world/sim session ids, trace buffer (in memory, tagged `world\|sim`). |

### Contact axis

| Piece | Existing | Phase 1 |
|-------|----------|---------|
| **World rim** | Project sandbox: `ensure_chat_project_workspace` + `session_id=project-<id>`. Isolation = CWD/HOME remap + rlimits + blocklist, **not** a container. | Treat that workdir as world. Host reports `world_session_id` and runs python/terminal there when mode=world. |
| **Sim rim** | Per-thread sandbox exists. **No clone.** `fork_chat_thread` copies messages only (“Sandbox starts fresh”). | New `rims.clone.clone_tree(src, dst)` (pure, Apache). Host creates `sim-<episode>-<n>` session dir via `get_sandbox_workdir`, copies contents, skips `.unsloth_sandbox` and `.unsloth_sandbox_remap.json`. Teardown via existing `remove_session_sandbox`. |
| **World / sim eyes** | None | `eyes.basic`: non-zero terminal exit, python traceback, or tool `Error:` prefix → `RecognizedFailure`. Sim eyes: same grades inside the clone. |
| **Gate eyes** | Training eval only | Protocol + log-only implementation. |

### Grok Build ideas we actually take (reference, not code)

| Idea | Phase 1 form |
|------|----------------|
| Explicit remember | `memory.write` tool + required provenance |
| Search + get | `memory.search` (FTS) + `memory.get` |
| Forget = not delete | `memory.deprecate` (status=deprecated, excluded from default retrieve) |
| First-turn inject | Retriever before first Host.generate |
| End-of-episode flush | Extractor after last turn (naive + explicit) |
| Source weights | `provenance` column + retrieve prefers `world`/`mixed`/`human` over `sim`/`infer` |
| Staleness | `updated_at` on records; retriever may append a one-line age note (simple) |

Not in Phase 1: `/dream` topic rewrite, vector index, temporal decay curves, MMR, confirm-panel UX (Studio already has tool approval if we mark writes high-risk later).

---

## Store schema (implement now)

New file `$STUDIO_HOME/memory/memory.db` (or test path). WAL, `foreign_keys=ON`, per-call connections, `CREATE TABLE IF NOT EXISTS` + `PRAGMA table_info` migrations — copy the **pattern** from `studio/backend/storage/studio_db.py` / `rag_db.py`, not the tables.

```
namespaces
  id TEXT PK
  name TEXT
  admission TEXT          -- 'auto' | 'propose' | 'deny'
  created_at TEXT

records
  id TEXT PK
  namespace_id TEXT
  kind TEXT               -- claim | procedure | error_fix | entity | episode | directive | twin_note
  status TEXT             -- active | superseded | deprecated | proposed | rejected
  title TEXT
  body TEXT
  provenance TEXT         -- world | sim | mixed | human | infer
  confidence REAL
  supersedes_id TEXT      -- previous record, history kept
  source_episode_id TEXT
  contact_tag TEXT        -- denormalized world|sim|mixed for the write path
  created_at TEXT
  updated_at TEXT

record_fts                       -- FTS5 on title+body, content-sync with records
```

Phase 1 implements **all kinds and statuses** so later work does not migrate the soul of the schema. Vec0 is a reserved comment / future table, not a dependency (keeps tests and machines without sqlite-vec working).

Do **not** write into: `rag.db` documents/chunks, `chat_messages`, `app_settings`, `chat_settings`.

---

## Internal agents (Phase 1 bodies)

These are Python objects with a stable `run(...)` contract. Later they can call the model; Phase 1 bodies are deterministic so the skeleton is testable without a GPU.

### Retriever

- Input: last user text + `EpisodeState` + throne retrieve policy.
- Action: `store.search` (FTS), filter `status=active`, bias provenance (`world/mixed/human` first).
- Output: short markdown block of titled records with ids (not a blob dump). Cap (e.g. 6).
- Injected as a **system** prefix by the episode runner, not as fake tool results (keeps OpenAI transcripts cleaner). Optional: also expose `memory.search` so the inner model can pull more.

### Extractor

- **Explicit path (implement):** honor `memory.write` / `memory.supersede` / `memory.deprecate` during the episode (those already hit the store through admissions).
- **End-of-episode path (basic):** if the episode had a recognized failure **and** a later success, write one `error_fix` proposal: “tried X in world → failed; Y in sim/world succeeded.” Use traces in `EpisodeState`, no extra LLM call.
- **LLM extract:** function exists, returns `[]`. Flagged gap.

### Admissions

- `human` + `directive` + explicit tool write in namespace `default` → **admit** (status=active).
- `infer` / auto-extract → **proposed** (visible to `memory.get`, not default retrieve).
- `sim`-only dynamics claims stay `proposed` unless throne says otherwise (default: do not auto-promote).
- Every decision appended to an `admissions_log` table (or a JSONL next to the db) so gate eyes have an audit trail.

### Act/sim controller (throne + middle)

Default policy data (from MemoryWheels §6):

```
WORLD_ACT → on RecognizedFailure → ENTER_SIM (clone if needed)
         → SIM_UNTIL_SUCCESS or budget
         → RETRY_WORLD
         → still fail → ESCALATE (stop, write error_fix, do not loop forever)
```

Phase 1 budget: max 1 clone, max N sim tool-turns (constant, e.g. 8), then escalate. No human-in-the-loop beyond Studio’s existing tool approval.

---

## Host protocol (the Studio contract)

`unforgettable.host.Host` is a `typing.Protocol`. Studio implements it. Tests implement a fake.

```python
class Host(Protocol):
    def memory_db_path(self) -> Path: ...
    def world_session_id(self, request) -> str: ...
    def create_sim_session(self, episode_id: str) -> str: ...
    def sandbox_path(self, session_id: str) -> Path: ...
    def remove_sim_session(self, session_id: str) -> None: ...
    async def generate(self, req: GenerateRequest) -> GenerateStream:
        """Run ONE inner-wheel pass (existing Studio tool loop).
        req.session_id selects the active rim’s sandbox.
        req.extra_tools are memory tool specs.
        """
    def execute_memory_tool(self, name: str, args: dict) -> str:
        """Optional; default dispatch is unforgettable.tools.handlers."""
```

`GenerateRequest` carries messages, stream flag, enabled built-in tools, `session_id`, `thread_id`, permission fields — a **narrow** DTO, not `ChatCompletionRequest`, so the Apache package does not depend on Studio models.

The episode runner:

```
state = EpisodeState(world_session=host.world_session_id(req))
inject retriever(state, last_user)
loop:
    stream = host.generate(..., session_id=state.active_session)
    observe tools + eyes
    if throne says ENTER_SIM:
        sim = host.create_sim_session(...)
        rims.clone.clone_tree(host.sandbox_path(world), host.sandbox_path(sim))
        state.enter_sim(sim)
        continue
    if throne says RETRY_WORLD: state.enter_world(); continue
    if final text: break
extractor + admissions
```

Memory tool calls that occur **inside** `host.generate` must still hit the Apache handlers. That is a Studio touchpoint: `execute_tool` dispatches `memory.*` to `unforgettable.tools.handlers` (passing namespace / episode id).

---

## Studio touchpoints (AGPL, keep tiny)

Dependency arrow: **Studio → unforgettable**. No new Studio plugin framework beyond these call sites. Data-designer entry points are unrelated; do not reuse them.

| File | Change | Why |
|------|--------|-----|
| `pyproject.toml` | include `unforgettable*` | Ship the Apache package |
| `studio/backend/core/unforgettable_host.py` **(new)** | `StudioHost(Host)` using `get_sandbox_workdir`, `remove_session_sandbox`, and a callback into the existing generate path | All AGPL/Studio knowledge lives here |
| `studio/backend/routes/inference.py` | Early in `openai_chat_completions`: if `payload.model` is `unforgettable` (or `unforgettable/<base>`), run `episode.run(StudioHost(), ...)` instead of the raw inner path. Strip the alias so the inner generate uses the loaded/default model. | Virtual model as the public face |
| `studio/backend/routes/inference.py` | `_openai_catalog_objects` / `_openai_model_objects`: append `{id: "unforgettable", owned_by: "unforgettable"}` | `GET /v1/models` lists it |
| `studio/backend/core/inference/tools.py` | Add five specs to `ALL_TOOLS`; `execute_tool` branch for `memory.write\|search\|get\|supersede\|deprecate` → Apache handlers | Inner model can remember/correct |
| `studio/backend/models/inference.py` | Document `unforgettable` as a virtual model id in the `model` field description | API docs only |

**Do not** edit the three tool-loop implementations for extract if the episode runner wraps `generate` at the route. Extract runs in Apache `episode.run` after the inner stream finishes.

**Do not** put schema, clone logic, or throne policy in Studio files.

**Frontend:** none required. If the chat model picker is catalog-driven, `unforgettable` appears automatically. If it filters to “real” GGUF/safetensors only, add a one-line allow for this id — only if a quick check shows it would be hidden. Treat that as a follow-up if the picker ignores synthetic ids.

`/v1/messages` and `/v1/responses` stay untouched (Responses stream does not run the Studio tool loop). Public face is chat completions only.

---

## Memory tools (implement now)

OpenAI function names (stable):

| Name | Args (core) | Effect |
|------|-------------|--------|
| `memory_write` | kind, title, body, provenance, namespace? | Admissions → active or proposed |
| `memory_search` | query, top_k?, kinds?, provenance? | Active records, FTS |
| `memory_get` | id | One record including superseded history pointer |
| `memory_supersede` | id, body, title?, provenance? | New record, old status=superseded |
| `memory_deprecate` | id, reason? | status=deprecated; excluded from default search |

Wire names use underscores: OpenAI `function.name` (and Studio's validator) reject dots.

These are the “internally managed” write path. The inner model is instructed (short system addendum from the episode runner) that durable facts go through these tools, not through hoping the chat is saved.

---

## World / sim clone details (implement now)

This is foundational, so Phase 1 actually copies trees.

1. World path = `Host.sandbox_path(world_session)`. For Studio, `world_session` is `project-<projectId>` when the chat is in a project, else the thread sandbox (still valid: “this chat’s tree is the world”).
2. `create_sim_session` → new id `sim-<episode>-<n>` that does **not** start with `project-`, so it gets a private dir under `sandbox_root()`.
3. `clone_tree`:
   - `shutil.copytree(..., dirs_exist_ok=True)`
   - ignore `.unsloth_sandbox`, `.unsloth_sandbox_remap.json`, `*.deleting-*`
   - do not copy the dest marker; Host already claimed the dir
4. Subsequent python/terminal in that episode use `session_id=sim-...` until RETRY_WORLD.
5. On episode end: keep the sim dir if an `error_fix` was admitted (path stored on the record); otherwise `remove_sim_session`. Phase 1 can always keep until a size cap if simpler — prefer delete-on-success, keep-on-failure for debugging.

`rims.clone` is pure filesystem (Apache). Claiming/markers stay in StudioHost.

---

## Tests (Phase 1, no GPU)

Under `unforgettable/tests/` (or `tests/unforgettable/` if the repo’s test runner prefers that — prefer **inside the package** so it relocates with Apache code).

- Schema migrate + CRUD + supersede keeps history + deprecate hidden from search
- FTS search ranking / provenance filter
- Admissions: human admitted, infer proposed, sim-only not auto-promoted
- `clone_tree` copies files, skips markers, dest independent of src edits
- Episode runner with `FakeHost`: injects retrieve, ENTER_SIM on fake failure, RETRY_WORLD, writes error_fix
- Tool handlers round-trip through the store
- Host protocol: FakeHost satisfies the surface the runner calls

StudioHost gets a **narrow** AGPL test only if we can call `get_sandbox_workdir` without booting FastAPI; otherwise skip and trust the clone unit test + a later integration test.

---

## Explicitly out of Phase 1

- LLM-based extract / dream / contradiction detection
- sqlite-vec for memory
- C sidecar / any training job
- Frontend memory browser (use `sqlite3` / tests)
- `/v1/messages` and `/v1/responses` wrappers
- High-fidelity twin, physics, or extra sandbox isolation
- Changing Studio RAG or chat history to be B
- Grok Build code in the tree

Those stay as named stubs or comments at the module that will own them.

---

## Implementation order

1. **Package + store** — `unforgettable/` skeleton, `pyproject` include, `memory.db` schema, CRUD, FTS, tests.
2. **Tools + admissions + retriever** — handlers, default namespace, inject formatter, tests.
3. **Rims + throne + eyes.basic** — `clone_tree`, policy decide(), failure detection from tool result strings, tests.
4. **Episode runner + FakeHost** — full middle-wheel path in unit tests, including error_fix extract.
5. **StudioHost + three AGPL hooks** — virtual model, catalog entry, `execute_tool` dispatch.
6. **Smoke** — if a Studio backend can be imported in this environment, one request with `model=unforgettable` against a fake/local generate; otherwise document the manual smoke (`curl /v1/chat/completions`).

Each step is independently reviewable. 1–4 are Apache-only.

---

## License / relocatable future

- New code under `unforgettable/` is written as if it will be copied next to a different UI. No Studio types, no FastAPI, no Tauri.
- The only coupling is `Host`. A future TUI or headless server implements Host and is done.
- Prefer CC0-quality simplicity in this package (small modules, no framework), but the **license** is Apache 2.0 to match Unsloth core.
- AGPL files contain wiring only. If a function grows policy or schema, move it back to Apache.

---

## Success criteria

- `unforgettable` is importable without Studio on `sys.path`.
- Store tests pass on an empty machine (stdlib + sqlite3 FTS5).
- FakeHost episode test: fail in world → clone → succeed in sim → retry world → `error_fix` row exists with provenance `mixed` or `sim` as tagged.
- Studio: `GET /v1/models` includes `unforgettable`; `POST /v1/chat/completions` with that model reaches `episode.run` (assert via a unit test on the alias branch if we can patch Host).
- `git grep` of `unforgettable/` shows no `from studio` / `import studio`.
