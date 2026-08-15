# Unforgettable

Progressive memory for a single AI face: remember, correct, rehearse after failure, and (optionally) internalize stable skill into an adapter — without treating every lesson as a full retrain or a disposable chat turn.

Unforgettable is an **Apache 2.0** Python package. It lives in this directory and does not import Studio or Unsloth at import time. Studio is an optional face: pick the virtual model `unforgettable` in chat and the episode loop runs behind a normal OpenAI completions request.

Architecture notes that this package implements: [`plans/MemoryWheels.md`](plans/MemoryWheels.md) and the phase plans in [`plans/`](plans/). Developer-depth documentation: [`TECHNICAL.md`](TECHNICAL.md).

## What it does

A live agent accumulates working context that vanishes when the session ends. Unforgettable keeps three things distinct:

| Layer | Holds | How you change it |
|-------|--------|-------------------|
| **A — working** | This episode’s mode, traces, and repaired-plan notes | Dropped when the request ends |
| **B — notebook** | Claims, procedures, error→fix lessons, twin-drift notes | Tools, extract + admission, compact |
| **C — sidecar** | Optional LoRA adapter trained from admitted B | Operator CLI: pack → train → eval → promote / rollback |

Default motion (the “act path”):

1. Act in the **world** (the project sandbox).
2. On a recognized failure, clone the tree into a **sim** and rehearse (run tests when a command is known).
3. Retry the world with a repaired-plan note. Confirm-before-retry is available; it is off unless stakes are high or permission mode is `ask`.
4. Write lessons through admission. Extract stays **proposed** until you `admit` it. Explicit `memory_write` of a trusted world/mixed fact can stick immediately; tools cannot claim `human`, sim contact cannot claim `world`, and directives stay proposed until you admit them.

World remains the judge of record. Sim-only dynamics are not auto-promoted to world truth.

## Requirements

- Python 3.9–3.14
- stdlib + SQLite with FTS5 (present on normal CPython builds)
- Optional: Unsloth Studio, to use the virtual model in chat
- Optional: Unsloth + GPU, only if you train a real adapter (`python -m unforgettable train --backend unsloth --base …`)

The inspect CLI and the Apache test suite do not need a GPU.

## Use it from Studio

1. Load a real inner model as usual (GGUF, safetensors, or a catalog LoRA).
2. Send chat completions with `model` set to `unforgettable` or `unforgettable/<inner-id>`.
3. The route strips the alias, runs `unforgettable.loop.episode.run`, and streams tokens plus Studio tool frames (`tool_start` / `tool_end`, …).
4. **Settings → Unforgettable** holds episode defaults and the approver. **Unforgettable** in the sidebar (under More) opens the dashboard at `/unforgettable` for the proposed queue, notebook, standing, hygiene, and adapter registry.

```http
POST /v1/chat/completions
{
  "model": "unforgettable",
  "stream": true,
  "messages": [{"role": "user", "content": "run the tests"}]
}
```

Optional extras on the same request (also first-class fields on Studio’s chat payload):

| Field | Effect |
|-------|--------|
| `stakes` | `"high"` drops sim/infer from world retrieve and can require confirm on retry |
| `test_command` | Sim harness command (else a stored procedure titled `test command`, else tree detect) |
| `confirm_retry` | `true` / `false` to force or skip the Allow/Deny card before world retry |
| `permission_mode` | `"ask"` also requires confirm on retry |
| `max_clones` / `max_sim_turns` | Budgets (defaults 1 clone / 8 sim turns) |
| `adapter_id` | Attach a trained sidecar adapter and shrink standing playbooks it already learned |
| `skip_standing` | Skip compiled standing inject |
| `planner` | `"on"` asks a larger supervisor model for a temporary plan (this episode only) |
| `planner_model` | Model id for that planner complete (else `UNFORGETTABLE_PLANNER_MODEL`, else the inner model) |

The inner model can call memory tools. Durable facts must go through those tools — chat history is not B, and a successful generate is stored as a short grade, not the full completion.

| Tool | Role |
|------|------|
| `memory_write` | Remember (kind, title, body, provenance) |
| `memory_search` / `memory_get` | Look up |
| `memory_supersede` | Correct; old row kept as history |
| `memory_deprecate` | Archive (excluded from default retrieve) |
| `memory_compact` | Hygiene preview (`dry_run` defaults true) |
| `memory_compile` | Pin a procedure into the standing prompt (`dry_run` defaults true) |
| `rims_enter_sim` | Explicit “enter the twin” |

## Inspect and operate from the CLI

Point the CLI at Studio’s store when you used the virtual model:

```bash
export UNFORGETTABLE_DB="$STUDIO_HOME/memory/memory.db"
# or: export STUDIO_HOME=…   (CLI picks $STUDIO_HOME/memory/memory.db when that tree exists)
# else: $UNFORGETTABLE_HOME/memory.db or ~/.unforgettable/memory.db

python -m unforgettable path
python -m unforgettable list --kind procedure --status active
python -m unforgettable search "pytest"
python -m unforgettable get <id>
python -m unforgettable admissions --limit 50
python -m unforgettable admit <id>          # promote a proposed extract
python -m unforgettable reject <id>
python -m unforgettable review              # approval voter over proposed rows
python -m unforgettable mine                # batch voter + optional new proposed drafts
python -m unforgettable compact             # preview (default)
python -m unforgettable compact --apply     # mutate
python -m unforgettable compiled
python -m unforgettable load                # standing / retrieve / traj chars
python -m unforgettable rollouts --contact sim
python -m unforgettable probes --run --world /path/to/project
```

Every subcommand accepts `--db PATH`.

### Optional sidecar (C)

Packs are built from **admitted** `procedure` / `error_fix` bodies. Episode transcripts are never training gold. Sim-only glory cannot vote unless you pass `--include-sim` and the same episode also has a world pass and no twin-note.

```bash
python -m unforgettable pack --dry-run
python -m unforgettable pack --apply
python -m unforgettable train --backend fake          # CI / wiring
python -m unforgettable train --backend unsloth --base <model>
python -m unforgettable eval <adapter-id>
python -m unforgettable promote <adapter-id>          # refuses if eval did not pass
python -m unforgettable rollback
```

Promote never merges into base weights. Rollback discards the promoted row; files stay on disk.

### Optional supervisor (approver + planner)

A larger model can judge promotions and (separately) sketch a plan for one episode. It is not trained. It does not replace `admit()` or the act/sim policy.

```bash
export UNFORGETTABLE_VOTER=advisory   # or binding (deny blocks admit/promote)
export UNFORGETTABLE_SUPERVISOR_URL=http://127.0.0.1:8080/supervise
python -m unforgettable review --apply
python -m unforgettable mine --apply
```

Planner is per request: set `planner: "on"` on the chat payload, or `UNFORGETTABLE_PLANNER=on` for Studio. The plan is working memory only.

## Brief architecture

```
User / goals
    │
    ▼
Throne  —  act/sim policy, admission, confirm, budgets
    │
    ▼
Middle  —  episode.run: retrieve + standing + trajectories → generate
    │         on fail: clone → sim tests / generate → retry world → extract
    │
    ├──────── world rim (project sandbox)     sim rim (cloned session dir)
    │
    ▼
B store (SQLite + FTS5)     optional C adapter (LoRA dir next to the db)
```

- **World** is the project sandbox (`session_id`, often `project-<id>`).
- **Sim** is a new `sim-<episode>-<n>` directory, never the same path as world.
- **B** is `$STUDIO_HOME/memory/memory.db` (sibling of RAG, not `rag.db`).
- **C** is an operator job. Live Studio does not attach an adapter unless you pass `adapter_id`.

## Tests

From the Unsloth repo root (do **not** rely on bare `pytest` — root `testpaths` is `tests/security`):

```bash
python -m pytest unforgettable/tests
```

No GPU. See [`TECHNICAL.md`](TECHNICAL.md) for the full build and test matrix, including Studio-face tests.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Studio files that import this package remain AGPL. The dependency arrow is **Studio → unforgettable**, never the reverse.
