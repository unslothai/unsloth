# Unforgettable — technical reference

This note is for a human or AI developer extending the package. Companion user doc: [`README.md`](README.md). Design history: [`plans/MemoryWheels.md`](plans/MemoryWheels.md) and [`plans/MemoryPhases.md`](plans/MemoryPhases.md). Phase locks that the tree implements: [`plans/MemPhase1.md`](plans/MemPhase1.md) through [`plans/MemPhase5.md`](plans/MemPhase5.md). What's next: [`Roadmap.md`](Roadmap.md).

The tree is the source of truth. Plans stay as the charter; this file describes what is actually shipped. The roadmap is current status plus deferred work — `plans/MemoryPhases.md` still reads as if Phase 2 is next.

---

## 1. Architecture

### 1.1 Purpose

Unforgettable is the **main car** of the Memory Wheels stack: progressive memory under standing watch, so one user-facing AI can remember, correct, rehearse after failure, and (rarely) internalize stable skill.

It is not a second RAG. Studio `rag.db` and chat history are **not** B. It is not a continuous fine-tune. Base weights stay frozen. It is not a physics twin unless the host provides one. The coding-domain sim is a **twin plugin**: default `fs.copy` (filesystem clone + test harness); `none` is verbal-only.

### 1.2 Three substrates

Keep these distinct. Mixing them is how designs fail.

| Substrate | Module home | Write | Correct | Good for |
|-----------|-------------|-------|---------|----------|
| **A working** | `loop/context.py` `EpisodeState` | Every step | Drop at episode end | What is true *now* (mode, traces, repaired-plan notes) |
| **B structured** | `store/` | Tools, extract + `admit()`, bookkeeping | Supersede / deprecate / reject | Facts, playbooks, error→fix, twin notes |
| **C parametric** | `sidecar/` | Batched PEFT after eval | Drop adapter row | Fluent skill that should work with less retrieve |

World and sim are **contact surfaces**, not memory buckets. Traces from either may propose writes to A/B/C under gates.

Rule of thumb: if you must edit it tomorrow with a sentence, it belongs in B. If it should change behavior when retrieval misses, consider C later. If it is only true for this run, keep it in A.

### 1.3 Timescale stack (wheels)

```
Throne     throne/policy.py     objectives, act/sim, admission, confirm, promote
Outer B/C  store/ + sidecar/    admit into B routinely; pack/train C rarely
Middle     loop/episode.py      one HTTP request = one episode
Inner      Host.generate        one tool-loop pass in the active rim sandbox
```

`decide()` in `throne/policy.py` is a pure function. Confirm, clone, and extract sit **outside** it.

### 1.4 Contact axis (dual rims)

```
                    INNER / MIDDLE
                          |
              +-----------+-----------+
              v                       v
         WORLD RIM                 SIM RIM
         project sandbox           cloned session dir
         world eyes                sim eyes (same graders + test harness)
              |                       |
              +-----------+-----------+
                          v
                 traces → extract → admit → B
```

- World session: host-reported (`payload.session_id` / `project-<id>` / thread sandbox).
- Twin plugin: `get_twin_plugin` (`fs.copy` default, `none` verbal-only). `episode.run` and probes do not call `clone_tree` themselves.
- Sim session (`fs.copy`): `sim-{episode[:8]}-{n}` (Studio) or `sim-{episode}-{n}` (FakeHost). Must not equal world and must not start with `project-`.
- Clone (`fs.copy`): `rims.clone.clone_tree` (`copytree` with `symlinks=True`, skip `.unsloth_sandbox` / remap json / `*.deleting-*`, refuse same resolved path and dest-inside-source). Failed spawn removes the sim session it created.
- Shared action interface: `Host.run_action(session_id, "python"|"terminal", …)` — same tools, different cwd.

Default act path (`MemoryWheels` §6, `decide()` at `throne/policy.py:83`):

1. World-act.
2. Recognized failure → `ENTER_SIM` (max 1 clone by default).
3. Sim until tests/tools pass or `max_sim_turns` (default 8 generate turns after an optional diagnostic).
4. `RETRY_WORLD` with a repaired-plan suffix (confirm optional).
5. Still fail → `ESCALATE` (extract, do not loop forever).

### 1.5 Record kinds, provenance, admission

Kinds (`constants.py`): `claim`, `procedure`, `error_fix`, `entity`, `episode`, `directive`, `twin_note`.

Statuses: `active`, `superseded`, `deprecated`, `proposed`, `rejected`.

Provenance (trust, lower weight first at retrieve): `world` 0, `mixed` 1, `human` 2, `sim` 3, `infer` 4.

Speaker (who asserted the row; runner-assigned, tools cannot self-certify `user`): `world`, `sim`, `user`, `model`, `other`. Optional `speaker_label` names which operator or document. `warrant` is internal proof; empty means unbacked.

Typology class (retrieve rank, lower first): WHAT world 0, WHAT mixed 1, WHAT sim 2, WHAT backed text 3, WHO user 4, WHO other 5, WHO model 6. `kind=directive` is always WHO user. Unbacked user/other claims are force-proposed. Retrieve drops a WHO hit that shares `normalize_title()` with a retrieved WHAT hit.

`admit()` total order is locked (`agents/admissions.py:46-67`):

1. namespace deny → rejected (**no row is inserted**)
2. namespace propose → proposed
3. `force_proposed_reason` (gate contradiction) → proposed
4. `bookkeeping` (deterministic twin_note, episode row) → active
5. sim claim **or** sim procedure → proposed
6. not explicit (extract) → proposed
7. `infer` (including directives) → proposed
8. `directive` unless provenance is `human` → proposed
9. else → active

Do not reorder this without a new phase lock. Operator promote is CLI `admit` (proposed/deprecated only unless `--force`).

Tool writes are further coerced before `admit()` (`tools/handlers.py`):

- `human` → `infer` (tools cannot self-certify a person)
- sim contact + claimed `world` → `sim`
- `kind=episode` is refused (the runner owns episode rows)
- `memory_supersede` cannot promote a proposed/rejected row to active
- unbound `dispatch` (no episode db) errors; it does not create `~/.unforgettable/memory.db`

### 1.6 What later phases parked

Phase 6 / research only: online distill, live serving-weight updates, attention-map match, productized parametric unlearn, auto twin calibration. Do not block the chariot on those. Ranked leftover product work (live LoRA attach, scheduled compact, DPO, UI, …) is in [`Roadmap.md`](Roadmap.md).

---

## 2. Boundaries with Unsloth

The Unsloth monorepo contains three license/role zones. Unforgettable is the Apache memory product.

| Zone | Path | License | Role vs Unforgettable |
|------|------|---------|------------------------|
| **This package** | `unforgettable/` | Apache 2.0 | Brains: store, policy, episode, tools, sidecar |
| **Studio face** | `studio/backend/core/unforgettable_host.py`, plus tiny hooks | AGPL | Host implementation, virtual-model route, tool dispatch, stream rewrite |
| **Unsloth core** | `unsloth/` | Apache 2.0 | Engine room for **C only**: `FastModel` (falls back to `FastLanguageModel`) `from_pretrained` + `get_peft_model` + TRL `SFTTrainer` / `DPOTrainer`, lazy-imported inside `sidecar/train.py` |
| **Studio trainer** | `studio/backend/core/training/` | AGPL | **Do not import.** Sidecar has its own recipe |
| **unsloth_cli** | `unsloth_cli/` | Apache | **Do not import.** |
| **Studio RAG** | `studio/backend` `rag.db` | AGPL | Unrelated. B is `memory/memory.db` |
| **Grok Build** | not in tree | — | Reference only (remember / search / inject / flush / dream-as-hygiene) |

### 2.1 Dependency arrow

```
studio  ──imports──►  unforgettable
unforgettable  ──must not import──►  studio, studio.backend, unsloth_cli
unforgettable.sidecar  ──lazy, inside UnslothTrainBackend only──►  unsloth, torch, trl
```

`tests/test_import_hygiene.py` is the tripwire: no `from studio` / `import studio` in production modules; sidecar package import must not load `unsloth` or `torch`.

### 2.2 Host protocol (the only Studio contract)

`unforgettable/host.py` defines `Host`. Studio implements `StudioHost`. Tests implement `FakeHost` (`tests/test_episode.py:55`). A future TUI implements `Host` and is done.

Required-in-practice methods:

| Method | Duty |
|--------|------|
| `memory_db_path()` | SQLite path |
| `world_session_id(request)` | World rim id |
| `create_sim_session(episode_id)` | New sim id + claimed dir |
| `sandbox_path(session_id)` | Absolute tree |
| `remove_sim_session(session_id)` | Teardown |
| `generate(GenerateRequest)` | One inner tool-loop pass; `session_id` selects the rim |
| `complete(messages, max_tokens=)` | One-shot, **no tools**, for LLM extract |

Optional (`getattr` skip):

| Method | Missing behavior |
|--------|------------------|
| `supervise(purpose, messages, model=)` | Voter abstains / planner skip / filter uses algo / judge uses algo |
| `run_action(...)` | No test command / no harness grade |
| `confirm(...)` | If confirm is required → `ESCALATE` |

`GenerateRequest` is a narrow DTO (not `ChatCompletionRequest`): messages, session, tools, `on_chunk`, `adapter_path`.

### 2.3 Studio call sites (AGPL, keep thin)

| File | What it may do |
|------|----------------|
| `studio/backend/core/unforgettable_host.py` | `StudioHost`, `handle_chat_completions`, sidecar prepare/restore, `supervise` one-shot. `install()` is called from `main.py`, not on import. |
| `studio/backend/core/unforgettable_stream.py` | Inner SSE rewrite / drain (`_rewrite_inner_frame`, `_forward_inner_stream`). Re-exported from the host for tests. |
| `studio/backend/core/unforgettable_patches.py` | `install()` wraps `execute_tool` (memory/rims `dispatch` + `note_tool_result`) and extends `_ALWAYS_SAFE_TOOLS`. Do not edit `tools.py`. |
| `studio/backend/routes/unforgettable.py` | Thin HTTP face over `operators.py` + store inspect + `/settings`. No status math. |
| `studio/backend/utils/unforgettable_settings.py` | Persisted episode defaults and voter knobs (`/api/unforgettable/settings`) |
| `studio/backend/routes/inference.py` | If `is_virtual_model(model)` and not `in_inner_generate()` → `handle_chat_completions`; catalog entry; union memory/rims specs on inner generate only |
| `studio/backend/core/inference/inference.py` | `generate_fn` calls `prepare_sidecar_adapter` / `restore_sidecar_adapter` around upstream `_apply_adapter_state` |
| `studio/backend/core/inference/tool_stream_exec.py` | Copies ContextVars into the tool worker so episode bind survives the thread |
| `studio/frontend/src/i18n/messages.ts` | Merges `features/unforgettable/i18n/en.ts` into each catalog at load |
| `studio/frontend/src/features/unforgettable/` | Dashboard, chat extras, settings tab loader, search keys, i18n overlay |

Policy, schema, clone, and admit **must not** grow in those files. If a function starts owning B or throne, move it back to Apache.

### 2.4 Data files (do not mix)

| File | Owner |
|------|--------|
| `$STUDIO_HOME/memory/memory.db` | Unforgettable B (+ packs/adapters tables) |
| `$STUDIO_HOME/memory/adapters/<id>/` | C LoRA dirs |
| `$STUDIO_HOME/rag/…` / `rag.db` | Studio RAG — never write here |
| chat message tables | Episode A on disk — not B |
| `~/.unforgettable/memory.db` | Headless / test default when no `UNFORGETTABLE_DB` / `STUDIO_HOME` |

### 2.5 Packaging

Root `pyproject.toml` includes `"unforgettable*"` in `setuptools.packages.find` and excludes `unforgettable.tests*` (plus `exclude-package-data` so tests do not ship as package data). The wheel license metadata stays Apache-2.0 for the project; Studio files keep AGPL SPDX headers. `unforgettable/LICENSE` makes the package relocatable.

Console script: `unforgettable = unforgettable.cli:main`. Relocatable `python -m unforgettable` stays.

Lint CI `compileall` / `ruff check` include `unforgettable/`. Backend CI runs the Apache suite (ignore `test_sidecar_gpu.py`) and puts the repo root on `sys.path` so Studio tests can `import unforgettable`.

Root `[tool.pytest.ini_options] testpaths = ["tests/security"]`. **Bare `pytest` at repo root does not collect this package.**

---

## 3. Source files

Production modules only. Tests are listed in §5. Plans under `plans/` are charter, not runtime.

### 3.1 Package root

| File | Description |
|------|-------------|
| `__init__.py` | Version `0.1.0`; `VIRTUAL_MODEL_ID`, `is_virtual_model`, `inner_model_id` |
| `__main__.py` | `python -m unforgettable` → `cli.main` |
| `cli.py` | argparse inspect / compact / compile / pack / train / eval / promote / rollback / review / mine. Also the `unforgettable` console script. `compact` and `pack` preview unless `--apply`. Thin wrap of `operators.py`. |
| `operators.py` | Shared admit / reject / review / mine / compile / promote / `summarize_store` used by CLI and Studio. Does not change `admit()` predicates. |
| `constants.py` | Kinds, statuses, provenances, speakers, typology class, namespace defaults, `PROVENANCE_WEIGHT` |
| `host.py` | `Host` protocol, `GenerateRequest` / `GenerateResult` / `ToolTrace`, run-action / supervise constants |
| `supervisor.py` | Approver (vote/mine), planner, filter (algo ∪ LLM), judge: config, prompts, parse, `HttpSupervisor` |
| `LICENSE` | Apache 2.0 text |
| `README.md` | User-facing overview |
| `TECHNICAL.md` | This file |

### 3.2 `store/` — B notebook

| File | Description |
|------|-------------|
| `__init__.py` | Public store façade |
| `db.py` | Per-call SQLite connections, WAL, `foreign_keys=ON`, lazy `ensure_schema` |
| `schema.py` | `namespaces`, `records`, FTS5, `admissions_log`, `rollouts`, `compiled`, `compiled_blocked`, `retrieve_uses`, `inject_stats`, `packs`, `pack_items`, `adapters` |
| `records.py` | CRUD, supersede (one transaction), deprecate, admissions log, rollouts, retrieve_uses, inject_stats |
| `search.py` | FTS5 MATCH (AND + stopwords), then typology-class then provenance-weight sort |
| `titles.py` | `normalize_title()` shared by compact and gate eyes |
| `compact.py` | Deterministic hygiene: empty proposed, stale proposed WHO/infer, title-dedupe, fold superseded chains |
| `compile.py` | Standing membership cache; live format of B; sticky `uncompile` |
| `trajectories.py` | Graded rollout retrieve (episode FTS + join); never injects episode bodies |

### 3.3 `agents/` — internal B maintainers

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports admit / retrieve / extract |
| `admissions.py` | Locked `admit()` predicate table + log |
| `retriever.py` | `RetrievePolicy`, char budget, stakes, sim lesson bias, format inject |
| `extractor.py` | Naive fail→success, twin-drift, episode summary, bounded `llm_extract` |

### 3.4 `tools/` — OpenAI function surface

| File | Description |
|------|-------------|
| `__init__.py` | `MEMORY_TOOLS`, `CONTACT_TOOLS`, `dispatch` |
| `specs.py` | Function JSON schemas (underscore names) |
| `handlers.py` | Dispatch: write/search/get/supersede/deprecate/compact/compile + `rims_enter_sim` |

### 3.5 `loop/` — middle wheel

| File | Description |
|------|-------------|
| `__init__.py` | `run`, `EpisodeRequest` / `EpisodeState`, `note_tool_result` |
| `context.py` | Request/state dataclasses; last-user text; created-sim tracking; planner flags |
| `episode.py` | `run()`: inject, optional planner, generate loop, twin.spawn_sim, harness, confirm, extract, probes, teardown |
| `runtime.py` | ContextVars for db / episode / namespace / contact / traces |

### 3.6 `rims/` — contact surfaces

| File | Description |
|------|-------------|
| `__init__.py` | `clone_tree`, `ContactMode`, twin plugin registry |
| `types.py` | `ContactMode = "world" \| "sim"` |
| `plugin.py` | `Location`, `TwinBinding`, `TwinPlugin`, `get_twin_plugin` |
| `fs_copy.py` | Reference plugin: sandbox clone via `clone_tree` |
| `none.py` | Verbal-only plugin; no copy |
| `clone.py` | Pure filesystem clone; same-path refuse |
| `detect.py` | Test-command resolution + tree detector (`pytest` / `npm test` / `go test`) |

### 3.7 `throne/`

| File | Description |
|------|-------------|
| `__init__.py` | Policy façade |
| `policy.py` | `Action`, `Policy`, `require_confirm_retry`, `policy_from_request`, `decide` |

### 3.8 `eyes/`

| File | Description |
|------|-------------|
| `__init__.py` | Eyes façade |
| `protocols.py` | `RecognizedFailure`, `WorldEyes` / `SimEyes` / `GateEyes`, `Contradiction` |
| `basic.py` | Tool-result eyes, runner fingerprints, user phrases, `grade_run_action` (phrases stay algo; episode may overlay `judge_model`) |
| `gate.py` | Title-contradiction scan, WHO-vs-WHAT dissonance, `review_write`, `LogGateEyes` |
| `probes.py` | `Probe:` procedures; CLI and post-extract run |

### 3.9 `sidecar/` — C

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports `pack_from_admitted_b` only; no Unsloth import |
| `pack.py` | Eligibility, votes, holdout-by-episode, persist |
| `format.py` | Title→body SFT messages; world fail/pass preference pairs |
| `train.py` | `TrainBackend`, `FakeTrainBackend`, lazy `UnslothTrainBackend`, `train_pack` |
| `export_gguf.py` | PEFT dir → GGUF LoRA via `convert_lora_to_gguf.py`; lazy Unsloth |
| `eval.py` | Holdout lean score vs base (prefix match; optional `judge_model` overlay); optional probes |
| `adapters.py` | Registry: shadow → promote / discard / rollback; optional `gguf_path` |

### 3.10 `plans/` (not imported)

`MemoryWheels.md`, `MemoryPhases.md`, `MemPhase1.md`–`MemPhase5.md`. Charter and PR history. Runtime code must not depend on them. Current cadence and deferred work: [`Roadmap.md`](Roadmap.md) (package root, also not imported).

---

## 4. Implementation overview

### 4.1 Store and schema

`get_connection` (`store/db.py:37`) opens a per-call connection, enables WAL + FK, and runs `ensure_schema` once per resolved path under a lock.

`records` is the soul of B (Phase 1). Later phases add tables with `CREATE TABLE IF NOT EXISTS` — no `ALTER` on `records` except additive missing columns (`schema.py` `_add_missing_columns`, including `speaker`, `speaker_label`, `warrant`). FTS5 (`record_fts`) indexes title+body; `record_id` is stored UNINDEXED and maintained in Python (`records._rewrite_fts`).

Supersede (`records.supersede_record`, `store/records.py`) marks the old row superseded and inserts the successor **in one connection**. Handlers run `review_write` + `admit` first and pass the resulting status (`tools/handlers.py` `_supersede`). A proposed row cannot be laundered to active by superseding, even with a more trusted provenance. Rejected rows cannot be superseded. Title/body are clipped to `RECORD_TITLE_CHARS` / `RECORD_BODY_CHARS`.

`set_record_status` to `active` strips trailing `[deprecated] …` suffixes so CLI `admit` after compact does not re-inject the scar.

### 4.2 Retrieve and standing compile

Default retrieve kinds exclude `episode` (`agents/retriever.py` `DEFAULT_RETRIEVE_KINDS`). Search tokenizes the query, drops a small stoplist, and ANDs quoted terms (`store/search.py` `_match_query`). Hits are filtered, then sorted by typology class, then `PROVENANCE_WEIGHT`, then FTS rank. `retrieve()` then drops WHO rows whose normalized title collides with a WHAT hit in the same bundle.

`RetrievePolicy` (`retriever.py:43`): `max_records=6`, `max_chars=2400`, `snippet_chars=280`, `high_stakes` (world only drops sim/infer), `max_twin_notes` (1 world / 3 sim), `contact`, `exclude_ids`. After FTS, compiled ids are dropped; `top_k` over-fetches by `len(exclude_ids)` so standing members do not starve retrieve.

Compiled form is **membership + live format**, not a stored skill blob (`store/compile.py`). A standing section always ends with `Source: {uuid}`. Eligibility: active `procedure`, provenance in `{world, mixed, human}`, not a `Probe:` title, not the `test command` nomination. Auto-pin after two distinct world-retrieve + world-pass episodes (`procedure_hits`). `unpin_compiled` inserts `compiled_blocked` so `maybe_compile` will not silently re-pin; explicit `compile ID` clears the block.

`loop/episode.py` `_inject_bundle` assembles preamble + standing + retrieve + trajectories and writes `retrieve_uses` + `inject_stats`. CLI `load` prints the char split.

On `ENTER_SIM` / `RETRY_WORLD` the bundle is rebuilt (sim retrieve turns off the high-stakes provenance filter and prefers sim/mixed lessons). `CONTINUE_SIM` does not FTS again; it does refresh the repaired-plan suffix.

### 4.2.1 Data path (context → B → C)

```
generate / tool result
    │
    ├─ note_tool_result (A traces; memory_* skipped by eyes)
    ├─ note_success / note_failure  → first line, EVENT_SUMMARY_CHARS
    │
    ├─ memory_write / memory_supersede
    │     coerce provenance → review_write → admit → insert (or refuse)
    │
    └─ _extract at episode end
          from_episode (proposed error_fix from fail→success grades)
          from_drift   (active twin_note if sim ok + later world fail)
          llm_extract  (proposed infer; no args in the prompt; no directive/episode)
          episode_summary (active bookkeeping pointer; last 200 user chars)
          rollouts     (last event per contact; summary clipped)
          maybe_compile (needs 2 world-retrieve + world-pass hits)

CLI admit (proposed/deprecated → active)
    │
    ▼
pack_from_admitted_b
    active procedure/error_fix + world|mixed|human
    + (compiled membership OR world-pass retrieve vote)
    include_sim votes need same-episode world/pass and no *active* twin_note
    gold = title → body (PACK_BODY_CHARS); never episode rows
    │
    ▼
train_pack (train role only) → eval holdout vs base → promote
```

Chat completions and tool argument blobs are not training gold. Preference pairs are taken only from pack **train** episode ids.

### 4.3 Episode loop

`run()` (`loop/episode.py`) is the middle wheel:

1. Bind ContextVars (`loop/runtime.py` `bind_episode`).
2. Resolve optional `adapter_id` → `adapter_path` + train-source exclude set (`_resolve_attached_adapter`). Missing/discarded adapter is ignored.
3. If `EpisodeRequest.planner` is on, one `Host.supervise("plan")` writes a temporary A suffix (fail-open). Refreshed once on `RETRY_WORLD`. Not B.
4. First-turn `_rebuild(contact="world")`.
5. Before the first world generate: closed user-phrase list (`that failed`, …) can enter sim with zero world generates.
6. `host.generate` in the active session. `_pass_failure` skips `memory_*`, treats `rims_enter_sim` as fail-wins, then `inspect_tool_result` (traceback, `Error:`, Studio sentinels, runner fingerprints, exit-code). Success/failure events keep a clipped first line (`EVENT_SUMMARY_CHARS`), never the raw completion.
7. In sim, if a test command is known and `run_action` exists, **grade wins** over “I fixed it” (`_maybe_run_sim_tests` + `grade_run_action`).
8. `decide` then optional `confirm` (`_confirm_retry_world`).
9. `ENTER_SIM`: `create_sim_session`, `track_sim`, refuse bad ids, `clone_tree`, resolve test command, rebuild inject, optional diagnostic `run_action` (does not burn a sim turn).
10. `RETRY_WORLD` / `CONTINUE_SIM`: rebuild inject + `_repair_context` (last failure, sim grade, clipped last generate text). World retry does **not** copy sim files onto world.
11. `_extract`: `from_episode` (prefers last **world** success), `from_drift`, optional `llm_extract` via `Host.complete`, episode row, last-fail and last-pass rollouts per contact, `maybe_compile`. `keep_sim` only for **active** `error_fix` or **active** `twin_note` on this episode.
12. Optional ≤3 `Probe:` runs on a **fresh** clone of current world.
13. `finally`: remove every created sim except the kept one; `reset_episode`.

`llm_extract` (`extractor.py`) is bounded: last 24 non-memory traces / 8k chars, max 8 drafts, forced `provenance=infer`, parse-fail → `[]`. Each trace is `name` + clipped result — **not** tool arguments. Forbidden kinds: `directive`, `episode`. Skip when fewer than two non-memory traces and no failure events.

### 4.4 Eyes and throne

`inspect_tool_result` / `grade_run_action` (`eyes/basic.py`) share fingerprints. Studio world strings (`Execution timed out after`, `Execution cancelled.`, `Blocked command(s) for safety:`, `Execution error:`, `No command provided.`) are failures on `python`/`terminal`, not success.

`review_write` (`eyes/gate.py`) forces proposed when a new **claim** or **procedure** shares a normalized title with an active peer of a different normalized body, when a WHO row shares a title with an active WHAT row (`dissonance: contradicts {id}`), or when an unbacked user/other claim has no warrant. Operator `admit` of a still-colliding WHO row needs `--force`.

`require_confirm_retry` (`throne/policy.py:46`): `confirm_retry is False` wins; else True; else `stakes=="high"`; else `permission_mode=="ask"`. Studio product default `auto` does **not** show a retry card.

### 4.5 Sidecar C

`pack_from_admitted_b` (`sidecar/pack.py`): only active `procedure`/`error_fix` with trusted provenance; probes and `test command` dropped; traces **vote** (and hold out by episode); they do not donate episode text. Compiled membership is also a vote (operator `compile` or auto-pin after two world-pass hits). `include_sim` default false; sim/pass votes only with world/pass and no **active** twin_note. Sim rows never become assistant gold. Empty candidate sets are not persisted. `pack_is_retrieval_heavy` counts compiled rows without refreshing/unpinning them.

`train_pack` refuses fewer than `PACK_MIN_TRAIN` (4) train items. Fake backend writes `adapter_config.json` + `fake_gold.json`. Unsloth backend lazy-imports `FastModel` (falls back to `FastLanguageModel`), prefers `text_only=True`, refuses `UNSLOTH_ENABLE_FULL_FINETUNING=1` and `full_finetuning=True`, writes a LoRA dir. Preference recipe: Fake writes `pairs.jsonl` from `preference_pairs(..., train_episode_ids=)` — only episodes that appear on **train** pack items. Unsloth preference flattens those pairs to string `prompt` / `chosen` / `rejected`, lazy-imports TRL `DPOTrainer` **after** Unsloth (same LoRA knobs as SFT, `ref_model=None`), and writes the LoRA dir plus `pairs.jsonl`. Missing `trl` raises `RuntimeError("preference recipe needs trl.DPOTrainer")` before any Unsloth import. SFT/distill do not import DPO.

`eval_adapter`: holdout title-only complete vs gold. Empty completions on both sides (`adapter_lean == base_lean == 0`) **fail**. CLI `eval` selects Fake vs Unsloth from `adapters.backend` and passes `base_model` into `UnslothTrainBackend.complete` (load base, then PEFT adapter).

`promote_adapter` refuses without metrics / without `passed`. One promoted row; previous promoted → discarded. `rollback_adapter` discards current; does not repromote. Files are never deleted by the registry.

Live Studio inject does **not** shrink on promote alone. Shrink + `GenerateRequest.adapter_path` only when `EpisodeRequest.adapter_id` is set. `StudioHost.generate` copies a PEFT adapter directory onto `payload.use_adapter`; the inference worker `load_adapter`s it for that generate and restores the previous adapter. Fake sidecar dirs and GGUF inners fail open (no attach).

### 4.6 ContextVars and Studio threads

`bind_episode` sets `_db_path`, `_episode_id`, `_namespace`, `_traces`, `_contact`. `execute_tool` calls `note_tool_result` and `dispatch` (which reads `current_db_path()`). Unbound `dispatch` returns an error instead of creating `~/.unforgettable/memory.db`. Studio’s inner loop runs tools in `stream_tool_execution` on a worker thread (`studio/backend/core/inference/tool_stream_exec.py`). That thread is started with `contextvars.copy_context().run` so the bind survives. `StudioHost.run_action` wraps `asyncio.to_thread` in `copy_context()` as well (needed on Python 3.9–3.10). Apache `tests/test_runtime_context.py` locks copy vs raw thread. Memory/rims tools are stripped from ordinary Studio chat (`_select_request_tools` when `not in_inner_generate()`). Non-stream confirm cannot show a card and therefore escalates if confirm is required.

`StudioHost.run_action` uses `asyncio.to_thread` (copies context). Extract uses `complete`, not `generate`, so it cannot re-enter the act/sim loop. `supervise` is the same one-shot path with an optional larger `model` (`planner_model` / `voter_model` / `filter_model` / `judge_model`).

### 4.7 Supervisor (approver + planner + filter + judge)

Not the MemoryWheels outer wheel. Jobs are separately configured. Neither trains the large model. Neither reopens `admit()` or `decide()`.

| Job | Config | Call site | Effect |
|-----|--------|-----------|--------|
| **Approver** | `UNFORGETTABLE_VOTER=off\|advisory\|binding` + `UNFORGETTABLE_SUPERVISOR_URL` | CLI `admit` / `compile` / `promote` / `review` / `mine` | Votes after local select, before promote. Binding deny blocks unless `--force`. `episode` rows are skipped. New mine drafts stay `proposed` `infer`. |
| **Planner** | `EpisodeRequest.planner` (`on`/`off`); Studio copies payload or `UNFORGETTABLE_PLANNER` | `episode.run` before first generate; refresh on `RETRY_WORLD` | Temporary A suffix. Fail-open. Not written to B. |
| **Filter** | `EpisodeRequest.filter` (`on`/`off`, default on); `UNFORGETTABLE_FILTER` | `episode.run` before first generate; cached spans on `memory_write` | Closed-list algo always runs; a parsed LLM reply **adds** spans (union). `kept` is recomputed from the original so the LLM cannot restore algo strips. Empty remainder → ENTER_SIM + confirm. Not an LLM rewrite of compact. |
| **Judge** | `UNFORGETTABLE_JUDGE_MODEL` / `judge_model` (default unset) | sidecar `eval_adapter` holdout scores; `episode.run` user-failure paraphrase | LLM if configured and the reply parses; else prefix-match eval / closed `user_declares_failure` list. Parse-fail does not ENTER_SIM. |

`HttpSupervisor` POSTs `{purpose, model, messages, max_tokens}` and reads `{text}`. `StudioHost.supervise` uses the loaded inner generate with tools off.

---

## 5. Automated tests

Apache suite: **299** CPU tests under `unforgettable/tests/` (ignore `test_sidecar_gpu.py`; the ledger-week file is marked `scenario` + `slow` and is deselected by root addopts), no GPU, tmp SQLite + tmp dirs. Fixture: `conftest.py` `db_path` → `tmp_path / "memory.db"`. `test_sidecar_gpu.py` is marked `gpu` and skips unless CUDA torch and cached `--base` weights are present. `tests/scenario/test_ledger_week.py` is an opt-in integration week: scripted inner, real unittest world judge, B under load, pack + fake C at the end.

| File | What it locks |
|------|----------------|
| `test_import_hygiene.py` | No `studio` imports; sidecar import does not load `unsloth`/`torch`; no module-level Unsloth in sidecar except indented `train.py`; SFT import excludes DPO |
| `test_virtual_model.py` | `unforgettable` / `unforgettable/<id>` alias strip; nested `unforgettable/unforgettable` |
| `test_store.py` | Schema CRUD, supersede history, deprecate hidden from search, world > infer rank |
| `test_admissions.py` | Locked `admit()` order including bookkeeping vs force_proposed vs sim procedure |
| `test_gate.py` | Contradiction `review_write`, conflicting write stays proposed, WHO-vs-WHAT dissonance, admissions log |
| `test_retrieve.py` | Char budget, high-stakes, episode exclusion, twin cap, sim contact, exclude compiled, WHO collision drop |
| `test_compact.py` | Dedupe kinds, namespace isolation, empty proposed age, stale proposed infer, fold chains, dry-run |
| `test_compile.py` | Hits, probes/test-command/sim/proposed refuse, sticky uncompile, refresh after deprecate, standing caps |
| `test_trajectories.py` | Episode FTS join, ranking, high-stakes drop, no episode body, cap 2 |
| `test_extract.py` | LLM drafts proposed infer, parse-fail, provenance overwrite, skip without `complete` |
| `test_eyes.py` | Runner fingerprints vs “failed to import”, enter_sim, user phrases, Studio sentinels |
| `test_rims_action.py` | Detector order, resolve requested vs procedure vs tree |
| `test_rims_throne.py` | Clone ignore list, same-path refuse, dest-inside-src refuse, symlink copy, `decide()` fail→sim→retry, confirm matrix |
| `test_probes.py` | Prefix identity, CLI `--run`, episode cap 3, skip without sim/`run_action` |
| `test_tools.py` | Tool names, CRUD, admit log id, supersede stays proposed, compact/compile dry-run defaults, compile-with-id needs hits |
| `test_supervisor.py` | Vote/mine/filter parse, algo filter + union, judge parse, env config, request-scoped planner, binding vs advisory, HTTP POST shape |
| `test_cli.py` | Subcommands, `--db`, compact/pack `--apply`, admit `--force`, voter admit/review/mine, fake train, honest eval, promote/rollback |
| `test_episode.py` | Happy path + drift + enter_sim + user phrase + harness + timeout + confirm + standing + re-retrieve + compile + adapter shrink; FakeHost |
| `test_stream_forward.py` | `on_chunk` forwarded into `GenerateRequest` |
| `test_runtime_context.py` | `copy_context` carries db + traces; raw thread does not |
| `test_sidecar_pack.py` | Drop reasons, no episode gold, sim vote matrix, rejected twin_note is not a veto, holdout-by-episode, preference pairs |
| `test_remember_path.py` | Unbound dispatch, no human/episode from tools, sim cannot mint world, supersede does not promote, generate-text clip |
| `test_sidecar_train.py` | Min size, fake shadow dir, full-FT refuse, preference pairs.jsonl, unpacked episode is not preference gold, missing-TRL DPO refuse, flattened DPO rows, stubbed Unsloth DPO |
| `test_sidecar_gpu.py` | Marked `gpu`. CUDA + cached `--base` only: SFT writes a real PEFT dir and `complete()` returns a string (adapter and base); preference writes PEFT + `pairs.jsonl`. Skips without GPU or weights |
| `test_sidecar_eval.py` | Seeded holdout 1.0 vs 0.0; **unseeded holdout fails**; empty holdout fails; optional judge score overlay |
| `test_sidecar_adapters.py` | Promote gate, one promoted, rollback keeps files, probe-fail refuse |
| `tests/scenario/test_ledger_week.py` | Marked `scenario` and `slow`. Multi-episode ledger week through `episode.run`: fail→sim→world retry, WHO/WHAT retrieve, filter, planner, twin_note, standing, compact, pack ≥16 train + holdout, fake SFT + preference, eval. Not CI |

Names that later phases must keep green: `test_episode_fail_sim_retry_writes_error_fix`, `test_episode_sim_ok_world_retry_fail_writes_twin_note`, `test_retrieve_injects_before_generate`, `test_episode_standing_excludes_from_retrieve`, `test_episode_re_retrieve_on_enter_sim`, `test_episode_maybe_compile_after_second_world_pass`.

### 5.1 Studio-face tests (AGPL)

`studio/backend/tests/test_unforgettable_stream.py` — `_rewrite_inner_frame` (tool frames unchanged, `finish_reason` nulled, inner `[DONE]` dropped), stream drain/`aclose`, enabled-tools union, `run_action`/`confirm` SSE, ContextVar copy through `stream_tool_execution`, PEFT `use_adapter` attach. Helpers live in `unforgettable_stream.py` and are re-exported from the host. Needs Studio backend on `PYTHONPATH` (repo root too, for `import unforgettable`) and Studio Python extras (`structlog`, …).

Frontend: `studio/frontend/tests/unforgettable-merge-extras.test.ts` — virtual-model extras merge and settings search index. Overlay: `unforgettable-i18n-overlay.test.ts`. Studio settings/routes: `studio/backend/tests/test_unforgettable_settings.py`, `test_unforgettable_routes.py`.

---

## 6. Build and test

### 6.1 Layout and install

From the Unsloth repository root (this package is not published separately):

```bash
# Editable install of the monorepo (pulls unforgettable* via setuptools include)
python -m pip install -e .

# or with uv
uv pip install -e .
```

Python: `>=3.9,<3.15` (`pyproject.toml`). Apache Unforgettable itself needs only the stdlib (and pytest to run tests). SQLite must have FTS5 (default CPython).

Optional extras:

- **Studio chat face:** install the repo’s Studio extra / follow Studio setup so `studio/backend` imports work.
- **Real C training:** Unsloth + GPU + `trl` + `datasets` + `peft`. Not required for `train --backend fake` or for CI.

### 6.2 Import smoke (no pytest)

```bash
python -c "import unforgettable, unforgettable.sidecar, unforgettable.loop.episode; print(unforgettable.__version__)"
python -c "import sys, unforgettable.sidecar; assert 'unsloth' not in sys.modules and 'torch' not in sys.modules"
python -m unforgettable --help
unforgettable --help   # after editable install; same as python -m
```

`git grep` of `unforgettable/*.py` (excluding tests) must not show `from studio` or `import studio`.

### 6.3 Apache test suite (required, no GPU)

Root `pytest` is **not** enough: `[tool.pytest.ini_options] testpaths = ["tests/security"]`.

```bash
# from repo root; pythonpath already includes "."
python -m pytest unforgettable/tests --ignore=unforgettable/tests/test_sidecar_gpu.py

# quieter / fail-fast
python -m pytest unforgettable/tests --ignore=unforgettable/tests/test_sidecar_gpu.py -q --tb=short

# one module
python -m pytest unforgettable/tests/test_episode.py -q
```

Expect the Apache suite green (the GPU file is marked `gpu` and the ledger week is marked `slow`/`scenario`, so default addopts deselects both; CI also `--ignore=`s the GPU file). Runtime is a few seconds on a laptop. On a CUDA box with `unsloth/Qwen3.5-4B` cached:

```bash
python -m pytest -o addopts= unforgettable/tests/test_sidecar_gpu.py
```

Optional CPU/B ledger week (must override addopts, same pattern as gpu):

```bash
python -m pytest -o addopts= -m scenario unforgettable/tests -s

# keep memory.db + pack JSONL + chronicle for a later GPU C job
UNFORGETTABLE_SCENARIO_OUT=/tmp/ledger-week python -m pytest -o addopts= -m scenario unforgettable/tests -s
```

SFT (including `complete()`) and preference should pass in well under two minutes. Root addopts is `-m 'not gpu and not slow'`; `-m gpu` or `-m scenario` ANDs with that and collects nothing, so override addopts as above.

If the environment has no `pytest` in the project venv:

```bash
uv pip install pytest
# or: python -m pip install pytest
```

### 6.4 Studio-face tests (optional here, required if you touch AGPL wiring)

Needs Studio’s import path and its Python deps (`structlog`, `huggingface_hub`, … — whatever `studio/backend` imports at collection time).

```bash
PYTHONPATH=studio/backend python -m pytest studio/backend/tests/test_unforgettable_stream.py -q
```

If collection fails on missing Studio extras, install Studio’s backend requirements (see `studio/backend/requirements/` and Studio README). Do not weaken `test_import_hygiene` to make this pass.

### 6.5 Manual Studio smoke

1. Start Studio with a loaded inner model.
2. `GET /v1/models` includes `id: unforgettable`.
3. `POST /v1/chat/completions` with `model: unforgettable`, `stream: true` yields multiple `delta.content` events and live `tool_start`/`tool_end` before a single `[DONE]`.
4. After a coding fail, a `sim-*` sandbox appears; after clean proposed-only success it is removed.
5. `python -m unforgettable --db "$STUDIO_HOME/memory/memory.db" list` shows episode / proposed extract rows.

### 6.6 Sidecar on a GPU box (optional)

```bash
python -m unforgettable --db "$STUDIO_HOME/memory/memory.db" pack --dry-run
python -m unforgettable --db "$STUDIO_HOME/memory/memory.db" pack --apply
python -m unforgettable --db "$STUDIO_HOME/memory/memory.db" train --backend unsloth --base <hf-or-local-id>
python -m unforgettable --db "$STUDIO_HOME/memory/memory.db" train --backend unsloth --base <hf-or-local-id> --recipe preference
python -m unforgettable --db "$STUDIO_HOME/memory/memory.db" eval <adapter-id>
python -m unforgettable --db "$STUDIO_HOME/memory/memory.db" promote <adapter-id>
```

`--base` is required for `--backend unsloth` (exit 2 otherwise). Full fine-tune is refused. CI must use `--backend fake` and must not construct `UnslothTrainBackend.train` against a real model.

### 6.7 What “green” means

- `unforgettable` imports with no Studio on `sys.path`.
- `import unforgettable.sidecar` does not import `unsloth` or `torch`.
- `pytest unforgettable/tests --ignore=unforgettable/tests/test_sidecar_gpu.py` 285 passed, 1 deselected (ledger week).
- FakeHost happy path: world fail → sim → world ok → `error_fix` **proposed**, sim **removed**.
- FakeHost drift path: sim ok + world retry fail → active `twin_note`, sim **kept**.
- Empty adapter set / no `adapter_id` leaves Phase 4 inject unchanged.

When adding code: new brains go in `unforgettable/` (Apache header). Studio only grows if the public face needs it (stream, catalog, payload fields, later UI). If a Studio function starts owning schema or policy, move it back.
