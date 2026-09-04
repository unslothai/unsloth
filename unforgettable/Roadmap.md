# Unforgettable — roadmap

This note is the living “what’s next” for the package. Architecture that is shipped lives in [`TECHNICAL.md`](TECHNICAL.md). User-facing overview: [`README.md`](README.md). Design history and phase locks stay in [`plans/`](plans/) — those files are the charter, not a to-do list.

**Read this file, not `plans/MemoryPhases.md`, for cadence.** That roadmap was written when Phase 1 had just shipped and still says Phase 2 is next.

---

## Current status

**Phases 1–5 are implemented on the tree. Phase 6 is open and parked.** There is no `MemPhase6.md`. Do not treat Phase 6 as the next product slice.

| Phase | Plan | MemoryWheels | Status |
|-------|------|--------------|--------|
| 1 Bones | [`plans/MemPhase1.md`](plans/MemPhase1.md) | A1–A2, crude A5–A6 | **Complete.** Apache package, B store, memory tools, clone + `decide()`, Studio virtual model. |
| 2 Notebook | [`plans/MemPhase2.md`](plans/MemPhase2.md) | A3, A4 v0, A7, stream, CLI, compact | **Complete.** Live stream, twin-notes, inspect CLI, retrieve budget, LLM extract, gate v0, episode + rollouts. |
| 3 Motion | [`plans/MemPhase3.md`](plans/MemPhase3.md) | A5–A6 for real, A4 v1 | **Complete.** `run_action` harness, richer eyes, confirm-before-retry, rim hygiene, `Probe:` suite. |
| 4 Leaner inner | [`plans/MemPhase4.md`](plans/MemPhase4.md) | A8–A9 | **Complete.** Standing compile, trajectory library, re-retrieve + sim bias, `inject_stats`. |
| 5 Side car C | [`plans/MemPhase5.md`](plans/MemPhase5.md) | B10–B13 | **Complete as product.** Pack → fake/Unsloth train → eval → promote/rollback. Standing shrinks when `adapter_id` is set. |
| 6 Research | [`plans/MemoryPhases.md`](plans/MemoryPhases.md) §Phase 6 | C14–C18 | **Open / parked.** No schedule. Do not block the chariot on it. |

`TECHNICAL.md` is the source of truth for what the tree actually does. Plans stay as locks and history. A few Phase 4/5 “optional later” one-liners already landed after the phase docs froze (`adapter_id` and `skip_standing` are first-class Studio payload fields and are copied onto `EpisodeRequest`).

### Shipped after Phase 5

These are supervisor jobs, **not** a rename of the MemoryWheels outer wheel (B + C). The local actor remains the only PEFT target. `admit()` and `decide()` stay code.

| Feature | Status | What shipped |
|---------|--------|----------------|
| **Approver model** | **Complete.** | Optional vote after local select, before promote. `UNFORGETTABLE_VOTER=off\|advisory\|binding`. CLI `admit` / `compile` / `promote` wrap a vote; `review` batches proposed rows; `mine` is the same voter over proposed + rollouts + admissions and may insert new **proposed** drafts. Binding deny blocks unless `--force`. `episode` rows are skipped. Host seam: `supervise("vote"\|"mine")` or `HttpSupervisor` at `UNFORGETTABLE_SUPERVISOR_URL`. |
| **Planner model** | **Complete.** | Runtime-selectable temporary overlay. `EpisodeRequest.planner=on` (Studio payload or `UNFORGETTABLE_PLANNER`). One `supervise("plan")` before the first generate; refresh on `RETRY_WORLD` only. Injected as working-memory A. Fail-open. Not written to B. |
| **Filter judge** | **Complete.** | `supervise("filter")` before first generate, unioned with a closed-list algo. Strips coercive and manipulative spans; keeps the technical remainder. Empty remainder enters sim and requires confirm. Default on. Missing/empty LLM uses the algo (does not fail-open skip). Compact still does not LLM-rewrite. |
| **Judge model** | **Complete.** | Optional `UNFORGETTABLE_JUDGE_MODEL` / Settings `judge_model`. Overlays sidecar holdout scoring and user-failure paraphrase. Unset keeps prefix-match eval and the closed phrase list. |
| **Twin plugin** | **Complete.** | Location+tools contract. Default `fs.copy` is the old sandbox clone. `none` rehearses in text with no copy. `UNFORGETTABLE_TWIN` / payload `twin_plugin` / Settings. Episode and probes no longer call `clone_tree` directly. |
| **PEFT + GGUF LoRA** | **Complete as artifacts.** | Unsloth train writes PEFT (source of truth) and tries a GGUF LoRA next to it. CLI `export-gguf`. Transformers/MLX attach PEFT live. GGUF is load-time `--lora` (no mid-chat llama-server restart). |

### What “complete” still left named

These are **not** new architecture. They are leftovers the phase docs parked on purpose, plus a couple of wires that exist on one side only.

| Leftover | Where it sits today |
|----------|---------------------|
| Compact is explicit-only | CLI / `memory_compact`. Never at episode end, never on a timer. |
| Extract stays proposed | Naive `error_fix` and `llm_extract` drafts do not auto-admit. Operator `admit` is the promote path. |

### Locks that later work must not reopen

- `admit()` total order (`TECHNICAL.md` §1.5 / MemPhase2 Decision 1).
- License arrow: brains in Apache `unforgettable/`; Studio stays a thin face.
- No Studio RAG / chat history as B. No Grok Build in the tree.
- Base weights stay frozen. No full FT. No merge-to-base.
- Compact is deterministic hygiene, not an LLM rewrite of the store.
- C is an operator CLI, never an inner-model `memory_train` firehose.

---

## Future improvements (easiest first)

Sorted by implementation effort against the bones that already exist — low-hanging fruit first. Source citations point at the plan sentence that parked the item. Items that are easy to type and expensive to get wrong are marked **policy-sensitive**.

Permanent anti-patterns (do not put on a sprint) are listed at the end, not here.

### 1. Hours — packaging and one-liners

**1. `console_scripts` entry** — **Done.**
`unforgettable = unforgettable.cli:main` in root `pyproject.toml`. Relocatable `python -m unforgettable` stays.

**2. Confirm the model picker lists `unforgettable`**
MemPhase1 frontend follow-up: if the desktop picker filters to real GGUF/safetensors, add a one-line allow for the synthetic id. Catalog already emits it. Check first; do nothing if it already appears.

**3. Rim-switch content markers**
MemPhase2 stream design: optional `\n[sim]\n` / `\n[world retry]\n` via existing `on_chunk`. A few lines in `episode.run`. Do not invent a new SSE type.

**4. `adapters gc` CLI**
MemPhase5: “Optional later `adapters gc` is out. Operator can `rm -r` a discarded dir.” Rollback never deletes files. A `gc` that removes **discarded** adapter directories (never promoted, never shadow-in-use) is a small CLI + test. Default dry-run.

**5. Use the unused `confidence` column**
Schema has `records.confidence`. Tools do not accept it; retrieve does not rank on it. MemoryWheels §15 still lists “confidence” as an open schema question. Smallest useful form: optional `memory_write` field + retrieve tie-break after provenance. Do not invent a scoring model.

**6. `Host.complete` token budget**
MemPhase2 open question 3: always 800 vs the user’s `max_tokens`. Default 800 is fine until extract quality is measured. Changing it is a one-argument copy. Measure first.

### 2. One small PR — leftover product from Phases 2–5

**7. Auto-admit world/mixed `error_fix` after evidence** — **policy-sensitive**
Parked in MemPhase2, 3, 4, and 5 as “a later tiny PR … one-line `admit()` change.” Only if a week (or more) of proposed rows shows the naive fail→success writer is consistently worth retrieving. Restrict to `kind=error_fix` and `provenance in {world, mixed}`. Do not auto-admit LLM drafts (`infer`) or sim-only rows.

**8. Live LoRA attach in `StudioHost.generate`** — **Done for PEFT.**
`GenerateRequest.adapter_path` on a PEFT dir becomes `payload.use_adapter`. The worker `load_adapter`s for that generate and restores. GGUF LoRA is a sibling artifact (`gguf_path`, CLI `export-gguf`); attach is load-time `--lora`, not a mid-chat llama-server restart.

**9. Scheduled compact**
`--older-than` is **shipped** on `compact` (stale proposed WHO/infer, default 30 days; world/mixed `error_fix` kept). Still explicit-only, dry-run default, not inside `episode.run`. A Studio cron / weekly job remains optional.

**10. Directive TTL**
MemoryWheels §7.2: directives may carry “scope and TTL optional.” Never implemented. Additive column (or a body convention plus a compact/retrieve filter). Exclude expired directives from default retrieve; do not hard-delete.

**11. Stakes heuristic from user text**
MemPhase2 retrieve: no heuristic on “prod” / “deploy” — “that is a later throne concern.” Closed word list → `stakes=high`, same as the payload flag. False positives are the risk; keep it opt-in (`EpisodeRequest` or a throne constant) rather than silent.

**12. Probe fail → force-propose the probed procedure**
MemPhase3 open question 7: “no, not in v1.” The hook is already there (`review_write` / `force_proposed_reason`). Still a one-function change. Do it only if operators are acting on failing `Probe:` logs by hand.

### 3. A focused PR or two — still product, more surface

**13. Wire Unsloth DPO** — **Done.**
`--recipe preference` on `UnslothTrainBackend` calls TRL `DPOTrainer` (imported after Unsloth). Fake path still writes `pairs.jsonl`. SFT/distill do not import DPO. GPU-box only; CI stays on the fake backend.

**14. Contradiction / supersession UX**
**Partial.** Gate covers claims, procedures, and WHO-vs-WHAT dissonance. CLI `contradictions` lists both. Operator `admit` of a colliding WHO row needs `--force`. Remaining fruit: print a warning from `compact --dry-run` / `admissions`, then a thin inspect view (titles, both bodies, `admit` / `reject` / `supersede` actions).

**15. Numeric twin-drift / auto-distrust**
MemPhase2 A7: write the note only; “no calibration loop, no numeric drift estimate, no auto-distrust of prior sim claims.” Next step that is still small: if an episode writes a `twin_note`, force-propose same-episode sim `procedure` / `claim` rows (or exclude them from sim-biased retrieve). Auto-calibration science stays Phase 6.

**16. Frontend memory browser**
**Shipped as first Studio face.** Settings tab + `/unforgettable` dashboard wrap Apache store + `operators.py` (queue, admit/reject, review/mine, compact, compile, adapter registry). Live LoRA attach remains item 8.

**17. Headless / TUI Host**
MemPhase1 / `TECHNICAL.md` §2.2: “A future TUI implements `Host` and is done.” The protocol is already the contract (`generate`, `complete`, `run_action`, `confirm`). No Studio import required. Good relocatable proof.

**18. `/v1/messages` and `/v1/responses`**
Parked since MemPhase1. Public face is chat completions only because those routes do not run the Studio tool loop. Real work is teaching those routes to call `episode.run` with the same stream contract, or documenting that they will stay out. Do not half-wrap them.

### 4. Medium — only with a measured need

**19. Paraphrase dedupe**
MemPhase2 compact chose exact `normalize_title()`. “Paraphrase dedupe waits for evidence.” Do not add embeddings “because we can.” If FTS + title-dedupe is visibly missing near-dupes in a real `memory.db`, then either a bounded `Host.complete` cluster pass or item 20.

**20. sqlite-vec**
Reserved comment on `record_fts` in `store/schema.py`. Every phase said “revisit only if a measured FTS miss shows up.” Additive `records_vec` table; keep FTS working on machines without sqlite-vec. Tests must not require the extension.

**21. Incremental clone**
MemPhase3 accepted full `copytree` of the project tree. Twin plugin `fs.copy` is the place for skip-unchanged / hardlink later. Same-path / dest-inside-source / marker-skip rules stay. Do not invent a second sandbox API.

**22. Retrieve extras from Phase 1 “not in Phase 1”**
Temporal decay curves, MMR rerank. Staleness is already a one-line age note at ≥30 days. Only worth it if `inject_stats` + operator `load` show the wrong records winning.

**23. Compile more kinds; shrink standing further once C is live**
MemPhase4 refused claims, error_fix, twin notes, episodes, `Probe:`, and `test command`. MemPhase4 open question 4: shrink `COMPILE_BODY_CHARS` once C is proven. Standing already drops pack sources when `adapter_id` is set. Further shrink wants a live attach (item 8) and a holdout that still passes.

**24. Studio training UI / traffic-splitting canary**
MemPhase5 out. Recipe/job from Apache is the product. A Studio wrapper around `pack`/`train`/`eval`/`promote` is optional sugar. Canary stays holdout eval, not a production traffic split, unless someone is actually serving two adapters.

### 5. Large / domain-hard

**25. Extra isolation (containers)**
Parked by Phase 3. World and sim already isolate by `session_id` + Studio’s existing sandbox. Containers are a Host implementation detail, not a new Apache API.

**26. Other-domain twins**
MemoryWheels §15: “Minimal viable twin per domain” and “Non-physics domains: what counts as a matched playground?” Today the coding-domain sim is a filesystem clone + test harness. Games, API mocks, robotics twins are new Hosts + eyes, not more SQLite.

**27. Physics / high-fidelity twins**
MemoryWheels: architecture ≠ free fidelity. Same contact interface (`run_action` / generate in a session), different world.

**28. LLM `/dream` rewrite of B, LLM rewrite of probes, skill.md files on disk**
Rejected as compact/compile designs. A later research-flavored pass can cluster and propose supersedes; it must still go through `admit()` and must not clobber bodies in place.

**29. Provenance graph; “was this worth keeping?”**
MemoryWheels §15. We have provenance tags, supersede chains, admissions log, retrieve_uses, rollouts. A graph (record → episode → rollout → pack) is inspectability, not a new kind. Calibration of keep/drop is still the operator + probes.

**30. Auto-train at episode end; `memory_pack` / `memory_train` tools**
MemPhase5 non-goals and MemoryWheels §14. C is never an unsupervised firehose. Leave it off the product path.

---

## Phase 6 — research side car (do not schedule)

From [`plans/MemoryPhases.md`](plans/MemoryPhases.md) and MemoryWheels Tier C (C14–C18). `TECHNICAL.md` §1.6 repeats the same list. Keep it so it is not forgotten; do **not** open a `MemPhase6.md` until Phases 1–5 leftovers above are boring.

| # | Item | Notes |
|---|------|--------|
| C14 | Stronger teacher/student distillation (online or near-online) | Phase 5 distill is offline title→body SFT. A live teacher that still holds B in context is the research step. |
| C15 | Continual learning on serving weights | Still discouraged. Adapters + rollback are the product. |
| C16 | Attention / activation alignment across unequal prefixes | Naive Frobenius match is ill-posed (MemoryWheels §8.3). Needs a real alignment recipe. |
| C17 | Productized parametric unlearn | Do not depend on it. B supersede + adapter rollback are the forget path. |
| C18 | Automated high-fidelity twin calibration | Twin notes stay write-the-disagreement. Calibration loops are science. |

Original CAS dual-model zero-context story lives here. Parked CAS math (MemoryWheels §15): equalizing representations when prefix lengths differ; train-spend vs serve-savings cost model; forgetting mitigations **if** base is ever touched (it should not be).

---

## Still not doing

From MemoryPhases “What we are still not doing” and the repeated phase non-goals. These are boundaries, not backlog.

- Treating Studio RAG (`rag.db`) or chat history as B
- Forking or vendoring Grok Build
- Fine-tuning on raw session logs, episode bodies, `infer` / `proposed`, or sim-only glory
- Promoting sim-only dynamics to world truth
- Building CAS before extract, gates, and a boring B (that precondition is now met; CAS is still research)
- Reopening `admit()` order
- Required Host-protocol churn for the sake of it
- Always-on confirm on every world retry (available; default off)

---

## How to pick the next slice

1. **If C should actually change Studio chat:** item 8 (live LoRA attach). Everything else in sidecar is already operable from the CLI.
2. **If B is getting messy:** item 9 (scheduled compact), then 14 (contradiction UX).
3. **If proposed `error_fix` rows are consistently good:** item 7, with evidence in the admissions log.
4. **If you want a face that is not Studio:** item 17 (TUI / headless `Host`).
5. **Do not start Phase 6** to avoid the list above.

---

## Sources

| Doc | Role |
|-----|------|
| [`TECHNICAL.md`](TECHNICAL.md) | What the tree implements; §1.6 parked Phase 6 list |
| [`plans/MemoryWheels.md`](plans/MemoryWheels.md) | Architecture; §12 tiers; §15 open questions |
| [`plans/MemoryPhases.md`](plans/MemoryPhases.md) | Original phase cadence (stale as status) |
| [`plans/MemPhase1.md`](plans/MemPhase1.md) | Bones; out-list: vec, UI, TUI Host, `/v1/messages` |
| [`plans/MemPhase2.md`](plans/MemPhase2.md) | Compact / extract / stream; parked auto-compact, auto-admit, paraphrase |
| [`plans/MemPhase3.md`](plans/MemPhase3.md) | Rims; parked containers, auto-calibration, probe auto-deprecate |
| [`plans/MemPhase4.md`](plans/MemPhase4.md) | Compile + trajectories; parked `skip_standing` payload (now shipped) |
| [`plans/MemPhase5.md`](plans/MemPhase5.md) | Sidecar C; parked live attach, DPO, training UI, `adapters gc` |
