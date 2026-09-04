# Remaining Memory Wheels phases

High-level roadmap for work after Phase 1. Companion to `MemoryWheels.md` (architecture) and `MemPhase1.md` (bones already shipped).

---

## Already done — Phase 1 (bones)

Shipped. See `MemPhase1.md` for the implementation note. In one line: Apache `unforgettable/` package with structured B store, memory tools, crude world/sim clone + act/sim policy, internal agents with trivial bodies, and a thin Studio virtual-model face. Remaining phases fill those stubs and then the MemoryWheels tiers that Phase 1 parked.

---

## How this list is ordered

`MemoryWheels.md` §12 is the priority spine:

- **Tier A first** (main car B, sim rim, act path, gates).
- **Tier B only after A is boringly reliable** (parametric sidecar C).
- **Tier C stays research / optional** (CAS-shaped work; do not block the chariot).

**Easy-win overrides:** a later Tier A item that is small and sits on existing bones is pulled into an earlier phase (twin-drift notes, compaction, episode summaries, a CLI, retrieve budget). Streaming the inner loop is pulled forward because the current one-chunk generate is a product hole, not new architecture.

License rule unchanged: new brains in `unforgettable/` (Apache 2). Studio only grows if the public face needs it (stream, catalog, optional inspect UI).

---

## Gap map (MemoryWheels §12 vs today)

| # | Item | After Phase 1 | Goes into |
|---|------|----------------|-----------|
| A1 | Structured B + memory tools | Working skeleton | Phase 2 (hygiene, retrieve policy) |
| A2 | Error→fix + provenance | Types and naive extract exist | Phase 2 (richer extract) |
| A3 | End-of-task / post-failure extract + admission | Rule-based fail→success only; LLM extract is empty | Phase 2 |
| A4 | Gate eyes | Protocol + log-only | Phase 2 (v0) then 3 (regressions) |
| A5 | Sim rim v0 / shared action interface | Filesystem clone of the sandbox | Phase 3 |
| A6 | Act → sim-on-fail → retry world | Policy + FakeHost path; Studio inner pass is non-streaming | Phase 3 (behavior) + Phase 2 (stream, easy win) |
| A7 | Twin-drift logging | Missing | **Phase 2 (easy win)** |
| A8 | Compile repeated procedures into prompts | Missing | Phase 4 |
| A9 | Trajectory library | Missing | Phase 4 (table in Phase 2 if cheap) |
| B10–13 | PEFT packs, adapters, distill, preference | `sidecar.py` stub | Phase 5 |
| C14–18 | Online distill, live weights, attention match, unlearning, auto-calibration | Not started | Phase 6 (parked) |

---

## Phase 2 — Main car that actually learns

**Goal:** B becomes the notebook the stack trusts: extract, compact, drift, inspect. Still no training.

Easy wins first inside this phase, then the extract/gate work A3–A4 need.

1. **Live stream of the inner pass (easy, Studio-facing).** Forward inner tokens instead of one buffered chunk. Does not change wheels; makes the virtual model usable.
2. **Twin-drift notes (A7, easy).** If sim succeeded and world retry still failed, write a `twin_note` (world vs sim disagreement). Manual is fine; no calibration loop.
3. **Inspect CLI (easy).** `python -m unforgettable …` search / get / list / admissions log. Avoids a frontend.
4. **Compaction pass (Grok `/dream` analog, easy-medium).** Dedupe near-identical titles, fold superseded chains, drop empty proposed. Scheduled or explicit tool. Not an LLM rewrite of the whole store.
5. **Retrieve policy.** Token/char budget; prefer `world`/`mixed`/`human`; optional stakes flag. Staleness note already exists.
6. **End-of-task extract (A3).** Keep the naive fail→success writer. Add a bounded LLM extract that proposes typed records from traces; everything still goes through admissions (`infer` stays proposed).
7. **Gate eyes v0 (A4).** Contradiction scan (same-title conflicting claims), refuse/propose sim-only dynamics claims (already partly policy), queryable admissions log. No domain regression suite yet.
8. **Episode summaries + thin trajectory rows.** One `episode` record per run; optional graded rollout row (world|sim, pass/fail, pointer to traces). Enough for later A9 without building a library product.

**Out:** PEFT, frontend memory browser, vec index (unless FTS clearly hurts).

**Done when:** a real Studio `unforgettable` episode can remember, correct, deprecate, auto-note drift, compact, and show up in the CLI; extract proposes more than the one hard-coded error→fix.

---

## Phase 3 — Rims and motion

**Goal:** world and sim feel like two contact surfaces with the same tools, not “a second folder and a regex.”

1. **Sim as a test harness.** In the coding domain: after clone, sim eyes can run the project’s tests / a nominated command and grade pass/fail. Shared action interface = same `python`/`terminal` tools, different `session_id`.
2. **Richer recognized failure.** Beyond traceback / non-zero / `Error:` — test failures, explicit `enter_sim`, optional user “that failed.” Configurable thresholds and sim budget (already constants).
3. **World retry policy.** Throne may require confirm before retry-act (reuse Studio tool approval). Escalate instead of looping.
4. **Rim hygiene.** Never share one sandbox line between world and twin; teardown rules (keep sim on admitted error_fix / drift; delete on clean success) stay explicit.
5. **Gate eyes v1.** A tiny regression pack for the domain (e.g. “old procedures still retrieve and still pass in sim”). Coverage is the hard part; start with a handful of probes stored as B procedures.

**Out:** physics twins, extra isolation (containers), auto-calibration science.

**Done when:** fail in world → tests in clone → retry world is the default coding path, with drift notes and a human gate available on retry.

---

## Phase 4 — Compile B into the inner wheel

**Goal:** stable lessons cost less prompt. Still not weights.

1. **Procedure compilation (A8).** Repeated admitted procedures become a standing prompt block or generated skill-like markdown the episode injects by default. Source of truth remains B; compiled form is a cache.
2. **Trajectory library (A9).** Retrieve graded world/sim rollouts when acting or rehearsing; bias sim lessons in sim mode.
3. **Measure durable-redundant load.** Simple metric: tokens of standing B inject vs task quality. Success is shrink-without-drop, as MemoryWheels states.

**Out:** adapters.

**Done when:** a boring procedure is no longer re-explained every turn, and you can point at the B record that compiled form came from.

---

## Phase 5 — Side car C (MemoryWheels Tier B)

**Goal:** Unsloth earns its keep. Only after Phases 2–4 are dull.

1. Pack construction from **admitted** B + graded **world** traces (trusted sim optional, never sim-only glory).
2. Batched PEFT / QLoRA via existing Unsloth training APIs; base stays frozen.
3. Shadow adapter → eval (gate eyes + “works with less retrieval?”) → promote or discard. Rollback = drop adapter.
4. Offline prompt/context distillation for stable curricula if packs stay retrieval-heavy.
5. Outcome-conditioned preference on graded traces if SFT is not enough.

`sidecar.py` is the home. Studio training UI is optional; a recipe/job from the Apache package is enough.

**Done when:** one domain adapter can be trained from B, held out, and rolled back without touching base weights.

---

## Phase 6 — Research side car (MemoryWheels Tier C, parked)

Do not schedule as product work. Keep the list so it is not forgotten:

- Stronger teacher/student distillation (online or near-online)
- Continual learning on serving weights (still discouraged)
- Attention/activation alignment across unequal prefixes
- Productized parametric unlearn (do not depend on it)
- Automated high-fidelity twin calibration

Original CAS dual-model zero-context story lives here.

---

## What we are still not doing

- Treating Studio RAG or chat history as B
- Forking or vendoring Grok Build
- Fine-tuning on raw session logs
- Promoting sim-only dynamics to world truth
- Building CAS before extract, gates, and a boring B

Open questions in MemoryWheels §15 stay open until the phase that needs an answer (admit autonomy in Phase 2, failure thresholds in Phase 3, pack leakage in Phase 5).

---

## Suggested cadence

| Phase | MemoryWheels | Character |
|-------|----------------|-----------|
| 1 | A1–A2, crude A5–A6 | Done — bones |
| 2 | A3, A4 v0, A7, stream, CLI, compact | Next — B becomes real |
| 3 | A5–A6 for real, A4 v1 | Motion you can trust |
| 4 | A8–A9 | Leaner inner wheel |
| 5 | B10–B13 | C, gated |
| 6 | C14–C18 | Research only |
