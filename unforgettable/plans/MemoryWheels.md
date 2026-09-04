# Memory wheels: progressive consolidation under an Ezekiel-shaped stack

Working architecture sketch. Companion to `EzekielAGI.md`.  
Earlier CAS-only draft lived in `MiddleWheel.md` (research-model paste; historical).  
This note is the synthesis: **what to build, in what order, on which substrate.**

Not a Moqui design doc. Not vetted implementation. Organizing frame for further review and research.

---

## 1. Purpose

`EzekielAGI.md` names the big unlike:

> Experience → consolidate → durable knowing  
> vs frozen weights + side databases + human ops

This document sketches **wheels that turn on memory**: nested loops under standing watch, so a single face of “the AI” can **grow**—remember, correct, and (selectively) internalize—without treating every lesson as a full retrain or a disposable chat turn.

**Center of gravity:** progressive memory.  
**Main car (near-term and default):** external structured memory + tools + eval gates.  
**Side car (later, gated):** parametric assimilation (adapters / context distillation / CAS-like ideas)—only after the main car works.  
**Dual contact:** world-facing eyes and an **inner simulation rim** (domain twin / playground)—wheel within wheel, both eyed—not a vague “mind’s eye.”

**Not claimed:** solved AGI, clean parametric forget, perfect twin fidelity, or continuous live weight updates as the default hippocampus.

---

## 2. Problem (plainly)

A live agent accumulates **working context**: instructions, tool traces, env state, intermediate results. That state lives in the prompt / KV cache. It is powerful, costly, and mostly **volatile**.

| Escape hatch | What it does | What it doesn’t |
|--------------|--------------|-----------------|
| Longer context | Carry more episode state | Consolidate; cost grows with history |
| Naive RAG dump | Park text beside the model | Structure, supersession, intentional remember/correct |
| Occasional LoRA / FT | Some durable skill shift | Continuous, selective, reality-gated growth as native law |
| “Just fine-tune on the logs” | Absorb noise + bias | Safe progressive memory |
| Chain-of-thought only | Cheap verbal rehearsal | Runnable domain dynamics; graded fail-before-world |

**Wanted:**

1. Durable patterns (facts-with-provenance, procedures, **errors and their corrections**) move from “gone when the session ends” toward “available next time, and increasingly default behavior”—without stuffing episodic scratch into weights.  
2. **Hypothesize → contact → update** as law of the machine: expensive contact with the **world**, cheap high-frequency contact with a **matched simulation** when the world has already said no (or throne requires rehearsal).

That is the **spirit-in-the-wheels** bet: motion and knowing stay coupled. Correction is first-class; erase-from-weights is not the design center.

---

## 3. Three substrates (keep these distinct)

Progressive memory is not one bucket. Mixing them is how designs fail.

| Substrate | Holds | Write | Correct / “forget” | Good for |
|-----------|--------|--------|---------------------|----------|
| **A. Working** | Episode state, live/sim sensors, current plan | Every step (KV, scratchpad, tool state) | Drop when episode ends; compact mid-task | What is true *now* in this run (world or sim branch) |
| **B. External structured (main car)** | Claims, notes, procedures, error→fix cases, entity state | Explicit tool, end-of-task extract, admission gate | **Supersede, deprecate, archive**—reliable | Ideas, citations, playbooks, lessons |
| **C. Parametric (side car)** | Smooth skills, domain priors, “how we usually do X” | Batched PEFT / distillation after eval | Weak unlearning; prefer **rollback adapter**, counter-train, or leave base frozen | Stable behavioral competence, not atomic notepad facts |

**Not substrates:** the **world** and the **sim** are *contact surfaces* the wheels roll on—not memory buckets. Traces from either may propose writes to A/B/C under gates.

**Rule of thumb:** if you must edit it tomorrow with a sentence, it belongs in **B**. If it should change how the model acts even when retrieval misses, consider **C** later. If it is only true for this run, keep it in **A**.

---

## 4. Two axes of structure

Ezekiel’s image is easier if we separate **timescale** from **contact**. Do not overload “outer” to mean both “memory loop” and “world-facing.”

### 4.1 Timescale axis (vertical stack)

Wheels = nested loops. Not “middle wheel = distillation GPU.”

```
+-----------------------------------------------------------------------+
|    THRONE — objectives, values, admission, act/sim policy, promotion   |
+-----------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------+
|  OUTER WHEEL — progressive memory (promote into B and, rarely, C)     |
|    main car: structured stores     side car: adapters / distill (late)  |
+-----------------------------------------------------------------------+
                                    ^
                                    | candidates / evals / rollbacks
+-----------------------------------------------------------------------+
|  MIDDLE WHEEL — task/episode (curate, traces, remember, act↔sim mode) |
+-----------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------+
|  INNER WHEEL — step inference + action (generation, tools, KV/work)   |
+-----------------------------------------------------------------------+
```

| Vision | Duty | In this design |
|--------|------|----------------|
| **Throne** | What the stack serves; **what may stick**; **how to contact** | Goals; B/C admission; default act/sim policy; freeze base by default |
| **Outer wheel** | Long loop: consolidate and promote | Admit into **B** routinely; pack and train **C** only when justified |
| **Middle wheel** | Episode/task loop | Context curation; traces; memory ops; switch world-act vs sim-rehearse |
| **Inner wheel** | Step loop | Model + tools + working memory; issue actions into **world or sim** |
| **Spirit in the wheels** | Coupling | Lessons stay with motion (B/C); inner motion can spin in sim without turning the whole chassis in the world |

### 4.2 Contact axis (coaxial pair — wheel within wheel, both eyed)

```
                    INNER / MIDDLE (act or rehearse)
                              |
              +---------------+---------------+
              |                               |
              v                               v
   +------------------+             +------------------+
   |  WORLD RIM       |             |  SIM RIM         |
   |  (outer contact) |             |  (inner contact) |
   |  real domain     |             |  matched twin /  |
   |  sensors+acts    |             |  playground      |
   +--------+---------+             +--------+---------+
            |  world eyes                    |  sim eyes
            |  outcomes, cost,               |  cheap grades,
            |  irreversible-ish              |  counterfactuals
            +---------------+----------------+
                            v
                   traces → middle → outer memory
                   (tag provenance: world | sim | mixed)
```

| Rim | Sees / contacts | Role |
|-----|-----------------|------|
| **World rim** | Production domain: physical plant, live game server, prod APIs, wet lab, etc. | Costly truth; final answerability |
| **Sim rim** | Domain twin with **matched dynamics** (not vague imagery) | Cheap hypothesize, fail, grade, rehearse |

**Example:** if the world is a 3D video game with physics, the sim is a 3D playground with the **same** (or intentionally approximated) physics—same control interface where possible—so the inner wheel can practice without spending world consequences every step.

**Not “mind’s eye.”** A runnable environment (engine, digital twin, test harness, sandbox with real semantics). Verbal CoT can assist planning; it does not replace the sim rim.

**Fidelity is domain-hard.** Architecture assumes a twin *exists at some fidelity*; calibration and drift watch are eye duties (see §9).

Optional instruments (dual teacher/student, etc.) hang off the outer **side car**. They do not redefine the inner wheel or replace the sim rim.

---

## 5. End-state shape (one face to the user)

```
                    User / goals  (throne inputs)
                            │
                            v
                 ┌──────────────────────┐
                 │  Policy + tools      │  inner/middle wheels
                 │  (base model, slow)  │  frozen or rarely touched
                 └──────────┬───────────┘
                            │
           act / sim mode (throne policy)
                    ┌───────┴───────┐
                    v               v
              WORLD RIM        SIM RIM
              (eyes)           (eyes)
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              v             v             v
         Working (A)   Durable stores (B)   Adapters (C)
         episode/KV    MAIN CAR             SIDE CAR
                       claims, procedures,  PEFT packs when
                       error→fix, state     eyes + throne allow
                       + sim|world tags
              │             │             │
              └─────────────┼─────────────┘
                            v
                    Admission + eval (gate eyes)
                    promote / supersede / reject
```

From outside: one AI that can be told “remember this,” “that was wrong; use Y when Z,” and that can also **propose** lessons after tasks—and that **rehearses in a twin** after recognized failure before burning the world again. Inside: almost all durable growth hits **B** first; **C** is an optimization pass on stable patterns, not the notebook.

---

## 6. Default act path (throne policy)

Throne owns contact policy. Domains may override (e.g. always sim-first for high-stakes actuators). **Default general policy:**

```
[1] ACT FREELY in the WORLD
        (use A + retrieved B [+ C]; world eyes watch)
        |
        v
[2] Success? ──yes──> continue / complete task
        |
        no (recognized failure: eyes or self-detect)
        |
        v
[3] SIMULATE until success
        (same goals/constraints; sim rim + sim eyes;
         search, rehearse, counterfactual; update A)
        |
        v
[4] RETRY ACT in the WORLD with the repaired plan
        |
        v
[5] Still fail? ──> more sim, escalate, or admit error→fix and stop
        yes success ──> continue; extract lessons (tag world / sim / mixed)
```

| Phrase | Meaning |
|--------|---------|
| **Act freely** | Prefer direct world contact; do not mandate rehearsal every step |
| **Recognized failure** | World eyes (fail grade, constraint break, anomaly) and/or internal detect (tool error, contradiction with B) |
| **Simulate until success** | Stay on sim rim until a candidate plan passes **sim eyes** (or budget/escalation hits) |
| **Retry act** | Commit to world again; world remains the judge of record |

**Why this default:** maximizes progress when the world is cheap enough to touch; uses the inner rim when the world has already vetoed a line of attack—Ezekiel-ish “spin the inner wheel” without turning the whole chassis every time.

**Throne may tighten:** sim-first for irreversible actions; human approve before world retry; max sim budget; ban sim-only promotion into B/C for safety-critical claims.

---

## 7. Main car — external structured memory

### 7.1 Why main

- **Correctable** (supersession beats unlearning).  
- **Inspectable** (provenance, “why do you believe this?”).  
- **Selectable** (retrieve what the task needs; don’t FT the universe).  
- **Realistic now** (agent memory products, claim stores, notebooks + retrieval).  
- Matches `EzekielAGI.md`: memory-first, selective parametric consolidation.

RAG-as-blob is not enough. Structure matters.

### 7.2 Record kinds (illustrative)

| Kind | Role | Example |
|------|------|---------|
| **Claim** | Vetted or provisional fact | “Pump X max rate is … (source, confidence)” |
| **Procedure** | How to do a recurring job | Playbook steps that passed eyes (note if sim-only vs world-baked) |
| **Error→fix** | Lesson from failure | Tried A → measured B → use C when D; keeps the mistake as teacher |
| **Entity / state** | Long-lived world-model slice | Device config, project decisions, open issues |
| **Episode summary** | Compressed history | What happened last run (not full token log) |
| **Directive** | User/self instruction to remember | Explicit `memory.write` with scope and TTL optional |
| **Twin note** | Sim fidelity / drift | “Sim friction understates world by ~10% on surface X” |

**Error→fix is first-class.** Prefer remembering the wrong turn *and* the correction over deleting the wrong turn—including **sim failures** (cheap curriculum) and **world failures** (high trust). That helps related future cases and avoids the parametric-forget fantasy.

### 7.3 Provenance (required field)

Every durable record should carry contact provenance:

| Tag | Meaning | Default trust |
|-----|---------|----------------|
| `world` | Confirmed under world eyes | High |
| `sim` | Only under sim eyes | Lower; fine for rehearsal tips |
| `mixed` | Sim-found, world-confirmed (or vice versa) | High if world leg present |
| `human` | User/operator asserted | Per throne |
| `infer` | Model-proposed, ungated env | Lowest until eyes touch |

**Do not promote `sim`-only dynamics claims to equal `world` truth** without calibration or a world leg. Procedures may start as `sim` and upgrade to `mixed`/`world` after retry-act succeeds.

### 7.4 Write paths

1. **Explicit:** user or agent tool — “remember …”, “correct …”, “deprecate …”.  
2. **End-of-task extract:** middle wheel proposes records from world and sim traces + outcomes; eyes/throne admit.  
3. **Failure branch extract:** on recognized failure and after sim recovery, write error→fix (what failed in world, what worked in sim, what retried).  
4. **Scheduled compaction:** merge duplicates, supersede stale claims, archive dead procedures; refresh twin notes from drift watches.

Nothing durable lands without **admission** (even if admission is “auto-allow low-risk notes in a sandbox namespace”).

### 7.5 Read paths

- Retrieve by task / entity / similarity; **rerank**; ground answers in returned IDs.  
- Prefer procedures and error→fix when acting; prefer `world`/`mixed` when stakes are high.  
- On sim-rehearse branch, retrieval may bias toward prior sim lessons and twin notes.  
- Do not dump the whole store into the prompt.

### 7.6 Correct, supersede, “forget”

| Intent | Mechanism on B |
|--------|----------------|
| Fix a fact | New claim supersedes old; old marked superseded (history kept) |
| Change a procedure | Version bump; prior version archived |
| “Forget that” | Deprecate/archive + exclude from default retrieval; hard delete rare |
| Learn from a bug | Add error→fix; optionally link superseded claim/procedure |
| Twin wrong | Twin note + supersede sim-derived claims that depended on bad dynamics |

No requirement to edit base weights for these operations.

---

## 8. Side car — parametric assimilation (CAS-shaped, deferred)

Historical CAS idea (`MiddleWheel.md`): distill context-rich behavior into weights toward a “zero-context” skill state, often via teacher-with-\(C\) vs student-without-\(C\).

**Status here:** optional **outer side car**, not the main progressive-memory path. Use when **B** is working and some stable skill is still too slow/heavy to retrieve every time.

### 8.1 What it is good for

- Repeated procedures that should become fluent.  
- Domain style / priors that retrieval only approximates.  
- Shrinking **durable-redundant** prompt load.  
- Later: packs that include **graded sim rollouts** plus world-confirmed successes.

### 8.2 What it is bad for

- Atomic facts, citations, user secrets, one-off episode state.  
- Anything that must be corrected with a single sentence tomorrow.  
- Unsupervised firehose of traces (self-bias amplification).  
- **Training only on uncalibrated sim** (false competence).

### 8.3 How it should run (when eventually built)

```
[curated pack from B + graded world traces + optional calibrated sim rollouts]
        →  batched PEFT / context-distill on adapter (or bounded slice)
        →  gate eyes: task metrics + regressions + “works with less retrieval?”
        →  prefer world-validated items; sim-heavy packs need twin-trust checks
        →  throne: promote adapter / keep shadow / discard
        →  base model stays frozen unless a rare, deliberate merge
```

**Prefer async batches** over continuous online backprop while serving.  
**Prefer adapters** over full FT.  
**Prefer outcome-aware selection** over pure logit imitation of a teacher that still holds the answer in context.

Dual-size teacher/student and attention-map matching are **research options**, not requirements. Naive attention Frobenius match across unequal prefix lengths is ill-posed; park that until a real alignment recipe exists.

### 8.4 Relation to “zero-context”

**Aspiration:** large classes of durable skill need less and less re-prompting and re-retrieval.

**Grounding:**

| Context kind | Destination |
|--------------|-------------|
| Episode / live or sim sensors | **A** only |
| Facts, lessons, playbooks | **B** first (with provenance) |
| Stable fluent skill | **C** candidate after B proves the pattern |

Success metric:

> Shrink **durable-redundant** prompt and retrieval load while holding task quality and base competence.

Not: literal empty context for all state forever.

---

## 9. Eyes and throne (three watches)

Without these, outer loops are self-training cosplay. **Rims full of eyes** means more than one watch.

### 9.1 World eyes

- Production telemetry, constraint violations, task graders in the real domain.  
- Cost and irreversibility signals (what must not be retried casually).  
- Final judge on retry-act after sim.

### 9.2 Sim eyes

- Same *kinds* of grades inside the twin (success, bounds, physics breaks).  
- Rollout statistics, counterfactual comparisons.  
- “Plan looks good in here” signal that unblocks world retry—not a substitute for world eyes.

### 9.3 Gate eyes (memory / identity hygiene)

- Regression probes: general skills and old procedures still work.  
- Contradiction and stale-claim detection on **B**.  
- Twin-drift watch: world vs sim disagreement → twin notes, distrust sim-only packs.  
- Side car: competence with reduced retrieval; no promote on sim-only false glory.

### 9.4 Throne

- Objectives and hard constraints.  
- **Act/sim policy** (default §6; domain overrides; budgets; human-in-loop triggers).  
- **Admission** into B (namespaces; min provenance for type).  
- **Promotion** into C; never auto-touch full base by default.  
- Prefer B when unsure; rollback authority.

---

## 10. Remember, correct, autonomy (single-AI behavior)

| Capability | Supported? | How |
|------------|------------|-----|
| User: “remember X” | Yes | Tool → B (admission); C only if later pack includes it |
| User: “that was wrong; use Y” | Yes | Supersede + error→fix in B; optional later C refresh |
| User: “forget X” | Partial | Deprecate in B; adapters rolled back if they encoded X; base best-effort only |
| Self: propose what to keep | Yes | Middle extract + eyes + throne after tasks / recovery |
| Self: enter sim after failure | Yes | Default policy §6 |
| Self: silent weight drift | **No (by policy)** | No admit → no promote; C never unsupervised firehose |

**Autonomy** means gated self-proposal and policy-driven act/sim switching—not unsupervised self-rewriting. Progressive **correction** is the design center; parametric forget is not.

---

## 11. Runtime flow

### 11.1 Step / episode with act↔sim (inner + middle)

```
[1] Throne mode: default WORLD-ACT (unless domain says otherwise)
        |
        v
[2] Inner: act in WORLD with A + retrieved B [+ C]
        |
        v
[3] World eyes: outcome
        |
        +── success ──> [6]
        |
        +── recognized failure ──>
                [4] Middle: enter SIM mode; encode failure into A
                        |
                        v
                [5] Inner: simulate until sim eyes pass (or budget/escalate)
                        |  (search plans, use B error→fix, twin notes)
                        v
                [5b] Retry WORLD act with repaired plan
                        |
                        +── success ──> [6]
                        +── fail ──> more sim / escalate / stop + error→fix
        |
        v
[6] Middle: update A; buffer traces (tag world|sim); optional memory tools
        |
        v
[7] Propose B writes (claims, procedures, error→fix, twin notes)
        |
        v
[8] Gate eyes + throne: admit / edit / reject → B updated
```

### 11.2 Outer promote into B (continuous, lightweight)

```
World + sim traces + extracts
        → dedupe / supersede / compact / upgrade provenance
        → B stays coherent over weeks
```

### 11.3 Outer side car into C (rare, heavy, batched)

```
Stable patterns in B + graded world successes
        [+ calibrated sim rollouts if twin trusted]
        → pack dataset
        → train adapter
        → gate eyes + regressions
        → promote or discard
        → middle may drop redundant standing prompt once C proven
```

Cadence: B can be every task; C might be weekly/monthly or “when pack is fat enough.”

---

## 12. Build order (realism tiers)

Do not start with live CAS. **Do** start a crude domain twin/test harness early if the domain allows.

### Tier A — build first (realistic now)

1. Structured stores **B** + memory tools (write, correct, supersede, retrieve).  
2. Error→fix as a normal record type; **provenance** `world|sim|…`.  
3. End-of-task and post-failure lesson extract with admission.  
4. Gate eyes: task grades + basic regressions + provenance on writes.  
5. **Sim rim v0:** domain test harness / twin / playground at whatever fidelity is available; shared action interface where possible.  
6. **Act path §6:** world-act → on failure sim-until-success → retry world.  
7. Twin-drift logging (even manual) when world and sim disagree.  
8. Prompt/program compilation from repeated B procedures (still not weights).  
9. Trajectory library: graded rollouts (world and sim) for retrieval.

### Tier B — selective parametric (after A is boringly reliable)

10. Curated PEFT packs from admitted B + graded **world** traces (+ trusted sim).  
11. Domain adapters; shadow deploy; promote under eval.  
12. Offline context/prompt distillation for stable curricula.  
13. Outcome-conditioned preference on traces in graded envs.

### Tier C — research side car (optional)

14. Stronger teacher/student context distillation online or near-online.  
15. Continual learning on serving weights (usually still discouraged).  
16. Fancy activation/attention alignment across unequal contexts.  
17. Productized parametric unlearning (don’t depend on it).  
18. High-fidelity twin calibration loops as first-class automated science.

**Original CAS continuous dual-model zero-context story ≈ Tier C.** Keep the aspiration; do not block the chariot on it.

---

## 13. Technical realism (summary)

| Piece | Realism | Note |
|-------|---------|------|
| Nested agent loops + async jobs | High | Systems engineering |
| Gate eyes as CI/evals/logs | High | Coverage is the hard part |
| Progressive memory on **B** | High | Best ROI; correctable |
| Act free → sim on failure → retry | High (policy) | Needs failure detection + some sim |
| Domain twin / playground | High where twin exists | Games, code tests, robotics twins |
| Matched high-fidelity physics | Domain-hard | Architecture ≠ free fidelity |
| Open social/web “full sim” | Low | Mocks, sandboxes, critics instead |
| Admission + explicit remember | Medium–high | Noisy extractors; designable |
| Batched PEFT on curated packs | Medium–high | Narrow domain + evals |
| Context distillation offline | Medium | Skills/procedures > arbitrary facts |
| Online continuous weight updates | Low–medium | Fragile, costly |
| General autonomous safe parametric growth | Low | Open research |
| Clean forget in weights | Low | B supersession + adapter rollback |
| Absolute zero context | Low | Shrink redundancy; keep working memory |

**Architecture language (wheels, dual rims, throne): sound.  
CAS-as-primary-mechanism: not the best plan.  
B-first, C-later, sim-as-inner-rim: the synthesis this doc commits to.**

---

## 14. Anti-patterns

- Fine-tuning on raw session logs without admission.  
- Treating RAG chunk soup as progressive memory.  
- Putting secrets or one-off state into **C**.  
- Calling imitation of a context-rich teacher “reality-tested” without outcome terms.  
- Pruning teacher context before the lean path is proven (target rot).  
- Expecting parametric forget to replace supersession.  
- Building dual-model AMAD before memory tools and evals exist.  
- **One sandbox line** that confuses world and twin.  
- Promoting **sim-only** dynamics claims as world truth.  
- Training **C** only on uncalibrated sim rollouts.  
- Mandating sim every step when world-act is cheap and safe (throne can choose; default doesn’t).  
- Ignoring recognized failure (no enter-sim) or infinite sim with no escalate.

---

## 15. Open questions (for later research / manual vetting)

**Main car (B)**  
- Schema: claim vs procedure vs error→fix fields, confidence, provenance graph.  
- Retrieval policy under token budgets; bias by `world` vs `sim` when stakes high.  
- Contradiction and supersession UX.  
- How much autonomy on admit by namespace.

**Sim / world**  
- Minimal viable twin per domain (game physics, unit tests, digital twin, API mock).  
- Failure recognition thresholds (what counts as enter-sim).  
- Sim budget and escalate policy.  
- Automatic twin calibration from world mismatch.  
- Non-physics domains: what counts as “matched playground”?

**Gates**  
- Minimal regression suite for a given domain.  
- When human-in-the-loop is mandatory (especially world retry after sim).  
- Calibrating “was this worth keeping?”

**Side car (C)**  
- Pack construction from B without leaking episode junk or untrusted sim.  
- Adapter lifecycle (shadow, canary, merge, roll back).  
- Training signal: SFT vs preference vs distill; how eyes enter the loss.  
- Measuring durable-redundant context reduction honestly.

**Parked CAS math**  
- Equalizing or pooling representations when prefix lengths differ.  
- Cost model: train spend vs serve savings.  
- Forgetting mitigations if base is ever touched.

---

## 16. Relation to other notes

| Doc | Role |
|-----|------|
| `EzekielAGI.md` | Metaphor + gap analysis; memory-first distance check |
| `MiddleWheel.md` | Earlier CAS-heavy research-model draft; not the build plan |
| **This file** | Synthesis: dual-rim contact + B main car + C side car + act/sim policy + tiers |

---

## 17. Closing

In Ezekiel terms: **wheel within a wheel, rims full of eyes**—a **world rim** that answers for real, a **sim rim** that lets the inner wheel turn fast after failure, and an **outer memory wheel** that keeps lessons (mostly in correctable stores, rarely in weights) so the chariot does not drag every past page as prompt.

Default motion: **act freely in the world; on recognized failure, simulate until success; retry the world; remember the error and the fix.**

Spirit-in-the-wheels here means: **lessons stay with the motion**, under throne policy and many eyes.

**Main car first. Inner sim rim early. Side car later. Correction over erase. Vet mechanisms; don’t inflate claims.**
