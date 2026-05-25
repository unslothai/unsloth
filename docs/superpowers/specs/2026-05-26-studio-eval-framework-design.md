# Studio Eval Framework (Phase 1: text-output eval) — Design

**Date:** 2026-05-26
**Status:** Approved (brainstorming) — pending implementation plan
**Branch:** json-document-score

## Problem

Studio can train and run models, but there's no way to **evaluate** a model against a dataset. We want an **Eval** section where you pick a model (trained checkpoint, Hugging Face repo, or local), pick a dataset, run the model over it, and score the outputs with a task-appropriate metric — then see an aggregate score and a per-example breakdown, with run history. This replaces the ad-hoc "Score" panel.

## Scope & phasing

Evaluation spans three independent pipelines that should not be crammed into one spec:

- **Phase 1 (this spec): framework + text-output eval.** The Eval nav, eval-job engine, dataset/column mapping, persistence, results UI, and a **pluggable metric registry** shipping text metrics: exact match, ANLS text similarity, JSON document score. Uses the existing LLM inference path.
- **Phase 2 (separate spec): audio/ASR eval.** Audio-input handling + ASR inference + WER/CER metric, plugged into the framework.
- **Phase 3 (separate spec): image-generation eval.** Diffusion inference + image-similarity metrics (CLIP/SSIM/FID), its own runner on the framework.

Phases 2–3 are registry/runner plugins on the Phase 1 framework, so the framework is built first. The pluggable design (metric `reference_kind`, injected `generate`/`examples`) is what makes them drop-in.

## Goal & non-goals

**Goal:** an evaluation workflow modeled on Studio's **training** feature — start → live progress → cancel, persisted runs, a "Recents" list, and a run-detail view — for **text-output** models scored by a per-run-selected metric.

**Non-goals (Phase 1):** audio/image metrics (later phases); batched inference (Studio generates one prompt at a time); resuming a partial eval (re-run instead); running multiple evals at once (single shared model → one at a time); BLEU/ROUGE/LLM-judge (not in the initial registry).

## Requirements (decided during brainstorming)

- **Nav:** the sidebar `Score` entry becomes **`Eval`** (`/eval`). The manual document-score panel, its `/document-score` route, and the `/api/scoring/score` route are **removed** (superseded). The `json_score` **library stays** as the JSON metric.
- **Model sources:** trained checkpoint / HF repo / local — all via the existing inference load path (`ModelConfig.from_identifier` + load).
- **Dataset → (prompt, reference):** column mapping. User picks an **input** column (becomes the user message, optionally wrapped by a system prompt and/or a template with an `{input}` placeholder) and a **reference** column (expected output). HF repo+split or local file, via the existing `/api/datasets` loaders.
- **Run size:** configurable; default the **first 100 rows**, with an "all rows" option.
- **Metric:** chosen **per run** from a pluggable registry; metric-specific config supplied at start.
- **Execution:** Approach A — an **in-process background-thread runner that reuses the existing inference path** (no double model-load), with a training-like job lifecycle (start/progress/cancel/SSE) and persistence.

## Architecture

New framework under `studio/backend/eval/` (alongside the existing `json_score/`).

```
studio/backend/eval/
├── json_score/              # existing — reused by the JSON metric
├── metrics/
│   ├── base.py              # Metric interface, MetricResult
│   ├── exact_match.py
│   ├── text_similarity.py   # ANLS, reuses json_score.comparators
│   ├── json_document.py     # reuses json_anls_score / score_from_text
│   └── registry.py          # make_scorer(name, config), list_metrics()
├── runner.py                # the eval loop (dependency-injected generate + examples)
└── jobs.py                  # EvalJobManager (training-like lifecycle)
studio/backend/routes/eval.py
studio/backend/models/eval.py
```

**Reuse (no reinvention):** model resolution/loading via the existing inference path; generation via `InferenceBackend.generate_chat_response`; dataset rows via the existing `/api/datasets` loaders.

## Metric registry

```python
@dataclass
class MetricResult:
    score: float                  # in [0, 1]
    breakdown: dict | None = None # e.g. serialized ScoreNode for JSON
    error: str | None = None      # per-example issue (e.g. unparseable JSON)

# registry.py
def make_scorer(name: str, config: dict) -> Callable[[str, Any], MetricResult]
def list_metrics() -> list[MetricInfo]  # name, label, reference_kind, config fields (+types/defaults)
```

`make_scorer` returns a closure `score(prediction_text, reference) -> MetricResult` the runner calls per example. `list_metrics()` feeds the UI a config schema for a dynamic metric-config form.

| name | `reference_kind` | scoring | config |
|---|---|---|---|
| `exact_match` | text | normalize both sides, `1.0` if equal else `0.0` | `case_insensitive` (bool), `strip` (bool, default true) |
| `text_similarity` | text | ANLS normalized-Levenshtein via `json_score.comparators.string_comparator` | `threshold` (float, default 0.5) |
| `json_document` | json | `score_from_text(reference_json, prediction_text, schema)` — extracts JSON from raw output, scores per-field, returns `ScoreNode` as `breakdown` | `schema` (JSON, optional), `default_comparator` (default `string`) |

**Reference handling:** the dataset reference column is a string. Text metrics use it as-is; `json_document` parses it as JSON (parse failure → `error`, `score 0`). `reference_kind` lets the UI label/validate the reference column and (later) gate modality-specific metrics.

## Runner

`runner.py` exposes a function that takes an injected `generate` callable and a pre-loaded `examples` list, so it is testable without a model/GPU:

```python
def run_eval(
    examples: list[tuple[str, Any]],          # (input_text, reference)
    generate: Callable[..., str],             # (messages, system_prompt, **gen) -> prediction text
    scorer: Callable[[str, Any], MetricResult],
    *, system_prompt: str, template: str | None,
    gen_params: dict, should_cancel: Callable[[], bool],
    on_result: Callable[[int, MetricResult, str, str, Any], None],  # persist + emit
) -> EvalSummary: ...
```

Per example: check `should_cancel()`; render the prompt (`template.format(input=…)` or the raw input); `prediction = generate(messages, system_prompt, **gen_params)`; `result = scorer(prediction, reference)`; `on_result(...)`. Returns the aggregate (`avg_score`, counts, status). In production the job wires the real in-process `generate_chat_response`; tests inject a stub.

## Job lifecycle & data flow (like training)

```
Eval page (config)
  └─ POST /api/eval/start ───────────────────────────────────────────────┐
       validate · resolve model · create eval_runs row (running)          │
       EvalJobManager.start() → background thread runs the runner → run_id│
                                                                          ▼
  Frontend ── GET /api/eval/progress/{run_id} (SSE) ◀── thread emits progress events
       live: done/total, running avg score, ETA, last per-example result
```

The job's background thread does the heavy wiring, then drives the pure `run_eval` loop: **load the chosen model** via the existing inference path (evicts any chat-loaded model — the "no chat mid-eval" tradeoff); **load first N rows** with the chosen columns; build the scorer; bind `generate` to the loaded backend; call `run_eval` with `on_result` persisting an `eval_results` row + emitting a progress event per example (cancellable between examples); finalize `eval_runs` with `avg_score`/`status`/`ended_at`. (`run_eval` itself stays model-agnostic — see Runner — so it's unit-testable with a stub `generate`.)

**Progress model:** `EvalProgress { run_id, status, done, total, avg_score, eta_sec, last_result: {idx, score, error?} }`.

`EvalJobManager` mirrors training: `start(config) -> run_id` (background thread), `get(run_id)`, `cancel(run_id)` (sets a stop flag, effective after the current example), SSE subscription. In-memory runtime state + DB persistence.

## Routes

Registered under `/api/eval` in `main.py` (+ `routes/__init__.py`), auth via `get_current_subject` like other routes:

- `POST /api/eval/start` → `{run_id}` (409 if an eval is already running)
- `GET /api/eval/progress/{run_id}` → SSE stream (mirrors `/api/train/progress`, Last-Event-ID reconnect)
- `POST /api/eval/cancel/{run_id}`
- `GET /api/eval/runs` → paginated history
- `GET /api/eval/runs/{run_id}` → run detail + paginated `eval_results`
- `GET /api/eval/metrics` → registry (names, labels, `reference_kind`, config schema) for the UI

## Persistence (`storage/studio_db.py`)

Two tables mirroring `training_runs`/`training_metrics`, with the same module-level helper pattern (`create_eval_run`, `update_eval_progress`, `finish_eval_run`, `insert_eval_result`, `list_eval_runs`, `get_eval_run`, `get_eval_results`):

- `eval_runs` (id, model_identifier, dataset_ref, metric_name, config_json, status, started_at, ended_at, num_examples, avg_score, error_message, display_name)
- `eval_results` (run_id FK ON DELETE CASCADE, idx, input_text, prediction_text, reference_text, score, breakdown_json, error, UNIQUE(run_id, idx))

On startup, reconcile any `eval_runs` left `running` → `interrupted` (mirrors training's stale-run handling).

## Frontend (`src/features/eval/`)

React + TanStack Router + shadcn + a zustand runtime store, like `chat`/`training`.

- **Sidebar:** `Score` `NavItem` → **`Eval`** (`/eval`). Remove the `document-score` route/feature; **relocate** its `BreakdownTree` component into the eval feature (reused for the JSON metric breakdown).
- **`eval-page.tsx` (config & launch):** model selector (reuse existing); dataset selector + preview with **input**/**reference** column dropdowns (from `/api/datasets` columns); optional system prompt + template (`{input}`); metric select (`GET /api/eval/metrics`) with a **dynamic config form** from the metric schema; run-size (default 100 + "all"); generation params (`max_new_tokens`, `temperature` default 0); **Run eval** → `POST /api/eval/start` → progress view.
- **Progress view (like training):** progress bar, big running average score, ETA, **Cancel**, live-streaming per-example table (idx · score · truncated prediction) via SSE.
- **Run detail / results:** aggregate header; per-example table (input · prediction · reference · score) **sortable by score** to surface worst cases; row expand → JSON breakdown tree (`json_document`) or text diff.
- **Recents:** eval runs in the sidebar on the eval route (like training runs); click opens detail. Backed by `GET /api/eval/runs`.
- **`api/eval-api.ts`:** `startEval`, `openEvalProgress` (EventSource), `cancelEval`, `listEvalRuns`, `getEvalRun`, `listMetrics`. **`eval-runtime-store.ts`:** `currentRunId`, `status`, `progress`, `isEvalRunning` (gates a second run).

## Error handling & edge cases

| Situation | Behavior |
|---|---|
| Per-example generation/scoring error (incl. unparseable JSON, bad reference) | record the example with `score 0` + `error`; **continue** the run; UI flags it |
| Missing/empty dataset, unknown column/metric, invalid metric config, `N = 0` | reject at start → `400` with a clear message |
| Model load failure | run ends `error` with `error_message`, surfaced in the UI |
| Second start while one runs | `409` |
| Cancel | effective after the current example; partial results kept; `avg_score` over scored examples |
| Server restart mid-run | startup reconciles `running` → `interrupted` |
| Determinism | default generation `temperature = 0` (greedy); user can change |
| Long text | full values stored in `eval_results`; UI truncates |

## Testing strategy

Pytest under the Studio venv (`~/.unsloth/studio/unsloth_studio/bin/python`):

1. **Metrics:** per-metric unit tests (exact-match case/strip; ANLS threshold; `json_document` reference-parse, breakdown shape, unparseable→error/0). Registry: `list_metrics()` schema; `make_scorer` unknown-name error.
2. **Runner (GPU-free via dependency injection):** stub `generate` + tiny in-memory `examples` → verify looping, aggregation, `limit`, per-example error handling, cooperative cancellation. No model/GPU.
3. **Job manager:** start with a stub runner → poll status → cancel → assert state transitions + `eval_runs`/`eval_results` persistence.
4. **Routes:** FastAPI `TestClient` with `get_current_subject` overridden + a stubbed job manager — `start` returns `run_id`; `runs`, `runs/{id}`, `metrics`, `cancel`; `409` on concurrent start.
5. **Persistence:** `studio_db` `eval_runs`/`eval_results` CRUD round-trips.
6. **Frontend:** clean build + driving the live app.

## Dependencies

No new Python deps in Phase 1 — metrics reuse `json_score` (rapidfuzz/scipy/dateutil already in `no-torch-runtime.txt`); inference/datasets/jobs reuse existing Studio infrastructure. (Phase 2 adds `jiwer`; Phase 3 adds diffusion/image-metric deps.)
