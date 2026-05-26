# Studio Eval Frontend (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `/eval` UI for Unsloth Studio — configure an eval (model + dataset + column mapping + metric + generation params), launch it, watch live progress over SSE, and browse per-example results with a JSON breakdown — and remove the superseded `document-score` panel.

**Architecture:** A self-contained `src/features/eval/` feature mirroring the `training` feature's layering (API client → zustand runtime store → components → page → sidebar integration). It reuses only low-level helpers (`authFetch`, `readFastApiError`, `checkDatasetFormat`, `listLocalModels`) and the `streamTrainingProgress` SSE-parsing pattern. The backend (`/api/eval/*`) already exists and is the contract this UI consumes.

**Tech Stack:** React 19, TanStack Router (code-based routes), zustand, shadcn/ui (Card, Button, Input, Textarea, Label, Select, Table, Tabs, Progress, Badge), Tailwind, Vite. No TanStack Query (manual `authFetch` + `useEffect`/`useState`). Frontend has no unit-test runner — **the per-task gate is `npm run typecheck`**, and the final gate is `npm run build` + driving the live app.

---

## Backend contract (already implemented — do NOT change)

Routes under `/api/eval` (all require auth; `authFetch` attaches the token):

- `GET  /api/eval/metrics` → `{ metrics: MetricInfo[] }`
- `POST /api/eval/start` → `{ run_id: string }` · `409` if an eval is already running · `400` on bad config (unknown metric, `limit<=0`)
- `POST /api/eval/cancel/{run_id}` → `{ cancelled: true }` · `404` if not the active run
- `GET  /api/eval/runs?limit=&offset=` → `{ runs: EvalRunSummary[], total: number }`
- `GET  /api/eval/runs/{run_id}?limit=&offset=` → `EvalRunDetail` · `404` if unknown
- `GET  /api/eval/progress/{run_id}` → SSE stream. `retry: 3000` first; then events named `progress` (while running) and `complete` (terminal). Each `data:` is a JSON `EvalProgress`. `id:` is the `done` count.

Backend Pydantic shapes (mirror these as TS types):

```
EvalStartRequest {
  model_identifier: string
  dataset: { is_local: boolean, path?: string|null, name?: string|null, split: string="train", subset?: string|null }
  input_column: string
  reference_column: string
  metric_name: string
  metric_config: object = {}
  system_prompt: string = ""
  template?: string | null
  limit?: number | null     // null = all rows; default 100
  max_new_tokens: number = 256
  temperature: number = 0.0
}
EvalProgress { run_id, status, done, total, avg_score, eta_sec?: number|null,
               last_result?: { idx: number, score: number, error?: string|null } | null }
EvalRunSummary { id, status, model_identifier, dataset_ref, metric_name,
                 started_at, ended_at?: string|null, num_examples?: number|null,
                 avg_score?: number|null, display_name?: string|null }
EvalResultRow { idx, input_text?: string|null, prediction_text?: string|null,
                reference_text?: string|null, score?: number|null,
                breakdown?: object|null, error?: string|null }
EvalRunDetail { run: EvalRunSummary, results: EvalResultRow[], total_results: number }
MetricInfo { name, label, reference_kind, config_fields: MetricConfigField[] }
MetricConfigField { name, type, default, label }   // type ∈ "bool" | "float" | "json" | "string"
```

`status` ∈ `running | completed | cancelled | error | interrupted`.

**Two contract facts that shape the UI:**
1. The SSE `last_result` carries only `{idx, score, error}` — **not** prediction text. So the live table shows idx/score/error accumulated as events arrive; full prediction/reference text comes from `GET /api/eval/runs/{id}` once a run is selected/completed.
2. The backend always sends `eta_sec: null` (not computed server-side). The UI computes ETA client-side from elapsed time and `done/total`.

---

## File structure

New feature:
```
studio/frontend/src/features/eval/
├── index.ts                              # barrel: exports used by sidebar + route
├── eval-page.tsx                         # top-level page (Tabs: Configure / Run / History)
├── api/
│   └── eval-api.ts                       # types + all /api/eval calls + SSE stream
├── stores/
│   └── eval-runtime-store.ts             # zustand: currentRunId, status, progress, isEvalRunning, selectedHistoryRunId
├── hooks/
│   ├── use-eval-progress-stream.ts       # connect SSE → runtime store for a runId
│   └── use-eval-history-sidebar.ts       # recents list + change events
└── components/
    ├── breakdown-tree.tsx                # relocated from document-score (ScoreNode tree)
    ├── eval-config-form.tsx              # model + dataset + columns + metric + params + Run
    ├── metric-config-fields.tsx          # dynamic config form from MetricInfo.config_fields
    ├── live-eval-view.tsx                # progress bar, avg score, ETA, cancel, live idx/score table
    └── eval-run-detail.tsx               # fetch a run → aggregate + sortable results + expand→breakdown
```

New route + integration edits:
```
studio/frontend/src/app/routes/eval.tsx             # NEW route
studio/frontend/src/app/router.tsx                  # register eval, drop document-score
studio/frontend/src/components/app-sidebar.tsx      # Score→Eval nav + eval Recents; drop document-score nav
```

Removals:
```
studio/frontend/src/app/routes/document-score.tsx   # DELETE
studio/frontend/src/features/document-score/         # DELETE (whole dir)
```

---

## Conventions every task must follow

- First two lines of every new `.ts`/`.tsx` file are the license header (copy from any sibling):
  ```
  // SPDX-License-Identifier: AGPL-3.0-only
  // Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
  ```
- Import the auth fetch as `import { authFetch } from "@/features/auth";` and error formatter as `import { readFastApiError } from "@/lib/format-fastapi-error";`.
- Use `@/` path alias (maps to `src/`). shadcn components live at `@/components/ui/<name>`.
- Per-task gate: `cd studio/frontend && npm run typecheck` must pass (0 errors) before the task is done. (Pre-existing errors in unrelated files: there should be none — a clean build was verified earlier; if any unrelated error appears, report it, don't "fix" by editing unrelated files.)

---

### Task 1: Eval API client + types

**Files:**
- Create: `studio/frontend/src/features/eval/api/eval-api.ts`

This file is pure logic (no JSX) and is the foundation every later task imports. Mirror `studio/frontend/src/features/training/api/train-api.ts` for the SSE parser.

- [ ] **Step 1: Write the full file**

```typescript
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

// ── Types (mirror studio/backend/models/eval.py) ──────────────────────

export type EvalStatus =
  | "running"
  | "completed"
  | "cancelled"
  | "error"
  | "interrupted";

export interface EvalDatasetRef {
  is_local: boolean;
  path?: string | null;
  name?: string | null;
  split: string;
  subset?: string | null;
}

export interface EvalStartRequest {
  model_identifier: string;
  dataset: EvalDatasetRef;
  input_column: string;
  reference_column: string;
  metric_name: string;
  metric_config: Record<string, unknown>;
  system_prompt: string;
  template?: string | null;
  limit: number | null; // null = all rows
  max_new_tokens: number;
  temperature: number;
}

export interface EvalLastResult {
  idx: number;
  score: number;
  error?: string | null;
}

export interface EvalProgress {
  run_id: string;
  status: EvalStatus;
  done: number;
  total: number;
  avg_score: number;
  eta_sec?: number | null;
  last_result?: EvalLastResult | null;
}

export interface EvalRunSummary {
  id: string;
  status: EvalStatus;
  model_identifier: string;
  dataset_ref: string;
  metric_name: string;
  started_at: string;
  ended_at?: string | null;
  num_examples?: number | null;
  avg_score?: number | null;
  display_name?: string | null;
}

// ScoreNode breakdown produced by the json_document metric.
export interface ScoreNode {
  score: number;
  n_leaves: number;
  note?: string;
  matched_option?: number;
  children?: Record<string, ScoreNode> | ScoreNode[];
}

export interface EvalResultRow {
  idx: number;
  input_text?: string | null;
  prediction_text?: string | null;
  reference_text?: string | null;
  score?: number | null;
  breakdown?: ScoreNode | null;
  error?: string | null;
}

export interface EvalRunDetail {
  run: EvalRunSummary;
  results: EvalResultRow[];
  total_results: number;
}

export interface MetricConfigField {
  name: string;
  type: "bool" | "float" | "json" | "string" | string;
  default: unknown;
  label: string;
}

export interface MetricInfo {
  name: string;
  label: string;
  reference_kind: string;
  config_fields: MetricConfigField[];
}

// ── Helpers ───────────────────────────────────────────────────────────

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

// ── Calls ─────────────────────────────────────────────────────────────

export async function listMetrics(): Promise<MetricInfo[]> {
  const res = await authFetch("/api/eval/metrics");
  const data = await parseJson<{ metrics: MetricInfo[] }>(res);
  return data.metrics;
}

export async function startEval(
  payload: EvalStartRequest,
): Promise<{ run_id: string }> {
  const res = await authFetch("/api/eval/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJson<{ run_id: string }>(res);
}

export async function cancelEval(runId: string): Promise<void> {
  const res = await authFetch(`/api/eval/cancel/${encodeURIComponent(runId)}`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res));
  }
}

export async function listEvalRuns(
  limit = 50,
  offset = 0,
): Promise<{ runs: EvalRunSummary[]; total: number }> {
  const res = await authFetch(`/api/eval/runs?limit=${limit}&offset=${offset}`);
  return parseJson<{ runs: EvalRunSummary[]; total: number }>(res);
}

export async function getEvalRun(
  runId: string,
  limit = 200,
  offset = 0,
): Promise<EvalRunDetail> {
  const res = await authFetch(
    `/api/eval/runs/${encodeURIComponent(runId)}?limit=${limit}&offset=${offset}`,
  );
  return parseJson<EvalRunDetail>(res);
}

// ── SSE progress stream (mirrors streamTrainingProgress) ───────────────

type EvalEventName = "progress" | "complete";

interface ParsedEvalEvent {
  event: EvalEventName;
  payload: EvalProgress;
}

function parseEvalSseEvent(rawEvent: string): ParsedEvalEvent | null {
  const lines = rawEvent.split(/\r?\n/);
  let eventName: EvalEventName = "progress";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (!line) continue;
    if (line.startsWith("event:")) {
      const value = line.slice(6).trim();
      if (value === "progress" || value === "complete") eventName = value;
      continue;
    }
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (dataLines.length === 0) return null;
  const parsed = JSON.parse(dataLines.join("\n")) as EvalProgress;
  return { event: eventName, payload: parsed };
}

export async function streamEvalProgress(options: {
  runId: string;
  signal: AbortSignal;
  onOpen?: () => void;
  onEvent: (event: ParsedEvalEvent) => void;
}): Promise<void> {
  const res = await authFetch(
    `/api/eval/progress/${encodeURIComponent(options.runId)}`,
    { method: "GET", signal: options.signal },
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
  if (!res.body) throw new Error("Progress stream unavailable");

  options.onOpen?.();

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep = buffer.search(/\r?\n\r?\n/);
    while (sep >= 0) {
      const rawEvent = buffer.slice(0, sep);
      const sepLen = buffer[sep] === "\r" ? 4 : 2;
      buffer = buffer.slice(sep + sepLen);

      if (!rawEvent.startsWith("retry:")) {
        try {
          const evt = parseEvalSseEvent(rawEvent);
          if (evt) options.onEvent(evt);
        } catch (error) {
          if (!isAbortError(error)) throw error;
        }
      }
      sep = buffer.search(/\r?\n\r?\n/);
    }
  }
}

export { isAbortError };
```

- [ ] **Step 2: Typecheck**

Run: `cd studio/frontend && npm run typecheck`
Expected: PASS (no new errors).

- [ ] **Step 3: Commit**

```bash
git add studio/frontend/src/features/eval/api/eval-api.ts
git commit -m "feat(studio/eval-ui): eval API client + types"
```

---

### Task 2: Eval runtime store

**Files:**
- Create: `studio/frontend/src/features/eval/stores/eval-runtime-store.ts`

A zustand store mirroring the lean parts of `training-runtime-store`. Holds the active run id, live progress, an accumulated list of per-example mini-results (idx/score/error from SSE), `isEvalRunning` (gates a second start), `startedAtMs` (for client ETA), and `selectedHistoryRunId` (which run the History view/sidebar shows). Reference the training store at `studio/frontend/src/features/training/stores/training-runtime-store.ts` for the `create<...>()` style.

- [ ] **Step 1: Write the full file**

```typescript
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { EvalProgress, EvalStatus } from "../api/eval-api";

export interface EvalMiniResult {
  idx: number;
  score: number;
  error?: string | null;
}

interface EvalRuntimeState {
  currentRunId: string | null;
  status: EvalStatus | "idle";
  done: number;
  total: number;
  avgScore: number;
  startedAtMs: number | null;
  isEvalRunning: boolean;
  liveResults: EvalMiniResult[]; // accumulated from SSE last_result, deduped by idx
  startError: string | null;
  selectedHistoryRunId: string | null;

  beginRun: (runId: string, total: number) => void;
  applyProgress: (p: EvalProgress) => void;
  finishRun: (status: EvalStatus) => void;
  setStartError: (msg: string | null) => void;
  setSelectedHistoryRunId: (id: string | null) => void;
  resetRuntime: () => void;
}

const initial = {
  currentRunId: null as string | null,
  status: "idle" as EvalStatus | "idle",
  done: 0,
  total: 0,
  avgScore: 0,
  startedAtMs: null as number | null,
  isEvalRunning: false,
  liveResults: [] as EvalMiniResult[],
  startError: null as string | null,
};

export const useEvalRuntimeStore = create<EvalRuntimeState>()((set) => ({
  ...initial,
  selectedHistoryRunId: null,

  beginRun: (runId, total) =>
    set({
      currentRunId: runId,
      status: "running",
      done: 0,
      total,
      avgScore: 0,
      startedAtMs: Date.now(),
      isEvalRunning: true,
      liveResults: [],
      startError: null,
      selectedHistoryRunId: runId,
    }),

  applyProgress: (p) =>
    set((s) => {
      const liveResults = s.liveResults.slice();
      if (p.last_result) {
        const i = liveResults.findIndex((r) => r.idx === p.last_result!.idx);
        const entry: EvalMiniResult = {
          idx: p.last_result.idx,
          score: p.last_result.score,
          error: p.last_result.error,
        };
        if (i >= 0) liveResults[i] = entry;
        else liveResults.push(entry);
      }
      return {
        status: p.status,
        done: p.done,
        total: p.total || s.total,
        avgScore: p.avg_score,
        isEvalRunning: p.status === "running",
        liveResults,
      };
    }),

  finishRun: (status) => set({ status, isEvalRunning: false }),
  setStartError: (msg) => set({ startError: msg }),
  setSelectedHistoryRunId: (id) => set({ selectedHistoryRunId: id }),
  resetRuntime: () => set({ ...initial }),
}));
```

- [ ] **Step 2: Typecheck**

Run: `cd studio/frontend && npm run typecheck`
Expected: PASS. (If `zustand` import style differs, match `training-runtime-store.ts` exactly — it is the source of truth for the `create` signature in this repo.)

- [ ] **Step 3: Commit**

```bash
git add studio/frontend/src/features/eval/stores/eval-runtime-store.ts
git commit -m "feat(studio/eval-ui): eval runtime store"
```

---

### Task 3: Relocate BreakdownTree into the eval feature

**Files:**
- Create: `studio/frontend/src/features/eval/components/breakdown-tree.tsx`

Copy the `BreakdownTree` + `scoreColor` from `studio/frontend/src/features/document-score/document-score-page.tsx` (lines 64–121), but import `ScoreNode` from the eval API (not the document-score API, which is being deleted). Export both so other eval components and a later text-diff fallback can reuse `scoreColor`.

- [ ] **Step 1: Write the full file**

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ScoreNode } from "../api/eval-api";

export function scoreColor(score: number): string {
  if (score >= 0.999) return "text-emerald-500";
  if (score >= 0.5) return "text-amber-500";
  return "text-red-500";
}

/** Recursive per-field breakdown row for a json_document ScoreNode. */
export function BreakdownTree({
  label,
  node,
  depth,
}: {
  label: string;
  node: ScoreNode;
  depth: number;
}) {
  const children = node.children;
  const entries: [string, ScoreNode][] = Array.isArray(children)
    ? children.map((c, i) => [`[${i}]`, c])
    : children
      ? Object.entries(children)
      : [];

  return (
    <div className="text-sm">
      <div
        className="flex items-center justify-between gap-3 py-1"
        style={{ paddingLeft: `${depth * 16}px` }}
      >
        <span className="truncate font-medium text-muted-foreground">
          {label}
          {node.note ? (
            <span className="ml-2 text-xs text-red-500/80">({node.note})</span>
          ) : null}
        </span>
        <span className={cn("tabular-nums font-semibold", scoreColor(node.score))}>
          {node.score.toFixed(3)}
        </span>
      </div>
      {entries.map(([k, child]) => (
        <BreakdownTree key={k} label={k} node={child} depth={depth + 1} />
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `cd studio/frontend && npm run typecheck`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add studio/frontend/src/features/eval/components/breakdown-tree.tsx
git commit -m "feat(studio/eval-ui): relocate BreakdownTree into eval feature"
```

---

### Task 4: Eval config form + dynamic metric config

**Files:**
- Create: `studio/frontend/src/features/eval/components/metric-config-fields.tsx`
- Create: `studio/frontend/src/features/eval/components/eval-config-form.tsx`

The form owns all config in local React state and calls `onStart(payload: EvalStartRequest)`. It does NOT call the backend `/start` itself — the page wires that (Task 7). Reuse `listLocalModels` (`@/features/training/api/models-api`) for a model dropdown and `checkDatasetFormat` (`@/features/training/api/datasets-api`) for column detection. The dataset column inputs are plain `Input`s; a **Detect columns** button populates clickable column chips + a small preview, but the user can always type column names manually (robust even if detection fails).

shadcn components to use (verify exact export names against `studio/frontend/src/components/ui/`): `Card`/`CardHeader`/`CardTitle`/`CardContent` (`@/components/ui/card`), `Button`, `Input`, `Textarea`, `Label`, `Select`/`SelectTrigger`/`SelectValue`/`SelectContent`/`SelectItem` (`@/components/ui/select`), `Badge`, `Switch` (`@/components/ui/switch`).

- [ ] **Step 1: Write `metric-config-fields.tsx`**

Renders one input per `MetricConfigField` based on its `type`. Value map is `Record<string, unknown>`; `onChange` replaces the whole map. `bool` → `Switch`; `float` → numeric `Input`; `json`/`string` → `Textarea` (json validated lazily at submit — see form). Keep it dumb (controlled).

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import type { MetricConfigField } from "../api/eval-api";

export function MetricConfigFields({
  fields,
  values,
  onChange,
}: {
  fields: MetricConfigField[];
  values: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}) {
  if (fields.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        This metric has no extra options.
      </p>
    );
  }
  const set = (name: string, value: unknown) =>
    onChange({ ...values, [name]: value });

  return (
    <div className="flex flex-col gap-3">
      {fields.map((f) => {
        const current = values[f.name] ?? f.default;
        if (f.type === "bool") {
          return (
            <div key={f.name} className="flex items-center justify-between gap-3">
              <Label htmlFor={`mc-${f.name}`}>{f.label}</Label>
              <Switch
                id={`mc-${f.name}`}
                checked={Boolean(current)}
                onCheckedChange={(v) => set(f.name, v)}
              />
            </div>
          );
        }
        if (f.type === "float") {
          return (
            <div key={f.name} className="flex flex-col gap-1.5">
              <Label htmlFor={`mc-${f.name}`}>{f.label}</Label>
              <Input
                id={`mc-${f.name}`}
                type="number"
                step="0.01"
                value={current === undefined || current === null ? "" : String(current)}
                onChange={(e) =>
                  set(f.name, e.target.value === "" ? null : Number(e.target.value))
                }
              />
            </div>
          );
        }
        // json | string → textarea (json parsed at submit-time by the form)
        return (
          <div key={f.name} className="flex flex-col gap-1.5">
            <Label htmlFor={`mc-${f.name}`}>{f.label}</Label>
            <Textarea
              id={`mc-${f.name}`}
              value={typeof current === "string" ? current : current == null ? "" : JSON.stringify(current, null, 2)}
              onChange={(e) => set(f.name, e.target.value)}
              className="min-h-24 font-mono text-xs"
              spellCheck={false}
            />
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: Write `eval-config-form.tsx`**

State (all `useState`): `modelIdentifier` (string), `localModels` (LocalModelInfo[]), `datasetIsLocal` (bool), `datasetName` (string, HF repo or local path), `split` ("train"), `subset` (""), `inputColumn` (""), `referenceColumn` (""), `detectedColumns` (string[]), `previewSample` (Record<string,unknown>|null), `detecting` (bool), `detectError` (string|null), `systemPrompt` (""), `template` (""), `metrics` (MetricInfo[]), `metricName` (""), `metricConfig` (Record<string,unknown>), `runAll` (bool, default false), `limit` (number, default 100), `maxNewTokens` (256), `temperature` (0), `hfToken` ("").

Effects:
- on mount: `listMetrics()` → `setMetrics`; default `metricName` to first metric and seed `metricConfig` from its `config_fields` defaults. `listLocalModels()` → `setLocalModels` (ignore errors).
- when `metricName` changes: reseed `metricConfig` defaults for the newly-selected metric.

`detectColumns()`: guard `datasetName` non-empty; `setDetecting(true)`; call `checkDatasetFormat({ datasetName, hfToken: hfToken || null, subset: subset || null, split: split || "train" })`; on success set `detectedColumns = res.columns`, `previewSample = res.preview_samples?.[0] ?? null`, and if columns exist and inputs empty, prefill `inputColumn`/`referenceColumn` with the first two columns; on error set `detectError`. Always `setDetecting(false)`.

`buildPayload()`: validate `modelIdentifier`, `datasetName`, `inputColumn`, `referenceColumn`, `metricName` non-empty (else throw with a clear message). Parse any `json`-typed metric-config textarea values: for each selected metric field of type `json`, if the value is a non-empty string, `JSON.parse` it (throw a clear error on failure); empty string → omit the key. Assemble:

```ts
const payload: EvalStartRequest = {
  model_identifier: modelIdentifier.trim(),
  dataset: {
    is_local: datasetIsLocal,
    name: datasetIsLocal ? null : datasetName.trim(),
    path: datasetIsLocal ? datasetName.trim() : null,
    split: split.trim() || "train",
    subset: subset.trim() || null,
  },
  input_column: inputColumn.trim(),
  reference_column: referenceColumn.trim(),
  metric_name: metricName,
  metric_config: parsedMetricConfig,
  system_prompt: systemPrompt,
  template: template.trim() ? template : null,
  limit: runAll ? null : Math.max(1, Math.floor(limit)),
  max_new_tokens: Math.max(1, Math.floor(maxNewTokens)),
  temperature: Number.isFinite(temperature) ? temperature : 0,
};
```

`onSubmit`: try `buildPayload()`; on throw set a local `formError`; else call `props.onStart(payload)`.

Layout (each block a `Card` with a title; mirror the spacing in `document-score-page.tsx` and `eval` siblings):
1. **Model** — `Select` of `localModels` (value = `m.id`, label = `m.display_name`) AND an `Input` for free-text HF repo / path that overrides; simplest: an `Input` bound to `modelIdentifier` with a small `Select` beside it whose `onValueChange` sets `modelIdentifier`. Add a hint: "Trained checkpoint, Hugging Face repo, or local path." Optional `Input type="password"` for `hfToken`.
2. **Dataset** — a source toggle (`Switch` "Local file" ↔ HF) bound to `datasetIsLocal`; `Input` for `datasetName` (placeholder switches: HF → `org/dataset`, local → `/path/to/data.jsonl`); `Input`s for `split` and `subset`; **Detect columns** `Button` (disabled while `detecting`); show `detectError` in red; render `detectedColumns` as `Badge`s — clicking a badge: first click with empty input fills `inputColumn`, second fills `referenceColumn` (or simpler: two rows each with the column badges that set that row's value). Then two `Input`s: **Input column** (`inputColumn`) and **Reference column** (`referenceColumn`). If `previewSample`, show it as a `<pre>` of `JSON.stringify(previewSample, null, 2)` truncated.
3. **Prompt** — `Textarea` System prompt (`systemPrompt`); `Textarea` Template with hint "use `{input}` placeholder; leave blank to send the raw input column" (`template`).
4. **Metric** — `Select` of `metrics` (value `m.name`, label `m.label`); below it `<MetricConfigFields fields={selectedMetric.config_fields} values={metricConfig} onChange={setMetricConfig} />`.
5. **Run size & generation** — `Switch` "Evaluate all rows" (`runAll`); when off, `Input type="number"` for `limit` (default 100); `Input type="number"` for `maxNewTokens`; `Input type="number" step="0.1"` for `temperature` (hint "0 = greedy/deterministic").
6. **Footer** — `Button` "Run eval" (disabled when `props.disabled` — i.e. an eval already running) + `formError` in red.

Props:
```ts
export function EvalConfigForm({
  disabled,
  onStart,
}: {
  disabled: boolean;
  onStart: (payload: EvalStartRequest) => void;
}) { /* ... */ }
```

Imports needed: `useState`, `useEffect`; `listMetrics`, type `EvalStartRequest`, `MetricInfo` from `../api/eval-api`; `listLocalModels`, type `LocalModelInfo` from `@/features/training/api/models-api`; `checkDatasetFormat` from `@/features/training/api/datasets-api`; `MetricConfigFields` from `./metric-config-fields`; shadcn components above.

Write complete, compiling TSX implementing the above. Keep it readable and match the existing Tailwind idiom (`flex flex-col gap-*`, `text-sm text-muted-foreground`, `font-mono text-xs` for code areas).

- [ ] **Step 3: Typecheck**

Run: `cd studio/frontend && npm run typecheck`
Expected: PASS. Verify the shadcn `Switch`/`Select` export names against `studio/frontend/src/components/ui/switch.tsx` and `select.tsx`; fix imports if they differ.

- [ ] **Step 4: Commit**

```bash
git add studio/frontend/src/features/eval/components/eval-config-form.tsx studio/frontend/src/features/eval/components/metric-config-fields.tsx
git commit -m "feat(studio/eval-ui): eval config form + dynamic metric config"
```

---

### Task 5: Live eval view + SSE progress hook

**Files:**
- Create: `studio/frontend/src/features/eval/hooks/use-eval-progress-stream.ts`
- Create: `studio/frontend/src/features/eval/components/live-eval-view.tsx`

- [ ] **Step 1: Write `use-eval-progress-stream.ts`**

A hook that, given an active `runId`, opens `streamEvalProgress` and pipes events into the runtime store. Aborts on unmount or runId change. Mirror how `use-training-runtime-lifecycle.ts` manages an `AbortController` (reference: `studio/frontend/src/features/training/hooks/`).

```typescript
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { streamEvalProgress, isAbortError } from "../api/eval-api";
import { useEvalRuntimeStore } from "../stores/eval-runtime-store";

/** While `runId` is set and the run is active, stream progress into the store. */
export function useEvalProgressStream(runId: string | null, enabled: boolean) {
  const applyProgress = useEvalRuntimeStore((s) => s.applyProgress);
  const finishRun = useEvalRuntimeStore((s) => s.finishRun);

  useEffect(() => {
    if (!runId || !enabled) return;
    const controller = new AbortController();
    let cancelled = false;

    void (async () => {
      try {
        await streamEvalProgress({
          runId,
          signal: controller.signal,
          onEvent: ({ event, payload }) => {
            applyProgress(payload);
            if (event === "complete") finishRun(payload.status);
          },
        });
      } catch (error) {
        if (!cancelled && !isAbortError(error)) {
          // Stream dropped (e.g. server restart). Mark not-running; the
          // detail view re-fetches authoritative state from the DB.
          finishRun("interrupted");
        }
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [runId, enabled, applyProgress, finishRun]);
}
```

- [ ] **Step 2: Write `live-eval-view.tsx`**

Reads the runtime store. Shows: a header with the running average score (big, `tabular-nums`, colored via `scoreColor`), a `Progress` bar (`value={total ? (done/total)*100 : 0}`), `done / total`, a client-computed ETA, and a **Cancel** button (calls `cancelEval(currentRunId)` then `finishRun("cancelled")` optimistically; ignore/ toast errors). Below: a live `Table` of accumulated `liveResults` (idx · score · error badge). When `status` is terminal, render `<EvalRunDetail runId={currentRunId} />` (Task 6) instead of/below the live table so full text appears.

ETA helper (inline): `etaSec = startedAtMs && done > 0 && total > done ? ((Date.now()-startedAtMs)/done) * (total-done) / 1000 : null;` — but `Date.now()` in render won't tick; recompute from `done` changes is fine (updates each SSE event). Format with a small `formatDuration(sec)` (`${m}m ${s}s` or `${s}s`).

Props: `export function LiveEvalView({ onCancelled }: { onCancelled?: () => void })` — reads `currentRunId`, `status`, `done`, `total`, `avgScore`, `startedAtMs`, `liveResults`, `isEvalRunning` from the store. Use `scoreColor` from `./breakdown-tree`. Use `Badge` for errors. Use `Progress` from `@/components/ui/progress`. When `!currentRunId`, render a muted "No active eval." placeholder.

Cancel handler:
```ts
async function handleCancel() {
  if (!currentRunId) return;
  try {
    await cancelEval(currentRunId);
  } catch (err) {
    toast.error("Failed to cancel eval", { description: err instanceof Error ? err.message : undefined });
  }
}
```
(`toast` from `@/lib/toast`.) The store's `finishRun("cancelled")` will arrive via the SSE `complete` event, so don't force it here.

Write complete compiling TSX. Import `EvalRunDetail` from `./eval-run-detail` (created next task — for ordering, this import will resolve once Task 6 lands; if implementing strictly in order, stub a tiny local fallback and replace in Task 6, OR implement Task 6 first then this. **Recommended: do Task 6 before Task 5's detail wiring.** If typecheck runs before Task 6 exists, temporarily render the live table only and add the `<EvalRunDetail>` import in Task 6 — note this in the commit.)

- [ ] **Step 3: Typecheck**

Run: `cd studio/frontend && npm run typecheck`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add studio/frontend/src/features/eval/hooks/use-eval-progress-stream.ts studio/frontend/src/features/eval/components/live-eval-view.tsx
git commit -m "feat(studio/eval-ui): live eval progress view + SSE hook"
```

---

### Task 6: Run detail / results view

**Files:**
- Create: `studio/frontend/src/features/eval/components/eval-run-detail.tsx`

Fetches `getEvalRun(runId)` on mount / when `runId` changes (`useEffect` + `useState`, with a loading + error state). Renders:
- **Aggregate header** (`Card`): run status `Badge`, `metric_name`, `model_identifier`, `dataset_ref`, big avg score (`run.avg_score`, colored), `num_examples`, started/ended times.
- **Results table** (`Table`): columns Idx · Score · Input (truncated) · Prediction (truncated) · Reference (truncated). **Sortable by score** — a `useState<"idx"|"score-asc"|"score-desc">` with a clickable Score header cycling asc/desc; default `score-asc` to surface worst cases. Error rows show an error `Badge`.
- **Row expansion**: clicking a row toggles an expanded panel (track `expandedIdx: number|null`). Expanded content: if `row.breakdown` → `<BreakdownTree label="document" node={row.breakdown} depth={0} />`; else a two-column text diff-ish view: full `prediction_text` vs `reference_text` in side-by-side `<pre className="whitespace-pre-wrap font-mono text-xs">` blocks. Show `row.error` if present.

Props:
```ts
export function EvalRunDetail({ runId }: { runId: string }) { /* ... */ }
```

Imports: `useState`, `useEffect`; `getEvalRun`, types `EvalRunDetail as EvalRunDetailData`, `EvalResultRow` from `../api/eval-api`; `BreakdownTree`, `scoreColor` from `./breakdown-tree`; `Card`/`CardHeader`/`CardTitle`/`CardContent`, `Table`/`TableHeader`/`TableBody`/`TableRow`/`TableHead`/`TableCell`, `Badge`, `Button` from shadcn. A `truncate(s, n=120)` helper for cells.

Sorting:
```ts
const sorted = [...detail.results].sort((a, b) => {
  if (sort === "idx") return a.idx - b.idx;
  const sa = a.score ?? 0, sb = b.score ?? 0;
  return sort === "score-asc" ? sa - sb : sb - sa;
});
```

Write complete compiling TSX.

- [ ] **Step 2: Typecheck**

Run: `cd studio/frontend && npm run typecheck`
Expected: PASS. If Task 5 left a temporary stub, now wire `LiveEvalView` to render `<EvalRunDetail runId={currentRunId} />` on terminal status and re-typecheck.

- [ ] **Step 3: Commit**

```bash
git add studio/frontend/src/features/eval/components/eval-run-detail.tsx studio/frontend/src/features/eval/components/live-eval-view.tsx
git commit -m "feat(studio/eval-ui): run detail with sortable results + breakdown"
```

---

### Task 7: Eval page + history sidebar hook + barrel

**Files:**
- Create: `studio/frontend/src/features/eval/hooks/use-eval-history-sidebar.ts`
- Create: `studio/frontend/src/features/eval/eval-page.tsx`
- Create: `studio/frontend/src/features/eval/index.ts`

- [ ] **Step 1: Write `use-eval-history-sidebar.ts`**

Mirror `use-training-history-sidebar.ts`: fetch `listEvalRuns()` when `enabled`, expose `{ items }`, and re-fetch on a custom "eval runs changed" event. Define and export an emitter so the page can refresh the sidebar after a run starts/finishes.

```typescript
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { listEvalRuns, type EvalRunSummary } from "../api/eval-api";

const EVAL_RUNS_CHANGED = "unsloth:eval-runs-changed";

export function emitEvalRunsChanged(): void {
  window.dispatchEvent(new CustomEvent(EVAL_RUNS_CHANGED));
}

export function useEvalHistorySidebarItems(enabled: boolean): {
  items: EvalRunSummary[];
} {
  const [items, setItems] = useState<EvalRunSummary[]>([]);

  useEffect(() => {
    if (!enabled) return;
    let mounted = true;
    const load = () =>
      listEvalRuns()
        .then((res) => {
          if (mounted) setItems(res.runs);
        })
        .catch(() => {});
    load();
    const onChange = () => load();
    window.addEventListener(EVAL_RUNS_CHANGED, onChange);
    return () => {
      mounted = false;
      window.removeEventListener(EVAL_RUNS_CHANGED, onChange);
    };
  }, [enabled]);

  return { items };
}
```

- [ ] **Step 2: Write `eval-page.tsx`**

Composes the feature. Tabs `Configure | Run | History` (shadcn `Tabs`). Logic mirrors `studio-page.tsx`'s tab gating:
- `isEvalRunning` and `currentRunId` from the runtime store; `selectedHistoryRunId` from the store.
- `useEvalProgressStream(currentRunId, isEvalRunning)` to keep the stream alive while on the page.
- A `[requestedTab, setRequestedTab]` state. Effective tab: if `isEvalRunning` → force `"run"` (unless user explicitly opened history); when a run finishes, stay on `"run"` so the detail shows.
- **Start handler**: `async (payload) => { try { const { run_id } = await startEval(payload); beginRun(run_id, payload.limit ?? 0); emitEvalRunsChanged(); setRequestedTab("run"); } catch (err) { setStartError(...) ; if 409 → toast "An eval is already running" } }`. (Map a `409`/message to a friendly toast; `readFastApiError` already gives a message string via the thrown Error.)
- **Configure tab**: `<EvalConfigForm disabled={isEvalRunning} onStart={handleStart} />` + show `startError` if set.
- **Run tab**: `<LiveEvalView />`.
- **History tab**: if `selectedHistoryRunId` → `<EvalRunDetail runId={selectedHistoryRunId} />` with a "← Back to all runs" button that clears it; else a `listEvalRuns()`-backed list of cards (id, model, dataset, metric, status badge, avg score, started time) where clicking sets `selectedHistoryRunId`.
- When `emitEvalRunsChanged` fires after finish, the sidebar refreshes (handled by the hook). Also call `emitEvalRunsChanged()` once when `isEvalRunning` transitions true→false (use an effect watching `isEvalRunning`).
- A page heading "Eval" + one-line description, matching `document-score-page.tsx`'s header styling.

Wrap content in `<div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 p-6 overflow-y-auto">`.

Export: `export function EvalPage()`.

- [ ] **Step 3: Write `index.ts` (barrel)**

```typescript
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { EvalPage } from "./eval-page";
export { useEvalRuntimeStore } from "./stores/eval-runtime-store";
export {
  useEvalHistorySidebarItems,
  emitEvalRunsChanged,
} from "./hooks/use-eval-history-sidebar";
export type { EvalRunSummary } from "./api/eval-api";
```

- [ ] **Step 4: Typecheck**

Run: `cd studio/frontend && npm run typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add studio/frontend/src/features/eval/eval-page.tsx studio/frontend/src/features/eval/index.ts studio/frontend/src/features/eval/hooks/use-eval-history-sidebar.ts
git commit -m "feat(studio/eval-ui): eval page + history sidebar hook + barrel"
```

---

### Task 8: Route + sidebar integration; remove document-score

**Files:**
- Create: `studio/frontend/src/app/routes/eval.tsx`
- Modify: `studio/frontend/src/app/router.tsx`
- Modify: `studio/frontend/src/components/app-sidebar.tsx`
- Delete: `studio/frontend/src/app/routes/document-score.tsx`
- Delete: `studio/frontend/src/features/document-score/` (whole dir)

- [ ] **Step 1: Create the eval route**

`studio/frontend/src/app/routes/eval.tsx` (clone `routes/studio.tsx`):

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const EvalPage = lazy(() =>
  import("@/features/eval").then((m) => ({ default: m.EvalPage })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/eval",
  staticData: { title: "Eval" },
  beforeLoad: () => requireAuth(),
  component: EvalPage,
});
```

- [ ] **Step 2: Register eval, drop document-score in `router.tsx`**

Replace the document-score import line:
```ts
import { Route as documentScoreRoute } from "./routes/document-score";
```
with:
```ts
import { Route as evalRoute } from "./routes/eval";
```
In the `rootRoute.addChildren([...])` array, replace `documentScoreRoute,` with `evalRoute,`.

- [ ] **Step 3: Sidebar — Score→Eval nav**

In `studio/frontend/src/components/app-sidebar.tsx`, replace the entire `Score` `NavItem` block (lines ~512–525) with an `Eval` one. Use an existing imported icon — reuse `TestTubeOutlineIcon` is already taken by Train; use `CursorInfo02Icon`? No — pick a fitting already-imported icon. The cleanest: import a new icon. Add `ChartBarLineIcon` (or `Analytics01Icon`) to the `@hugeicons/core-free-icons` import block; if unsure an icon exists, reuse `LayoutAlignLeftIcon` (already imported, currently used by Score) to minimize risk:

```tsx
<NavItem
  icon={LayoutAlignLeftIcon}
  label="Eval"
  active={pathname === "/eval" || pathname.startsWith("/eval/")}
  disabled={chatOnly}
  onClick={() => {
    if (chatOnly) return;
    navigate({ to: "/eval" });
    closeMobileIfOpen();
  }}
/>
```

- [ ] **Step 4: Sidebar — eval Recents + imports**

Mirror the training "Recent Runs" block for eval. Add near the training imports:
```ts
import {
  useEvalHistorySidebarItems,
  useEvalRuntimeStore,
} from "@/features/eval";
import type { EvalRunSummary } from "@/features/eval";
```
Add route + data wiring inside `AppSidebar` (near the training-runs wiring, ~lines 209–246):
```ts
const isEvalRoute = pathname === "/eval" || pathname.startsWith("/eval/");
const [evalRunsOpen, setEvalRunsOpen] = useState(true);
useEffect(() => { if (isEvalRoute) setEvalRunsOpen(true); }, [isEvalRoute]);
const { items: evalRunItems } = useEvalHistorySidebarItems(!chatOnly && isEvalRoute);
const selectedEvalRunId = useEvalRuntimeStore((s) => s.selectedHistoryRunId);
const setSelectedEvalRunId = useEvalRuntimeStore((s) => s.setSelectedHistoryRunId);
```
Add a new collapsible block in `SidebarContent` (after the training "Recent Runs" block, ~line 697) gated on `isEvalRoute && evalRunItems.length > 0 && !chatOnly`. Reuse `runStatusDotClass`/`formatRelativeShort` (both already in the file; note `runStatusDotClass` is typed for `TrainingRunSummary["status"]` but the string union is compatible — if TS complains, widen its param to `string` or cast). Each item: status dot, `run.display_name ?? run.model_identifier`, `formatRelativeShort(run.started_at)`, and `run.dataset_ref` on the second line. On click: `setSelectedEvalRunId(run.id); closeMobileIfOpen();`. **No rename/delete** for eval (backend has no such endpoints) — omit the dropdown menu.

Use this block (adapt class names from the training block):
```tsx
{isEvalRoute && evalRunItems.length > 0 && !chatOnly && (
  <Collapsible open={evalRunsOpen} onOpenChange={setEvalRunsOpen} asChild>
    <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
      <SidebarGroupLabel className={cn("sidebar-sticky-label", scrolled && "is-scrolled")} asChild>
        <CollapsibleTrigger className="cursor-pointer flex w-full items-center justify-between">
          Recents
          <ChevronDown className="size-3.5 transition-transform duration-200 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg]" />
        </CollapsibleTrigger>
      </SidebarGroupLabel>
      <CollapsibleContent>
        <SidebarGroupContent className="px-2">
          <SidebarMenu>
            {evalRunItems.map((run) => (
              <SidebarMenuItem key={run.id} className="relative">
                <SidebarMenuButton
                  isActive={selectedEvalRunId === run.id}
                  className="sidebar-nav-btn h-auto flex-col items-start gap-0.5 py-[5px] rounded-[10px] pl-2.5 pr-2.5 text-[14.5px] tracking-nav font-medium"
                  onClick={() => {
                    setSelectedEvalRunId(run.id);
                    navigate({ to: "/eval" });
                    closeMobileIfOpen();
                  }}
                >
                  <div className="flex w-full items-center gap-[8.5px]">
                    <span className={cn("size-1.5 shrink-0 rounded-full", runStatusDotClass(run.status))} aria-hidden />
                    <span className="truncate">{run.display_name ?? run.model_identifier}</span>
                    <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                      {formatRelativeShort(run.started_at)}
                    </span>
                  </div>
                  <span className="w-full truncate pl-3.5 text-xs text-muted-foreground">
                    {run.dataset_ref}
                  </span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroupContent>
      </CollapsibleContent>
    </SidebarGroup>
  </Collapsible>
)}
```
If `runStatusDotClass(run.status)` fails typecheck (status union mismatch — eval adds `"interrupted"`), change its signature to `function runStatusDotClass(status: string): string` (it already has a `default` case).

- [ ] **Step 5: Delete document-score**

```bash
git rm studio/frontend/src/app/routes/document-score.tsx
git rm -r studio/frontend/src/features/document-score
```

- [ ] **Step 6: Verify no dangling references**

Run: `cd studio/frontend && grep -rn "document-score\|DocumentScorePage\|scoreDocument\|/api/scoring" src` 
Expected: no matches.

- [ ] **Step 7: Typecheck**

Run: `cd studio/frontend && npm run typecheck`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -A studio/frontend/src
git commit -m "feat(studio/eval-ui): /eval route + sidebar nav/recents; remove document-score"
```

---

### Task 9: Production build + drive the live app

**Files:** none (verification only).

- [ ] **Step 1: Production build**

Run: `cd studio/frontend && npm run build`
Expected: build succeeds (`tsc -b && vite build`), `dist/` regenerated.

- [ ] **Step 2: Restart Studio against the new build**

The backend serves `studio/frontend/dist`. Restart the running server:
```bash
~/.unsloth/studio/unsloth_studio/bin/unsloth studio stop 2>/dev/null || true
# then relaunch (background)
~/.unsloth/studio/unsloth_studio/bin/python studio/backend/run.py --port 8899 --host 127.0.0.1
```
Confirm: `curl -s http://127.0.0.1:8899/api/health` and `curl -s http://127.0.0.1:8899/api/eval/metrics` (with auth — or just confirm the route exists via `/api/docs`).

- [ ] **Step 3: Drive the UI**

Log in (`unsloth` / `unsloth-studio`), open `/eval`. Verify: sidebar shows **Eval** (not Score); the config form loads metrics; **Detect columns** works on a small HF dataset (or a local jsonl); starting a tiny eval (limit 2–3) streams live progress and lands on a results table with a JSON breakdown for `json_document`. Sidebar **Recents** lists the run; clicking opens its detail. If Playwright/browser automation is unavailable, do an API-level smoke test (`/api/eval/start` → poll `/api/eval/runs/{id}`) and report that the UI build is served.

- [ ] **Step 4: Final review + finish branch**

Dispatch a final holistic code review over the eval-frontend diff, then use **superpowers:finishing-a-development-branch**.

---

## Self-review notes (author checks)

- **Spec coverage:** Eval nav rename ✓ (T8); document-score + route removal ✓ (T8); BreakdownTree relocation ✓ (T3); model/dataset/column-mapping/system-prompt/template/metric+dynamic-config/run-size/gen-params config ✓ (T4); live progress + avg + ETA + cancel + live table ✓ (T5); run detail sortable + expand→breakdown/text ✓ (T6); Recents sidebar ✓ (T8); api functions `startEval/streamEvalProgress(open)/cancelEval/listEvalRuns/getEvalRun/listMetrics` ✓ (T1); runtime store `currentRunId/status/progress/isEvalRunning` ✓ (T2). `/api/scoring/score` is already removed server-side; T8 removes its last frontend caller.
- **Contract caveats baked in:** SSE `last_result` lacks prediction text (live table = idx/score/error; full text via `getEvalRun`); `eta_sec` computed client-side. Eval recents are view-only (no rename/delete endpoints).
- **Type consistency:** `ScoreNode` defined once in `eval-api.ts`, imported by `breakdown-tree.tsx` and `eval-run-detail.tsx`. `EvalStatus` includes `interrupted`; `runStatusDotClass` may need its param widened to `string` in T8.
- **Ordering risk:** `live-eval-view.tsx` (T5) imports `eval-run-detail.tsx` (T6). Implement T6's component before wiring it into T5, or land the live table first and add the import in T6 (noted in both tasks).
