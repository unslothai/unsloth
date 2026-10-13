// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { authFetch } from "@/features/auth";
import { listRecipes } from "@/features/data-recipes/data/recipes-db";
import type { RecipeRecord } from "@/features/data-recipes/types";
import { listLinkedFolders } from "@/features/rag/api/rag-api";
import type { LinkedFolder } from "@/features/rag/types/rag";
import { Link } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";

type RoutineStep =
  | { kind: "recipe"; label: string; recipe: RecipeRecord["payload"] }
  | { kind: "folder_sync"; label: string; folder_id: string };

type Routine = {
  id: string;
  name: string;
  steps: RoutineStep[];
  interval_minutes: number;
  next_run_at: number;
  enabled: boolean;
};

type RoutineRun = {
  id: string;
  trigger: "manual" | "schedule";
  status: string;
  step_statuses: Array<{ kind: string; label: string; status: string; job_id?: string; artifact_path?: string; summary?: Record<string, number>; error?: string }>;
  error: string | null;
  created_at: number;
  finished_at: number | null;
};

async function request<T>(path: string, method = "GET", body?: object): Promise<T> {
  const response = await authFetch(`/api/routines${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      typeof result?.detail === "string" ? result.detail : `Request failed (${response.status})`,
    );
  }
  return result as T;
}

function when(epoch: number | null): string {
  return epoch ? new Date(epoch * 1000).toLocaleString() : "—";
}

const INTERVALS = [
  [15, "Every 15 minutes"],
  [60, "Every hour"],
  [360, "Every 6 hours"],
  [1440, "Every day"],
  [10080, "Every week"],
] as const;

function scheduleLabel(minutes: number): string {
  return INTERVALS.find(([value]) => value === minutes)?.[1] ?? `Every ${minutes} minutes`;
}

function statusLabel(status: string): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function StatusBadge({ status }: { status: string }) {
  const tone = status === "completed"
    ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
    : status === "failed" || status === "cancelled"
      ? "border-destructive/20 bg-destructive/10 text-destructive"
      : "border-primary/20 bg-primary/10 text-primary";
  return <Badge variant="outline" className={tone}>{statusLabel(status)}</Badge>;
}

function RunResult({ run }: { run: RoutineRun }) {
  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <StatusBadge status={run.status} />
          <span className="text-sm text-muted-foreground">{run.trigger === "schedule" ? "Scheduled run" : "Started manually"}</span>
        </div>
        <span className="text-xs text-muted-foreground">{when(run.created_at)}</span>
      </div>
      {run.step_statuses.length === 0 ? (
        <p className="mt-4 text-sm text-muted-foreground">Waiting for the first step to start.</p>
      ) : (
        <ol className="mt-4 divide-y divide-border/70">
          {run.step_statuses.map((step, index) => (
            <li key={index} className="flex gap-3 py-3 first:pt-0 last:pb-0">
              <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-semibold text-muted-foreground">{index + 1}</span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-medium text-foreground">{step.label}</span>
                  <span className="text-xs text-muted-foreground">{statusLabel(step.status)}</span>
                </div>
                {step.summary && <p className="mt-1 text-xs text-muted-foreground">{Object.entries(step.summary).map(([key, count]) => `${count} ${key}`).join(" · ")}</p>}
                {step.artifact_path && <details className="mt-1 text-xs text-muted-foreground">
                  <summary className="w-fit cursor-pointer text-primary">Dataset saved · Show location</summary>
                  <p className="mt-1 break-all">{step.artifact_path}</p>
                </details>}
                {step.error && <p className="mt-1 text-xs text-destructive">{step.error}</p>}
              </div>
            </li>
          ))}
        </ol>
      )}
      {run.error && !run.step_statuses.some((step) => step.error) && <p className="mt-3 text-sm text-destructive">{run.error}</p>}
    </div>
  );
}

export function RoutinesPage() {
  const [routines, setRoutines] = useState<Routine[]>([]);
  const [recipes, setRecipes] = useState<RecipeRecord[]>([]);
  const [folders, setFolders] = useState<LinkedFolder[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [runs, setRuns] = useState<RoutineRun[]>([]);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState("");
  const [intervalMinutes, setIntervalMinutes] = useState(1440);
  const [steps, setSteps] = useState<RoutineStep[]>([]);
  const [stepKind, setStepKind] = useState<"recipe" | "folder_sync">("recipe");
  const [stepId, setStepId] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const signalReady = useAppShellReadySignal();
  useEffect(() => {
    if (loaded) signalReady();
  }, [loaded, signalReady]);

  const refresh = useCallback(async () => {
    try {
      const data = await request<{ routines: Routine[] }>("");
      setRoutines(data.routines);
      if (selectedId) {
        const history = await request<{ runs: RoutineRun[] }>(
          `/${encodeURIComponent(selectedId)}/runs`,
        );
        setRuns(history.runs);
      }
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load routines");
    } finally {
      setLoaded(true);
    }
  }, [selectedId]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    void listRecipes().then(setRecipes).catch(() => undefined);
    void listLinkedFolders().then(setFolders).catch(() => undefined);
  }, []);

  function beginCreate() {
    setSelectedId(null);
    setRuns([]);
    setName("");
    setIntervalMinutes(1440);
    setSteps([]);
    setEditing(true);
    setError("");
  }

  function selectRoutine(routine: Routine) {
    setSelectedId(routine.id);
    setName(routine.name);
    setIntervalMinutes(routine.interval_minutes);
    setSteps(routine.steps);
    setEditing(false);
    setRuns([]);
    void request<{ runs: RoutineRun[] }>(`/${encodeURIComponent(routine.id)}/runs`)
      .then((data) => setRuns(data.runs))
      .catch(() => undefined);
  }

  function addStep() {
    if (stepKind === "recipe") {
      const recipe = recipes.find((item) => item.id === stepId);
      if (!recipe) return;
      setSteps([...steps, { kind: "recipe", label: recipe.name, recipe: recipe.payload }]);
    } else {
      const folder = folders.find((item) => item.id === stepId);
      if (!folder) return;
      setSteps([...steps, { kind: "folder_sync", label: folder.displayName, folder_id: folder.id }]);
    }
    setStepId("");
  }

  async function save() {
    if (!name.trim() || steps.length === 0) {
      setError("Name the routine and add at least one step.");
      return;
    }
    setBusy(true);
    try {
      const saved = await request<Routine>(
        selectedId ? `/${encodeURIComponent(selectedId)}` : "",
        selectedId ? "PUT" : "POST",
        { name: name.trim(), steps, interval_minutes: intervalMinutes, enabled: selected?.enabled ?? true },
      );
      setSelectedId(saved.id);
      setEditing(false);
      setError("");
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save routine");
    } finally {
      setBusy(false);
    }
  }

  async function act(action: "run" | "enabled", routine: Routine) {
    setBusy(true);
    try {
      await request(
        `/${encodeURIComponent(routine.id)}/${action}`,
        action === "run" ? "POST" : "PATCH",
        action === "enabled" ? { enabled: !routine.enabled } : undefined,
      );
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Routine action failed");
    } finally {
      setBusy(false);
    }
  }

  const selected = routines.find((routine) => routine.id === selectedId);
  const choices = stepKind === "recipe" ? recipes : folders;

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-5 py-8 sm:px-8">
      {error && <p role="alert" className="mb-6 rounded-xl border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</p>}

      {editing ? (
        <div className="mx-auto max-w-3xl">
          <Button variant="ghost" className="mb-5 -ml-3 text-muted-foreground" onClick={() => setEditing(false)}>
            ← {selectedId ? "Back to routine" : "All routines"}
          </Button>
          <div className="mb-8 flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Studio automation</p>
              <h1 className="mt-2 text-3xl font-semibold tracking-tight">{selectedId ? "Edit routine" : "Create a routine"}</h1>
              <p className="mt-2 text-sm text-muted-foreground">Choose when it runs, then add the work in the order it should happen.</p>
            </div>
            <Button disabled={busy} onClick={() => void save()}>Save routine</Button>
          </div>
          <div className="space-y-7 rounded-2xl border border-border/70 bg-card p-5 sm:p-7">
            <section className="space-y-3">
              <div><h2 className="font-semibold">1. Name</h2><p className="text-sm text-muted-foreground">A short name you will recognize in run history.</p></div>
              <Input aria-label="Routine name" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} placeholder="Daily dataset refresh" />
            </section>
            <section className="space-y-3 border-t border-border/70 pt-6">
              <div><h2 className="font-semibold">2. Schedule</h2><p className="text-sm text-muted-foreground">Runs automatically while Studio is running. You can also run it any time.</p></div>
              <select aria-label="Schedule" className="h-10 w-full rounded-lg border border-border bg-background px-3 text-sm" value={intervalMinutes}
                onChange={(event) => setIntervalMinutes(Number(event.target.value))}>
                {INTERVALS.map(([minutes, label]) => <option key={minutes} value={minutes}>{label}</option>)}
              </select>
            </section>
            <section className="space-y-4 border-t border-border/70 pt-6">
              <div><h2 className="font-semibold">3. Steps</h2><p className="text-sm text-muted-foreground">Steps run from top to bottom. A failed step stops the routine.</p></div>
              <ol className="space-y-2">
                {steps.map((step, index) => (
                  <li key={`${step.kind}-${index}`} className="flex items-center gap-3 rounded-xl border border-border/70 bg-background p-3">
                    <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">{index + 1}</span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{step.label}</p>
                      <p className="text-xs text-muted-foreground">{step.kind === "recipe" ? "Data Recipe" : "Linked folder sync"}</p>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => setSteps(steps.filter((_, position) => position !== index))}>Remove</Button>
                  </li>
                ))}
              </ol>
              {steps.length === 0 && <p className="rounded-xl border border-dashed p-5 text-sm text-muted-foreground">No steps yet. Choose a saved recipe or linked folder below.</p>}
              <div className="rounded-xl bg-muted/50 p-4">
                <p className="mb-3 text-sm font-medium">Add a step</p>
                <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_auto]">
                  <label className="space-y-1 text-xs text-muted-foreground">Type
                    <select className="h-10 w-full rounded-lg border border-border bg-background px-3 text-sm text-foreground" value={stepKind}
                      onChange={(event) => { setStepKind(event.target.value as typeof stepKind); setStepId(""); }}>
                      <option value="recipe">Data Recipe</option>
                      <option value="folder_sync">Linked folder sync</option>
                    </select>
                  </label>
                  <label className="space-y-1 text-xs text-muted-foreground">Saved item
                    <select className="h-10 w-full rounded-lg border border-border bg-background px-3 text-sm text-foreground" value={stepId}
                      onChange={(event) => setStepId(event.target.value)}>
                      <option value="">Choose {stepKind === "recipe" ? "a recipe" : "a folder"}</option>
                      {choices.map((item) => <option key={item.id} value={item.id}>{"name" in item ? item.name : item.displayName}</option>)}
                    </select>
                  </label>
                  <Button className="self-end" variant="outline" disabled={!stepId} onClick={addStep}>Add step</Button>
                </div>
                {choices.length === 0 && <p className="mt-3 text-xs text-muted-foreground">Create a {stepKind === "recipe" ? "Data Recipe" : "linked folder"} first.</p>}
              </div>
            </section>
          </div>
        </div>
      ) : selected ? (
        <div>
          <Button variant="ghost" className="mb-5 -ml-3 text-muted-foreground" onClick={() => { setSelectedId(null); setRuns([]); }}>
            ← All routines
          </Button>
          <header className="flex flex-wrap items-start justify-between gap-5">
            <div>
              <div className="mb-3 flex items-center gap-2">
                <Badge variant="outline" className={selected.enabled ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300" : "text-muted-foreground"}>
                  {selected.enabled ? "Active" : "Paused"}
                </Badge>
                <span className="text-sm text-muted-foreground">{scheduleLabel(selected.interval_minutes)}</span>
              </div>
              <h1 className="text-3xl font-semibold tracking-tight">{selected.name}</h1>
              <p className="mt-2 text-sm text-muted-foreground">
                {selected.enabled ? `Next run ${when(selected.next_run_at)}` : "Automatic runs are paused"}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button disabled={busy} onClick={() => void act("run", selected)}>Run now</Button>
              <Button variant="outline" disabled={busy} onClick={() => void act("enabled", selected)}>{selected.enabled ? "Pause" : "Resume"}</Button>
              <Button variant="outline" onClick={() => setEditing(true)}>Edit</Button>
            </div>
          </header>
          <div className="mt-8 grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
            <section className="rounded-2xl border border-border/70 bg-card p-5 sm:p-6" aria-label="Workflow">
              <div className="flex items-baseline justify-between gap-3">
                <h2 className="text-lg font-semibold">What runs</h2>
                <span className="text-xs text-muted-foreground">{selected.steps.length} {selected.steps.length === 1 ? "step" : "steps"}</span>
              </div>
              <ol className="mt-5 space-y-4">
                {selected.steps.map((step, index) => (
                  <li key={index} className="flex gap-3">
                    <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">{index + 1}</span>
                    <div className="min-w-0 pt-0.5"><p className="break-words text-sm font-medium">{step.label}</p><p className="mt-0.5 text-xs text-muted-foreground">{step.kind === "recipe" ? "Data Recipe" : "Linked folder sync"}</p></div>
                  </li>
                ))}
              </ol>
            </section>
            <section className="rounded-2xl border border-border/70 bg-card p-5 sm:p-6" aria-label="Latest run">
              <h2 className="text-lg font-semibold">Latest run</h2>
              {runs[0] ? <div className="mt-5"><RunResult run={runs[0]} /></div> : (
                <div className="mt-5 rounded-xl bg-muted/50 p-5 text-sm text-muted-foreground">This routine has not run yet. Use Run now to try it.</div>
              )}
            </section>
          </div>
          {runs.length > 1 && (
            <details className="mt-6 rounded-2xl border border-border/70 bg-card p-5 sm:p-6">
              <summary className="cursor-pointer text-sm font-semibold">Earlier runs ({runs.length - 1})</summary>
              <div className="mt-5 space-y-4 border-t border-border/70 pt-5">
                {runs.slice(1).map((run) => <div key={run.id} className="rounded-xl border border-border/70 p-4"><RunResult run={run} /></div>)}
              </div>
            </details>
          )}
        </div>
      ) : (
        <div>
          <header className="flex flex-wrap items-start justify-between gap-5">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Studio automation</p>
              <h1 className="mt-2 text-3xl font-semibold tracking-tight">Routines</h1>
              <p className="mt-2 max-w-xl text-sm text-muted-foreground">Schedule saved work and see the result in one place. Routines run while Studio is running.</p>
            </div>
            <Button onClick={beginCreate}>Create routine</Button>
          </header>
          {routines.length === 0 ? (
            <div className="mt-9 rounded-2xl border border-dashed border-border bg-card px-6 py-12 text-center">
              <h2 className="text-lg font-semibold">Your first routine starts here</h2>
              <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">Choose a saved Data Recipe or linked folder, then set how often Studio should run it.</p>
              <div className="mt-5 flex justify-center gap-3"><Button onClick={beginCreate}>Create routine</Button><Button variant="outline" asChild><Link to="/data-recipes">Browse recipes</Link></Button></div>
            </div>
          ) : (
            <section className="mt-9" aria-label="Saved routines">
              <h2 className="mb-4 text-sm font-semibold text-muted-foreground">Your routines</h2>
              <div className="space-y-3">
                {routines.map((routine) => (
                  <button key={routine.id} type="button" onClick={() => selectRoutine(routine)}
                    className="group w-full rounded-2xl border border-border/70 bg-card p-5 text-left transition-colors hover:border-primary/40 hover:bg-muted/20 focus-visible:outline-2 focus-visible:outline-primary">
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="text-base font-semibold">{routine.name}</h3>
                          <Badge variant="outline" className={routine.enabled ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300" : "text-muted-foreground"}>{routine.enabled ? "Active" : "Paused"}</Badge>
                        </div>
                        <p className="mt-2 text-sm text-muted-foreground">{scheduleLabel(routine.interval_minutes)} · {routine.steps.length} {routine.steps.length === 1 ? "step" : "steps"}</p>
                        <p className="mt-1 text-xs text-muted-foreground">{routine.enabled ? `Next run ${when(routine.next_run_at)}` : "Automatic runs paused"}</p>
                      </div>
                      <span className="text-lg text-muted-foreground transition-transform group-hover:translate-x-1" aria-hidden="true">→</span>
                    </div>
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </main>
  );
}
