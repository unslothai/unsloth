// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
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
    <main className="mx-auto w-full max-w-6xl flex-1 overflow-y-auto px-6 py-8">
      <div className="mb-7 flex items-start justify-between gap-4">
        <div>
          <h1 className="font-heading text-3xl font-semibold">Routines</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Run Studio work on a schedule, or start it yourself. Scheduled work runs while Studio is running.
          </p>
        </div>
        <Button onClick={beginCreate}>New routine</Button>
      </div>
      {error && <p role="alert" className="mb-4 rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</p>}
      <div className="grid gap-6 lg:grid-cols-[minmax(260px,1fr)_minmax(0,2fr)]">
        <section aria-label="Saved routines" className="space-y-3">
          {routines.length === 0 && <div className="rounded-xl border p-6 text-sm text-muted-foreground">No routines yet. Create one to automate a Data Recipe or linked folder sync.</div>}
          {routines.map((routine) => (
            <button key={routine.id} type="button" onClick={() => selectRoutine(routine)}
              className={`w-full rounded-xl border p-4 text-left transition-colors hover:bg-muted/50 ${selectedId === routine.id ? "border-primary bg-muted/40" : ""}`}>
              <span className="font-medium">{routine.name}</span>
              <span className="ml-2 text-xs text-muted-foreground">{routine.enabled ? "Active" : "Paused"}</span>
              <span className="mt-2 block text-xs text-muted-foreground">
                {routine.steps.length} {routine.steps.length === 1 ? "step" : "steps"} · {routine.enabled ? `Next ${when(routine.next_run_at)}` : "Paused"}
              </span>
            </button>
          ))}
        </section>
        <section className="rounded-xl border p-5">
          {editing ? (
            <div className="space-y-5">
              <h2 className="text-xl font-semibold">{selectedId ? "Edit routine" : "Create routine"}</h2>
              <label className="block space-y-2 text-sm font-medium">Name
                <Input value={name} maxLength={120} onChange={(event) => setName(event.target.value)} placeholder="Weekly dataset refresh" />
              </label>
              <label className="block space-y-2 text-sm font-medium">Schedule
                <select className="h-10 w-full rounded-md border bg-background px-3" value={intervalMinutes}
                  onChange={(event) => setIntervalMinutes(Number(event.target.value))}>
                  {INTERVALS.map(([minutes, label]) => <option key={minutes} value={minutes}>{label}</option>)}
                </select>
              </label>
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Steps run in this order</h3>
                {steps.map((step, index) => (
                  <div key={`${step.kind}-${index}`} className="flex items-center justify-between rounded-lg border px-3 py-2 text-sm">
                    <span>{index + 1}. {step.label} <span className="text-muted-foreground">({step.kind === "recipe" ? "Data Recipe" : "Folder sync"})</span></span>
                    <Button variant="ghost" size="sm" onClick={() => setSteps(steps.filter((_, position) => position !== index))}>Remove</Button>
                  </div>
                ))}
                <div className="flex flex-wrap gap-2">
                  <select aria-label="Step type" className="h-10 rounded-md border bg-background px-3" value={stepKind}
                    onChange={(event) => { setStepKind(event.target.value as typeof stepKind); setStepId(""); }}>
                    <option value="recipe">Data Recipe</option><option value="folder_sync">Linked folder sync</option>
                  </select>
                  <select aria-label="Step source" className="h-10 min-w-44 flex-1 rounded-md border bg-background px-3" value={stepId}
                    onChange={(event) => setStepId(event.target.value)}>
                    <option value="">Choose {stepKind === "recipe" ? "a recipe" : "a folder"}</option>
                    {choices.map((item) => <option key={item.id} value={item.id}>{"name" in item ? item.name : item.displayName}</option>)}
                  </select>
                  <Button variant="outline" disabled={!stepId} onClick={addStep}>Add step</Button>
                </div>
                {choices.length === 0 && <p className="text-xs text-muted-foreground">Create a {stepKind === "recipe" ? "Data Recipe" : "linked folder"} first.</p>}
              </div>
              <div className="flex gap-2">
                <Button disabled={busy} onClick={() => void save()}>Save routine</Button>
                <Button variant="outline" onClick={() => setEditing(false)}>Cancel</Button>
              </div>
            </div>
          ) : selected ? (
            <div className="space-y-6">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div><h2 className="text-xl font-semibold">{selected.name}</h2><p className="text-sm text-muted-foreground">Next run {selected.enabled ? when(selected.next_run_at) : "paused"}</p></div>
                <div className="flex flex-wrap gap-2">
                  <Button disabled={busy} onClick={() => void act("run", selected)}>Run now</Button>
                  <Button variant="outline" disabled={busy} onClick={() => void act("enabled", selected)}>{selected.enabled ? "Pause" : "Resume"}</Button>
                  <Button variant="outline" onClick={() => setEditing(true)}>Edit</Button>
                </div>
              </div>
              <div><h3 className="mb-2 text-sm font-semibold">Workflow</h3>{selected.steps.map((step, index) => <p key={index} className="border-l-2 border-primary/30 py-1 pl-3 text-sm">{index + 1}. {step.label}</p>)}</div>
              <div><h3 className="mb-3 text-sm font-semibold">Run history</h3>
                {runs.length === 0 && <p className="text-sm text-muted-foreground">No runs yet.</p>}
                <div className="space-y-3">{runs.map((run) => <div key={run.id} className="rounded-lg border p-3 text-sm">
                  <div className="flex justify-between gap-3"><span className="font-medium">{run.status}</span><span className="text-muted-foreground">{when(run.created_at)} · {run.trigger}</span></div>
                  {run.step_statuses.map((step, index) => <div key={index} className="mt-2 text-muted-foreground">
                    <p>{step.label}: {step.status}</p>
                    {step.summary && <p className="text-xs">{Object.entries(step.summary).map(([key, count]) => `${key}: ${count}`).join(" · ")}</p>}
                    {step.artifact_path && <p className="break-all text-xs">Output: {step.artifact_path}</p>}
                    {step.error && <p className="text-xs text-destructive">{step.error}</p>}
                  </div>)}
                  {run.error && <p className="mt-2 text-destructive">{run.error}</p>}
                </div>)}</div>
              </div>
            </div>
          ) : <div className="space-y-3 text-sm text-muted-foreground"><p>Select a routine to view its runs, or create one.</p><Link className="text-primary underline" to="/data-recipes">Browse Data Recipes</Link></div>}
        </section>
      </div>
    </main>
  );
}
