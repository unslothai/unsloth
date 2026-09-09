// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  cancelProjectTask, getProjectTask, getProjectTaskReview, listProjectTasks, retryProjectTask, submitProjectTask,
  taskCanCancel, taskCanRetry, type ProjectTask, type TaskSelection, type TaskReview,
} from "../api/project-tasks-api";

function TaskChanges({ projectId, taskId }: { projectId: string; taskId: string }) {
  const [open, setOpen] = useState(false);
  const [review, setReview] = useState<TaskReview | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setReview(null);
    setError(null);
    void getProjectTaskReview(projectId, taskId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setReview(value);
    }).catch((cause) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Could not review changes.");
    });
    return () => controller.abort();
  }, [open, projectId, taskId]);
  return <details onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary className="cursor-pointer text-sm">Review worktree changes</summary>
    {open && !review && !error ? <p role="status" className="text-sm">Loading changes…</p> : null}
    {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
    {review ? <div className="mt-2 space-y-2">
      <p className="text-xs text-muted-foreground">Changes since the task’s starting commit. Inspect the worktree location in Git &amp; worktrees before committing or merging.</p>
      {review.truncated ? <p className="text-sm">This preview is incomplete. Inspect the remaining changes in the checkout.</p> : null}
      <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words text-xs">{review.diff || "No tracked changes."}</pre>
      {review.newFiles.map((file) => <div key={file.path}><p className="break-all text-sm">New file: {file.path}</p><pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs">{file.unavailable ? "Preview unavailable for this file." : file.diff}</pre></div>)}
    </div> : null}
  </details>;
}

function TaskResult({ projectId, task }: { projectId: string; task: ProjectTask }) {
  const [open, setOpen] = useState(false);
  const [output, setOutput] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!open || !task.resultTruncated) return;
    const controller = new AbortController();
    void getProjectTask(projectId, task.id, controller.signal).then((full) => {
      if (!controller.signal.aborted) setOutput(full.result?.output ?? "");
    }).catch((cause) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Could not load the full result.");
    });
    return () => controller.abort();
  }, [open, projectId, task.id, task.resultTruncated]);
  return <details onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary className="cursor-pointer text-sm">Task result</summary>
    {error ? <p role="alert">{error}</p> : null}
    {open ? <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-words text-sm">{output ?? task.result?.output}{task.resultTruncated && output === null ? "\nLoading full result…" : ""}</pre> : null}
  </details>;
}

export function ProjectTasksPanel({ projectId, selection }: { projectId: string; selection: TaskSelection }) {
  const [instruction, setInstruction] = useState("");
  const [tasks, setTasks] = useState<ProjectTask[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [pending, setPending] = useState(false);
  const lifetime = useRef<AbortController | null>(null);
  const requestVersion = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    lifetime.current = controller;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function refresh() {
      const version = ++requestVersion.current;
      try {
        const rows = await listProjectTasks(projectId, controller.signal);
        if (!controller.signal.aborted && version === requestVersion.current) {
          setTasks(rows);
          setLoaded(true);
          setError(null);
        }
      } catch (cause) {
        if (!controller.signal.aborted && version === requestVersion.current) {
          setError(cause instanceof Error ? cause.message : "Could not refresh tasks.");
          setLoaded(false);
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(() => void refresh(), 2000);
      }
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [projectId]);

  async function mutate(action: (signal: AbortSignal) => Promise<ProjectTask>, clearDraft = false) {
    const controller = lifetime.current;
    if (!controller || controller.signal.aborted || pending) return;
    setPending(true);
    ++requestVersion.current;
    try {
      await action(controller.signal);
      if (controller.signal.aborted) return;
      if (clearDraft) setInstruction("");
      const version = ++requestVersion.current;
      const rows = await listProjectTasks(projectId, controller.signal);
      if (!controller.signal.aborted && version === requestVersion.current) {
        setTasks(rows);
        setError(null);
      }
    } catch (cause) {
      if (!controller.signal.aborted) setError(`${cause instanceof Error ? cause.message : "Task request failed."} Refresh the task list before submitting again; the server may have accepted the request.`);
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }

  return (
    <section className="mt-8 space-y-5" aria-label="Project tasks">
      <div className="space-y-3 rounded-2xl border p-5">
        <h2 className="font-semibold">Run a project task</h2>
        <p className="text-sm text-muted-foreground">The coordinator can read this project and delegate up to two reviewers or implementers. Each attempt starts from the project’s current commit. Implementers can edit their own worktrees. Changes stay there for your review. Commands and automatic merges are unavailable.</p>
        <p className="text-sm">Model: {selection.model || "Select a model in Chat first"}</p>
        <p className="text-xs text-muted-foreground">15 minute deadline · 8,192 coordinator output tokens · 8,192 shared child output tokens. Uses a saved tool-capable provider or a loaded GGUF model. Subscription and safetensors runtimes are not supported yet.</p>
        <Textarea aria-label="Task instruction" placeholder="Describe the change or review, including relevant file paths…" value={instruction} maxLength={16000} onChange={(e) => setInstruction(e.target.value)} disabled={pending} />
        <Button disabled={pending || !loaded || !instruction.trim() || !selection.model} onClick={() => void mutate((signal) => submitProjectTask(projectId, instruction.trim(), selection, signal), true)}>Start task</Button>
      </div>
      {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
      {!loaded && !error ? <p role="status">Loading tasks…</p> : null}
      {loaded && tasks.length === 0 ? <p className="text-sm text-muted-foreground">No project tasks yet.</p> : null}
      {tasks.length === 100 ? <p className="text-xs text-muted-foreground">Showing the 100 most recent attempts.</p> : null}
      {tasks.map((task) => (
        <article key={task.id} className="space-y-2 rounded-2xl border p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold">{task.role} · {task.status} · attempt {task.attempt}</h3>
            <div className="flex gap-2">
              {taskCanCancel(task) ? <Button size="sm" variant="outline" disabled={pending} onClick={() => void mutate((signal) => cancelProjectTask(projectId, task.id, signal))}>Cancel</Button> : null}
              {taskCanRetry(task, tasks) ? <Button size="sm" variant="outline" disabled={pending} onClick={() => void mutate((signal) => retryProjectTask(projectId, task.id, signal))}>Retry</Button> : null}
            </div>
          </div>
          <p className="whitespace-pre-wrap break-words text-sm">{task.instruction}</p>
          <p className="break-all text-xs text-muted-foreground">Task {task.id}{task.parentId ? ` · Child of ${task.parentId}` : ""}{task.retryOf ? ` · Retry of ${task.retryOf}` : ""}</p>
          <p className="text-xs text-muted-foreground">{task.runtime.model} · {task.maxOutputTokens.toLocaleString()} output tokens{task.role === "root" ? ` · Children reserved ${task.childAllocated.toLocaleString()} / ${task.childBudget.toLocaleString()}` : ""}</p>
          {task.worktreeId ? <p className="break-all text-xs">Owned worktree: {task.worktreeId}. Inspect it in Git &amp; worktrees before committing or merging.</p> : null}
          {task.worktreeId && ["completed", "failed", "cancelled", "interrupted"].includes(task.status) ? <TaskChanges projectId={projectId} taskId={task.id} /> : null}
          {task.error ? <p className="text-sm text-destructive">{task.error}</p> : null}
          {task.result?.output ? <TaskResult projectId={projectId} task={task} /> : null}
        </article>
      ))}
    </section>
  );
}
