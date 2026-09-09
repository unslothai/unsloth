// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type ProjectTask = {
  id: string;
  projectId: string;
  parentId: string | null;
  rootId: string;
  retryOf: string | null;
  attempt: number;
  role: "root" | "reviewer" | "implementer";
  instruction: string;
  status: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled" | "interrupted";
  runtime: { kind: "local" | "provider"; model: string; providerId: string | null };
  result: { output?: string } | null;
  resultTruncated?: boolean;
  commandsEnabled?: boolean;
  error: string | null;
  worktreeId: string | null;
  maxOutputTokens: number;
  childBudget: number;
  childAllocated: number;
};

export type TaskSelection = { kind: "local" | "provider"; model: string; providerId?: string };
export type TaskReview = { worktreeId: string; diff: string; truncated: boolean; newFiles: { path: string; diff: string; unavailable: boolean }[] };

export function taskCanCancel(task: ProjectTask): boolean {
  return task.status === "queued" || task.status === "running";
}

export function taskCanRetry(task: ProjectTask, tasks: ProjectTask[]): boolean {
  return !task.parentId && task.attempt < 3 &&
    ["failed", "cancelled", "interrupted"].includes(task.status) &&
    !tasks.some((candidate) => candidate.retryOf === task.id);
}

async function request<T>(projectId: string, suffix = "", method = "GET", body?: unknown, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  if (signal?.aborted) controller.abort();
  const timer = setTimeout(abort, 15_000);
  try {
    const response = await authFetch(`/api/agent/projects/${encodeURIComponent(projectId)}/tasks${suffix}`, {
      method, signal: controller.signal, cache: "no-store",
      ...(body === undefined ? {} : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    }, { retryNetworkErrors: method === "GET" });
    if (!response.ok) throw new Error(await readFastApiError(response, "Project tasks are unavailable."));
    return await response.json() as T;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}

export const listProjectTasks = (projectId: string, signal?: AbortSignal) => request<ProjectTask[]>(projectId, "?limit=100", "GET", undefined, signal);
export const getProjectTask = (projectId: string, taskId: string, signal?: AbortSignal) => request<ProjectTask>(projectId, `/${encodeURIComponent(taskId)}`, "GET", undefined, signal);
export const getProjectTaskReview = (projectId: string, taskId: string, signal?: AbortSignal) => request<TaskReview>(projectId, `/${encodeURIComponent(taskId)}/review`, "GET", undefined, signal);
export const submitProjectTask = (projectId: string, instruction: string, selection: TaskSelection, signal?: AbortSignal, allowCommands = false) =>
  request<ProjectTask>(projectId, "", "POST", { instruction, ...selection, maxOutputTokens: 8192, childLimit: 2, childBudget: 8192, timeout: 900, allowCommands }, signal);
export const cancelProjectTask = (projectId: string, taskId: string, signal?: AbortSignal) => request<ProjectTask>(projectId, `/${encodeURIComponent(taskId)}/cancel`, "POST", undefined, signal);
export const retryProjectTask = (projectId: string, taskId: string, signal?: AbortSignal) => request<ProjectTask>(projectId, `/${encodeURIComponent(taskId)}/retry`, "POST", undefined, signal);


export type TaskCommand = {
  id: string; taskId: string; sequence: number; argv: string[]; timeout: number;
  status: "running" | "passed" | "failed" | "cancelled" | "timed_out" | "unavailable" | "interrupted" | "containment_pending";
  exitCode: number | null; output: string; outputBytes: number;
  outputTruncated: boolean; previewTruncated: boolean; createdAt: number; completedAt: number | null;
};
export const getTaskCapabilities = (projectId: string, signal?: AbortSignal) =>
  request<{ commands: { available: boolean; reason: string | null } }>(projectId, "/capabilities", "GET", undefined, signal);
export const listTaskCommands = (projectId: string, taskId: string, signal?: AbortSignal) =>
  request<TaskCommand[]>(projectId, `/${encodeURIComponent(taskId)}/commands`, "GET", undefined, signal);
export const getTaskCommand = (projectId: string, taskId: string, commandId: string, signal?: AbortSignal) =>
  request<TaskCommand>(projectId, `/${encodeURIComponent(taskId)}/commands/${encodeURIComponent(commandId)}`, "GET", undefined, signal);
export function commandStatusLabel(status: TaskCommand["status"]): string {
  return { running: "Running", passed: "Passed", failed: "Failed", cancelled: "Cancelled", timed_out: "Timed out", unavailable: "Unavailable", interrupted: "Outcome unconfirmed", containment_pending: "Cleanup unconfirmed" }[status];
}
