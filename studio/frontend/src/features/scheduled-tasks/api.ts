// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthSessionEpoch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { TaskSchedule } from "./schedule";

export const SCHEDULED_TASKS_API = "/api/scheduled-tasks";

export type TaskRunStatus = "queued" | "running" | "succeeded" | "failed";

/** Which finished runs a task announces with a toast and a system notification. Every run still
 *  brings its chat into Recents, marked unread. */
export type TaskNotify = "all" | "failures" | "none";

export interface ScheduledTaskRun {
  id: string;
  taskId: string;
  source: "schedule" | "manual";
  status: TaskRunStatus;
  scheduledFor: number | null;
  createdAt: number;
  startedAt: number | null;
  finishedAt: number | null;
  /** The chat the answer went to; null after that chat is deleted. */
  threadId: string | null;
  model: string | null;
  /** A reason the UI words itself (see `runErrorKey`); otherwise `error` is the backend's text. */
  errorCode: string | null;
  error: string | null;
}

export interface FinishedScheduledTaskRun extends ScheduledTaskRun {
  taskTitle: string;
  taskNotify: TaskNotify;
}

export interface ScheduledTask {
  id: string;
  title: string;
  prompt: string;
  schedule: TaskSchedule;
  timezone: string;
  enabled: boolean;
  notify: TaskNotify;
  /** False when the title was taken from the instructions rather than typed. */
  named: boolean;
  status: "active" | "paused" | "completed";
  nextRunAt: number | null;
  createdAt: number;
  updatedAt: number;
  lastRun: ScheduledTaskRun | null;
}

export interface ScheduledTaskInput {
  title: string;
  prompt: string;
  schedule: TaskSchedule;
  timezone: string;
  notify: TaskNotify;
  named: boolean;
}

export type ScheduledTaskPatch = Partial<ScheduledTaskInput> & { enabled?: boolean };

export interface ScheduledTaskActivity {
  now: number;
  runs: FinishedScheduledTaskRun[];
  /** More runs finished after these: ask again from the last one's `finishedAt`. */
  more: boolean;
  activeRuns: number;
  nextRunAt: number | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const epoch = getAuthSessionEpoch();
  const res = await authFetch(
    `${SCHEDULED_TASKS_API}${path}`,
    {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    },
    {
      // A write retried after a refresh must not land in an account that signed in meanwhile.
      beforeRetry: () => {
        if (getAuthSessionEpoch() !== epoch) {
          throw new Error("Signed out before the scheduled task was saved.");
        }
      },
    },
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
  return (res.status === 204 ? undefined : await res.json()) as T;
}

function taskPath(id: string, suffix = ""): string {
  return `/${encodeURIComponent(id)}${suffix}`;
}

export interface ScheduledTaskList {
  tasks: ScheduledTask[];
  /** The server's clock, which `nextRunAt` and the run times are on. */
  now: number;
}

export function listScheduledTasks(): Promise<ScheduledTaskList> {
  return request("");
}

export function createScheduledTask(input: ScheduledTaskInput): Promise<ScheduledTask> {
  return request("", { method: "POST", body: JSON.stringify(input) });
}

export function updateScheduledTask(
  id: string,
  patch: ScheduledTaskPatch,
): Promise<ScheduledTask> {
  return request(taskPath(id), { method: "PATCH", body: JSON.stringify(patch) });
}

export function deleteScheduledTask(id: string): Promise<void> {
  return request(taskPath(id), { method: "DELETE" });
}

export function runScheduledTaskNow(id: string): Promise<ScheduledTaskRun> {
  return request(taskPath(id, "/run"), { method: "POST" });
}

/** Runs finished at or after `since` (server ms), plus what the results poll paces itself by. */
export function getScheduledTaskActivity(since: number | null): Promise<ScheduledTaskActivity> {
  return request(since === null ? "/runs" : `/runs?since=${since}`);
}
