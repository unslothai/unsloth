// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT, getAuthSessionEpoch } from "@/features/auth";
import { notifyChatHistoryUpdated } from "@/features/chat";
import {
  type ScheduledTask,
  type ScheduledTaskInput,
  type ScheduledTaskPatch,
  createScheduledTask,
  deleteScheduledTask,
  listScheduledTasks,
  runScheduledTaskNow,
  updateScheduledTask,
} from "./api";
import { SCHEDULED_TASKS_CHANGED_EVENT, SCHEDULED_TASK_RUNS_FINISHED_EVENT } from "./results";

/** Bumped on every change so other tabs get a storage event. */
export const SCHEDULED_TASKS_WRITTEN_KEY = "unsloth_scheduled_tasks_written";

function announceChange(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(SCHEDULED_TASKS_CHANGED_EVENT));
  try {
    localStorage.setItem(SCHEDULED_TASKS_WRITTEN_KEY, `${Date.now()}.${Math.random()}`);
  } catch {
    // Without storage, another tab shows the change when it next refreshes.
  }
}

interface ScheduledTasksState {
  tasks: ScheduledTask[];
  /** The server's clock less this browser's, as of the last refresh. */
  serverOffset: number;
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  refresh: () => Promise<void>;
  create: (input: ScheduledTaskInput) => Promise<ScheduledTask>;
  update: (id: string, patch: ScheduledTaskPatch) => Promise<ScheduledTask>;
  remove: (id: string) => Promise<void>;
  runNow: (id: string) => Promise<void>;
}

let refreshGeneration = 0;

function withTask(tasks: ScheduledTask[], task: ScheduledTask): ScheduledTask[] {
  return tasks.some((entry) => entry.id === task.id)
    ? tasks.map((entry) => (entry.id === task.id ? task : entry))
    : [task, ...tasks];
}

export const useScheduledTasksStore = create<ScheduledTasksState>()((set, get) => {
  /** Applies a write's answer unless the account changed; drops a refresh that started before the write. */
  async function write<T>(request: () => Promise<T>, apply: (result: T) => void): Promise<T> {
    const epoch = getAuthSessionEpoch();
    const result = await request();
    if (getAuthSessionEpoch() === epoch) {
      refreshGeneration += 1;
      apply(result);
    }
    announceChange();
    return result;
  }

  return {
    tasks: [],
    serverOffset: 0,
    status: "idle",
    error: null,
    refresh: async () => {
      const generation = ++refreshGeneration;
      if (get().status === "idle") set({ status: "loading" });
      try {
        const list = await listScheduledTasks();
        if (generation !== refreshGeneration) return;
        set({
          tasks: list.tasks,
          serverOffset: list.now - Date.now(),
          status: "ready",
          error: null,
        });
      } catch (error) {
        if (generation !== refreshGeneration) return;
        set({
          status: get().status === "ready" ? "ready" : "error",
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },
    create: (input) =>
      write(
        () => createScheduledTask(input),
        (task) => set((state) => ({ tasks: withTask(state.tasks, task) })),
      ),
    update: (id, patch) =>
      write(
        () => updateScheduledTask(id, patch),
        (task) => set((state) => ({ tasks: withTask(state.tasks, task) })),
      ),
    remove: async (id) => {
      await write(
        () => deleteScheduledTask(id),
        () => set((state) => ({ tasks: state.tasks.filter((task) => task.id !== id) })),
      );
      // Its chat survives but the run record went with the task, so the results poll never reports it.
      notifyChatHistoryUpdated();
    },
    runNow: async (id) => {
      await write(
        () => runScheduledTaskNow(id),
        (run) =>
          set((state) => ({
            // A refresh may already hold this run finished: the queued answer must not undo that.
            tasks: state.tasks.map((task) =>
              task.id === id && task.lastRun?.id !== run.id ? { ...task, lastRun: run } : task,
            ),
          })),
      );
    },
  };
});

if (typeof window !== "undefined") {
  window.addEventListener(SCHEDULED_TASK_RUNS_FINISHED_EVENT, () => {
    const { status, refresh } = useScheduledTasksStore.getState();
    if (status !== "idle") void refresh();
  });
  // Module state: a sign-out must drop it or the next account sees these tasks.
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    refreshGeneration += 1;
    useScheduledTasksStore.setState({ tasks: [], serverOffset: 0, status: "idle", error: null });
  });
}
