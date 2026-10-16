// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Finished runs the user has not looked at yet: a dot on the sidebar's Scheduled row until the
 *  page is opened, and one on each task until its chat is opened (or, for a failed run, the task
 *  is touched). In the startup bundle, beside the results poll that fills it. */

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";

const STORAGE_KEY = "unsloth_scheduled_tasks_unseen";

interface Stored {
  /** Task id -> the chat its finished run made, or null for a failed run. */
  tasks: Record<string, string | null>;
  page: boolean;
}

interface UnseenState extends Stored {
  markFinished: (runs: { taskId: string; threadId: string | null }[], onPage: boolean) => void;
  clearTask: (taskId: string) => void;
  /** The chat a run made was opened, from anywhere. */
  clearThread: (threadId: string) => void;
  clearPage: () => void;
}

function load(): Stored {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as Partial<Stored> | null;
    if (value && typeof value.tasks === "object" && value.tasks !== null) {
      return { tasks: value.tasks, page: value.page === true };
    }
  } catch {
    // Unreadable storage starts with nothing unseen.
  }
  return { tasks: {}, page: false };
}

function save({ tasks, page }: Stored): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ tasks, page }));
  } catch {
    // Without storage the dots last until the next launch.
  }
}

export const useUnseenScheduledRuns = create<UnseenState>()((set, get) => {
  const update = (next: Stored) => {
    set(next);
    save(next);
  };
  return {
    ...(typeof window === "undefined" ? { tasks: {}, page: false } : load()),
    markFinished: (runs, onPage) => {
      const tasks = { ...get().tasks };
      for (const run of runs) tasks[run.taskId] = run.threadId;
      update({ tasks, page: get().page || !onPage });
    },
    clearTask: (taskId) => {
      if (!(taskId in get().tasks)) return;
      const tasks = { ...get().tasks };
      delete tasks[taskId];
      update({ tasks, page: get().page });
    },
    clearThread: (threadId) => {
      const taskId = Object.keys(get().tasks).find((id) => get().tasks[id] === threadId);
      if (taskId !== undefined) get().clearTask(taskId);
    },
    clearPage: () => {
      if (get().page) update({ tasks: get().tasks, page: false });
    },
  };
});

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    useUnseenScheduledRuns.setState({ tasks: {}, page: false });
    save({ tasks: {}, page: false });
  });
}
