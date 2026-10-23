// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Unseen finished runs: dots on the Scheduled sidebar row and on each task until opened. */

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT, AUTH_SESSION_STORED_EVENT, getAuthToken } from "@/features/auth";
import { browserAccountKey } from "@/lib/account-transition";
import { UNSEEN_KEY_PREFIX } from "./results";

/** The signed-in account's record (by id), or none while signed out. */
function storageKey(): string | null {
  return getAuthToken() ? UNSEEN_KEY_PREFIX + browserAccountKey() : null;
}

interface Stored {
  tasks: Record<string, string | null>;
  page: boolean;
}

interface UnseenState extends Stored {
  markFinished: (runs: { taskId: string; threadId: string | null }[], onPage: boolean) => void;
  clearTask: (taskId: string) => void;
  clearThread: (threadId: string) => void;
  clearPage: () => void;
}

const NOTHING_UNSEEN: Stored = { tasks: {}, page: false };

function stored(): Stored | null {
  const key = storageKey();
  if (key === null) return null;
  try {
    const value = JSON.parse(localStorage.getItem(key) ?? "null") as Partial<Stored> | null;
    if (value && typeof value.tasks === "object" && value.tasks !== null) {
      return { tasks: value.tasks, page: value.page === true };
    }
    return NOTHING_UNSEEN;
  } catch {
    return null;
  }
}

function load(): Stored {
  return stored() ?? NOTHING_UNSEEN;
}

function save({ tasks, page }: Stored): void {
  const key = storageKey();
  if (key === null) return;
  try {
    localStorage.setItem(key, JSON.stringify({ tasks, page }));
  } catch {
    // Without storage the dots last until the next launch.
  }
}

export const useUnseenScheduledRuns = create<UnseenState>()((set, get) => {
  // Start each change from the stored record: another tab may have changed it, and a stale copy would undo that.
  const change = (apply: (current: Stored) => Stored | null) => {
    const current = stored() ?? { tasks: get().tasks, page: get().page };
    const next = apply(current);
    if (next === null) return;
    set(next);
    save(next);
  };
  const withoutTask = (current: Stored, taskId: string | undefined): Stored | null => {
    if (taskId === undefined || !(taskId in current.tasks)) return null;
    const tasks = { ...current.tasks };
    delete tasks[taskId];
    return { tasks, page: current.page };
  };
  return {
    ...(typeof window === "undefined" ? NOTHING_UNSEEN : load()),
    markFinished: (runs, onPage) =>
      change((current) => {
        const tasks = { ...current.tasks };
        for (const run of runs) tasks[run.taskId] = run.threadId;
        return { tasks, page: current.page || !onPage };
      }),
    clearTask: (taskId) => change((current) => withoutTask(current, taskId)),
    clearThread: (threadId) =>
      change((current) =>
        withoutTask(
          current,
          Object.keys(current.tasks).find((id) => current.tasks[id] === threadId),
        ),
      ),
    clearPage: () =>
      change((current) => (current.page ? { tasks: current.tasks, page: false } : null)),
  };
});

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    useUnseenScheduledRuns.setState({ tasks: {}, page: false });
  });
  window.addEventListener(AUTH_SESSION_STORED_EVENT, () => {
    useUnseenScheduledRuns.setState(load());
  });
  window.addEventListener("storage", (event) => {
    if (event.key?.startsWith(UNSEEN_KEY_PREFIX) && event.key === storageKey()) {
      useUnseenScheduledRuns.setState(load());
    }
  });
}
