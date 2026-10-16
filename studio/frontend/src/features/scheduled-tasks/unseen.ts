// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Finished runs the user has not looked at yet: a dot on the sidebar's Scheduled row until the
 *  page is opened, and one on each task until its chat is opened (or, for a failed run, the task
 *  is touched). In the startup bundle, beside the results poll that fills it. */

import { create } from "zustand";
import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
  getAuthToken,
  sessionAccount,
} from "@/features/auth";
import { UNSEEN_KEY_PREFIX } from "./results";

/** The signed-in account's record, or none while signed out. Each account keeps its own, so
 *  signing out or into another account does not count as having looked. */
function storageKey(): string | null {
  const token = getAuthToken();
  return token ? UNSEEN_KEY_PREFIX + (sessionAccount(token)?.username ?? "") : null;
}

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

const NOTHING_UNSEEN: Stored = { tasks: {}, page: false };

/** The account's stored record, or null when there is none to read: signed out, or no storage. */
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
  // Each change starts from the stored record, not this tab's copy, which another tab of the
  // account may have changed since: written back, a stale copy would bring back markers that tab
  // cleared, or drop ones it added. Without storage the copy is all there is.
  const change = (apply: (current: Stored) => Stored | null) => {
    const current = stored() ?? { tasks: get().tasks, page: get().page };
    const next = apply(current) ?? current;
    set(next);
    if (next !== current) save(next);
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
  // Signing out hides the markers but keeps them for the account; signing in shows its own.
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    useUnseenScheduledRuns.setState({ tasks: {}, page: false });
  });
  window.addEventListener(AUTH_SESSION_STORED_EVENT, () => {
    useUnseenScheduledRuns.setState(load());
  });
  // Another tab of the account marked or cleared something: show it here too.
  window.addEventListener("storage", (event) => {
    if (event.key !== null && event.key === storageKey()) {
      useUnseenScheduledRuns.setState(load());
    }
  });
}
