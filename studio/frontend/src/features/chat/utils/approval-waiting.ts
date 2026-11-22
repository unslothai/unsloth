// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { create } from "zustand";

// An "Always allow" call still parks until its card answers it in one request; only a call
// parked longer than this is waiting on the user.
export const APPROVAL_SETTLE_MS = 1_500;

type PendingConfirmations = Readonly<Record<string, { threadId: string }>>;

type ConfirmationSource = {
  getState: () => { toolConfirmations: PendingConfirmations };
  subscribe: (
    listener: (
      state: { toolConfirmations: PendingConfirmations },
      previous: { toolConfirmations: PendingConfirmations },
    ) => void,
  ) => () => void;
};

const NONE: ReadonlySet<string> = new Set();

/** Chats with a tool call parked on Allow / Deny for at least APPROVAL_SETTLE_MS. */
export const useApprovalWaitingStore = create<{
  threadIds: ReadonlySet<string>;
}>(() => ({ threadIds: NONE }));

const parked = new Map<string, { threadId: string; since: number }>();
const settled = new Set<string>();
let timer: ReturnType<typeof setTimeout> | null = null;
let attentionRequested = false;

function windowIsInBackground(): boolean {
  if (typeof document === "undefined") return false;
  return (
    document.visibilityState !== "visible" ||
    (typeof document.hasFocus === "function" && !document.hasFocus())
  );
}

// null withdraws the request where the platform supports it.
function setWindowAttention(on: boolean): void {
  if (!isTauri) return;
  void import("@tauri-apps/api/window")
    .then(({ getCurrentWindow, UserAttentionType }) =>
      getCurrentWindow().requestUserAttention(
        on ? UserAttentionType.Informational : null,
      ),
    )
    .catch(() => undefined);
}

function publish(now: number): void {
  if (timer !== null) {
    clearTimeout(timer);
    timer = null;
  }
  const threadIds = new Set<string>();
  let newlySettled = false;
  let nextDue = Number.POSITIVE_INFINITY;
  for (const [id, { threadId, since }] of parked) {
    const due = since + APPROVAL_SETTLE_MS;
    if (due > now) {
      nextDue = Math.min(nextDue, due);
      continue;
    }
    if (!settled.has(id)) {
      settled.add(id);
      newlySettled = true;
    }
    // No saved chat yet: no row to mark, but it still asks for attention.
    if (threadId) threadIds.add(threadId);
  }
  if (nextDue !== Number.POSITIVE_INFINITY) {
    timer = setTimeout(() => {
      timer = null;
      publish(Date.now());
    }, nextDue - now);
  }

  const previous = useApprovalWaitingStore.getState().threadIds;
  if (
    previous.size !== threadIds.size ||
    [...threadIds].some((id) => !previous.has(id))
  ) {
    useApprovalWaitingStore.setState({
      threadIds: threadIds.size > 0 ? threadIds : NONE,
    });
  }

  if (newlySettled && windowIsInBackground()) {
    attentionRequested = true;
    setWindowAttention(true);
  } else if (settled.size === 0 && attentionRequested) {
    attentionRequested = false;
    setWindowAttention(false);
  }
}

/** The chat "Return to Chat" opens: the newest running one, or while any of them waits, the newest
 *  waiting one, since the row then shows the waiting mark. Insertion order is start order. */
export function returnToChatThreadId(
  runningByThreadId: Readonly<Record<string, boolean>>,
  waiting: ReadonlySet<string>,
): string | null {
  const running = Object.entries(runningByThreadId)
    .filter(([, on]) => on)
    .map(([id]) => id);
  const waitingRunning = running.filter((id) => waiting.has(id));
  const pool = waitingRunning.length > 0 ? waitingRunning : running;
  return pool.length > 0 ? pool[pool.length - 1] : null;
}

/** Feed the current parked calls; ids that are gone stop counting at once. */
export function syncApprovalWaiting(
  confirmations: PendingConfirmations,
  now: number = Date.now(),
): void {
  for (const id of [...parked.keys()]) {
    if (!Object.prototype.hasOwnProperty.call(confirmations, id)) {
      parked.delete(id);
      settled.delete(id);
    }
  }
  for (const [id, { threadId }] of Object.entries(confirmations)) {
    const known = parked.get(id);
    // A first turn's call gains its chat once the run is adopted; it keeps its park time.
    if (known) known.threadId = threadId;
    else parked.set(id, { threadId, since: now });
  }
  publish(now);
}

/** Follow the runtime store's parked calls until the returned function is called. */
export function watchApprovalWaiting(source: ConfirmationSource): () => void {
  syncApprovalWaiting(source.getState().toolConfirmations);
  const unsubscribe = source.subscribe((state, previous) => {
    if (state.toolConfirmations !== previous.toolConfirmations) {
      syncApprovalWaiting(state.toolConfirmations);
    }
  });
  return () => {
    unsubscribe();
    syncApprovalWaiting({});
  };
}
