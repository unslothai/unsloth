// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { create } from "zustand";

// A call covered by "Always allow" still parks, and its card answers it within one request.
// Only a call still parked after this long is waiting on the user, so the sidebar dot does not
// flicker and the taskbar does not flash for a call nobody has to answer.
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

// Parked tool-call id -> its chat and when it was first seen parked.
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

// Windows flashes the taskbar button until the window is focused, macOS bounces the Dock icon
// once, Linux sets the urgency hint. null withdraws the request where the platform supports it.
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
    // A run with no saved chat yet has no sidebar row to mark, but still asks for attention.
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
    if (!parked.has(id)) parked.set(id, { threadId, since: now });
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
