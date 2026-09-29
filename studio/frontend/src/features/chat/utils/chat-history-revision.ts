// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How a history change crosses documents: the event chat-api raises alongside it is
// same-document. Only that something changed is published, never a chat id or any text.
export const CHAT_HISTORY_REVISION_KEY = "unsloth_chat_history_revision";

// Long enough to swallow a generation's per-chunk saves, short enough that a tab going quiet
// publishes before anyone reads a stale row. Exported for the tests.
export const CROSS_TAB_REVISION_DEBOUNCE_MS = 500;

let revisionWriteTimer: ReturnType<typeof setTimeout> | null = null;

/** Whether a same-document history event came from the coalesced streaming autosave rather than a
 *  structural change. A listener that retires work on a history change needs the difference:
 *  chunk saves arrive faster than any debounce, so treating one as structural starves that work
 *  for a whole generation. Anything without the detail counts as structural, including the event
 *  the cross-tab listener re-raises. */
export function isCoalescedHistoryEvent(event: Event): boolean {
  return (event as CustomEvent<{ coalesce?: boolean }>).detail?.coalesce === true;
}

function storage(): Storage | null {
  // no window under node, and the localStorage getter itself throws in some privacy modes
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

function writeRevision(): void {
  const store = storage();
  if (!store) return;
  try {
    store.setItem(CHAT_HISTORY_REVISION_KEY, `${Date.now()}.${Math.random()}`);
  } catch {
    // A full quota costs one stale open, not correctness: other tabs revalidate anyway.
  }
}

function clearPending(): boolean {
  if (revisionWriteTimer === null) return false;
  clearTimeout(revisionWriteTimer);
  revisionWriteTimer = null;
  return true;
}

/** Publishes a history change to the other documents. `coalesce` is for the per-chunk streaming
 *  path alone, where a write per chunk would block this tab and wake every other one. A
 *  structural change must not use it: sharing the stream's quiet window leaves a deleted chat
 *  live in another tab for a whole generation. */
export function publishChatHistoryRevision(coalesce: boolean): void {
  if (!coalesce) {
    // Anything still waiting says no more than this does.
    clearPending();
    writeRevision();
    return;
  }
  // Rescheduled rather than left to run, so a generation collapses into one write instead of one per debounce period.
  clearPending();
  revisionWriteTimer = setTimeout(() => {
    revisionWriteTimer = null;
    writeRevision();
  }, CROSS_TAB_REVISION_DEBOUNCE_MS);
}

/** Publishes a coalesced write that is still waiting. Nothing to do when none is. */
export function flushChatHistoryRevision(): void {
  if (clearPending()) writeRevision();
}

// A coalesced write would otherwise leave with the page, stranding the other tabs.
if (typeof window !== "undefined") {
  window.addEventListener("pagehide", flushChatHistoryRevision);
}
