// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How a history change crosses documents. The event chat-api raises alongside this is
// same-document, so a delete in one tab would otherwise never reach a cache built in
// another. Only the fact that something changed is published: a value that differs from
// the last one, never a chat id and never any chat text.
export const CHAT_HISTORY_REVISION_KEY = "unsloth_chat_history_revision";

// Long enough to swallow a generation's per-chunk saves, short enough that a tab which
// goes quiet publishes before anyone reads a stale row. Exported for the tests, which
// drive the real timer rather than a stand-in for it.
export const CROSS_TAB_REVISION_DEBOUNCE_MS = 500;

let revisionWriteTimer: ReturnType<typeof setTimeout> | null = null;

function storage(): Storage | null {
  // Guarded like features/auth/session.ts: no window under the node test runner, and
  // localStorage throws outright in some privacy modes.
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
    // A full quota costs other tabs one stale open, not correctness: they revalidate anyway.
  }
}

function clearPending(): boolean {
  if (revisionWriteTimer === null) return false;
  clearTimeout(revisionWriteTimer);
  revisionWriteTimer = null;
  return true;
}

/**
 * Publishes a history change to the other documents.
 *
 * `coalesce` belongs to the per-chunk streaming path alone, where a synchronous write per
 * chunk would block the producing tab and wake every other one. A structural change must
 * not use it: sharing the stream's quiet window would leave a deleted chat unannounced for
 * the length of a generation, and another tab offering it as a live row for that whole time.
 */
export function publishChatHistoryRevision(coalesce: boolean): void {
  if (!coalesce) {
    // Anything still waiting on the stream's quiet window says no more than this does.
    clearPending();
    writeRevision();
    return;
  }
  // Rescheduled rather than left to run, so a generation collapses into the one write its
  // quiet window earns instead of one every debounce period throughout.
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

// A coalesced write would otherwise go with the page, leaving the other tabs on a snapshot
// of a history that has moved on.
if (typeof window !== "undefined") {
  window.addEventListener("pagehide", flushChatHistoryRevision);
}
