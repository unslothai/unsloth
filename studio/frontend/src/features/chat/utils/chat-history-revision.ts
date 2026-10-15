// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Cross-document signal only; publishes that something changed, never ids or text.
export const CHAT_HISTORY_REVISION_KEY = "unsloth_chat_history_revision";

// Long enough to absorb per-chunk saves, short enough to publish before stale reads.
export const CROSS_TAB_REVISION_DEBOUNCE_MS = 500;

let revisionWriteTimer: ReturnType<typeof setTimeout> | null = null;

/** Coalesced chunk saves must not count as structural, or listeners starve all generation. */
export function isCoalescedHistoryEvent(event: Event): boolean {
  return (event as CustomEvent<{ coalesce?: boolean }>).detail?.coalesce === true;
}

function storage(): Storage | null {
  // No window under node, and the localStorage getter throws in some privacy modes.
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

/** `coalesce` is only for per-chunk streaming; structural changes must publish promptly. */
export function publishChatHistoryRevision(coalesce: boolean): void {
  if (!coalesce) {
    clearPending();
    writeRevision();
    return;
  }
  clearPending();
  revisionWriteTimer = setTimeout(() => {
    revisionWriteTimer = null;
    writeRevision();
  }, CROSS_TAB_REVISION_DEBOUNCE_MS);
}

export function flushChatHistoryRevision(): void {
  if (clearPending()) writeRevision();
}

// A pending coalesced write would otherwise be lost when the page unloads.
if (typeof window !== "undefined") {
  window.addEventListener("pagehide", flushChatHistoryRevision);
}
