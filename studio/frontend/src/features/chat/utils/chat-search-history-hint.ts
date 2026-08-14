// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the last build found any chats, remembered across page loads. The index is built
// only while the dialog is open, so without this the first open of a page load cannot tell
// "no chats" from "not looked yet". A bare flag: never a count, never any chat text.
const CHAT_SEARCH_HAS_ROWS_KEY = "unsloth_chat_search_has_rows";

function storage(): Storage | null {
  // Guarded like features/auth/session.ts: no window under node, and localStorage throws
  // outright in some privacy modes.
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

export function rememberChatSearchHasRows(hasRows: boolean): void {
  const store = storage();
  if (!store) return;
  try {
    store.setItem(CHAT_SEARCH_HAS_ROWS_KEY, hasRows ? "1" : "0");
  } catch {
    // A hint, not state: a full quota costs a resize on one open, not correctness.
  }
}

export function forgetChatSearchHasRows(): void {
  const store = storage();
  if (!store) return;
  try {
    store.removeItem(CHAT_SEARCH_HAS_ROWS_KEY);
  } catch {
    // As above.
  }
}

/**
 * The last completed build's answer, or null when unknown: never built here, forgotten on a
 * session change, or no localStorage. Unknown must stay distinct from known-empty, since
 * only known-empty can size compact without risking a mid-open resize.
 */
export function chatSearchHadRows(): boolean | null {
  const store = storage();
  if (!store) return null;
  try {
    const raw = store.getItem(CHAT_SEARCH_HAS_ROWS_KEY);
    if (raw === null) return null;
    return raw === "1";
  } catch {
    return null;
  }
}
