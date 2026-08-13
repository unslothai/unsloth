// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the last completed search-index build found any chats, remembered across page
// loads. The index is only ever built while the dialog is open, so without this the first
// open of every page load cannot tell "no chats" from "not looked yet" and sizes itself for
// an empty history it may not have. Only the presence of rows is stored: a bare flag, never
// a count and never any chat text.
const CHAT_SEARCH_HAS_ROWS_KEY = "unsloth_chat_search_has_rows";

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
 * True or false from the last completed build, null when the history is unknown: never
 * built here, forgotten on a session change, or localStorage unavailable. Absence has to
 * stay distinguishable from a known-empty history, since only one of the two can be
 * sized compact without risking the mid-open resize this dialog exists to avoid.
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
