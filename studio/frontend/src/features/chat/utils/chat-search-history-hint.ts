// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether the last build found any chats, remembered across page loads. The index is built only
// while the dialog is open, so without this the first open of a page load cannot tell "no
// chats" from "not looked yet". A bare flag: never a count, never any chat text.
const CHAT_SEARCH_HAS_ROWS_KEY = "unsloth_chat_search_has_rows";

// Only the empty answer is aged out, and it carries the time it was written. Chats created on
// another device or through the API never reach this tab, so emptiness cannot stay
// authoritative: a stale "has rows" costs a full-height dialog over a short history, while a
// stale "empty" sizes a populated one compact and then grows it mid-open. Long enough to
// survive a restart within a working day.
const EMPTY_HINT_TTL_MS = 12 * 60 * 60 * 1000;

function storage(): Storage | null {
  // no window under node, and the localStorage getter itself throws in some privacy modes
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
    store.setItem(CHAT_SEARCH_HAS_ROWS_KEY, hasRows ? "1" : `0.${Date.now()}`);
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

/** The last completed build's answer, or null when unknown: never built here, forgotten on a
 *  session change, an empty answer past its window, or no localStorage. Unknown must stay
 *  distinct from known-empty, since only known-empty can size compact without a mid-open resize. */
export function chatSearchHadRows(): boolean | null {
  const store = storage();
  if (!store) return null;
  try {
    const raw = store.getItem(CHAT_SEARCH_HAS_ROWS_KEY);
    if (raw === null) return null;
    if (raw === "1") return true;
    // A bare "0" was written before the stamp existed, so its age is unknown.
    if (!raw.startsWith("0.")) return null;
    const writtenAt = Number(raw.slice(2));
    if (!Number.isFinite(writtenAt)) return null;
    // A negative age is a clock change, which says as little as an expired one.
    const age = Date.now() - writtenAt;
    if (age < 0 || age > EMPTY_HINT_TTL_MS) return null;
    return false;
  } catch {
    return null;
  }
}
