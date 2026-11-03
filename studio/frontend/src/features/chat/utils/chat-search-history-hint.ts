// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Persisted flag so the first open can tell "no chats" from "not indexed". Never a count or text.
const CHAT_SEARCH_HAS_ROWS_KEY = "unsloth_chat_search_has_rows";

// Only the empty answer expires: chats from other devices never reach this tab.
const EMPTY_HINT_TTL_MS = 12 * 60 * 60 * 1000;

function storage(): Storage | null {
  // No window under node, and the localStorage getter throws in some privacy modes.
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

/** null means unknown, which must stay distinct from known-empty. */
export function chatSearchHadRows(): boolean | null {
  const store = storage();
  if (!store) return null;
  try {
    const raw = store.getItem(CHAT_SEARCH_HAS_ROWS_KEY);
    if (raw === null) return null;
    if (raw === "1") return true;
    // A bare "0" predates the timestamp, so its age is unknown.
    if (!raw.startsWith("0.")) return null;
    const writtenAt = Number(raw.slice(2));
    if (!Number.isFinite(writtenAt)) return null;
    const age = Date.now() - writtenAt;
    if (age < 0 || age > EMPTY_HINT_TTL_MS) return null;
    return false;
  } catch {
    return null;
  }
}
