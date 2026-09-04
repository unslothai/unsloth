// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-thread composer drafts persisted in localStorage. New (unsaved) chats share the
// NEW_CHAT_DRAFT_ID slot; callers clear it when a fresh chat starts so one new chat's draft
// never bleeds into the next.
const DRAFT_PREFIX = "chat-draft:";
const PASTE_DRAFT_PREFIX = "chat-draft-pastes:";
const NEW_CHAT_DRAFT_ID = "__new__";

export function composerDraftKey(threadId: string | null | undefined): string {
  return `${DRAFT_PREFIX}${threadId ?? NEW_CHAT_DRAFT_ID}`;
}

// Pasted attachments live in their own slot rather than inside the text draft, so typing never
// rewrites a paste that can run to megabytes.
export function composerPasteDraftKey(
  threadId: string | null | undefined,
): string {
  return `${PASTE_DRAFT_PREFIX}${threadId ?? NEW_CHAT_DRAFT_ID}`;
}

// The names are not stored: a pasted file is named from its own text, so recreating it reproduces the name it had.
export function readPasteDraft(key: string): string[] {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(key);
  } catch {
    return [];
  }
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((entry): entry is string => typeof entry === "string");
  } catch {
    return [];
  }
}

// A paste large enough to blow the storage quota throws here, leaving the text draft untouched,
// which is why the two slots are written separately.
export function writePasteDraft(key: string, pastes: readonly string[]): void {
  try {
    if (pastes.length > 0) {
      window.localStorage.setItem(key, JSON.stringify(pastes));
    } else {
      window.localStorage.removeItem(key);
    }
  } catch {
    // ignore write failures
  }
}

// All storage access is best-effort: localStorage throws when unavailable (private mode, blocked
// storage) or full, so swallow failures.
export function readComposerDraft(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeComposerDraft(key: string, text: string): void {
  try {
    if (text.length > 0) window.localStorage.setItem(key, text);
    else window.localStorage.removeItem(key);
  } catch {
    // ignore write failures
  }
}

export function clearComposerDraft(threadId: string | null | undefined): void {
  try {
    window.localStorage.removeItem(composerDraftKey(threadId));
    window.localStorage.removeItem(composerPasteDraftKey(threadId));
  } catch {
    // ignore
  }
}

// Drop the shared new-chat draft so a freshly started chat opens empty.
export function clearNewChatDraft(): void {
  clearComposerDraft(null);
}
