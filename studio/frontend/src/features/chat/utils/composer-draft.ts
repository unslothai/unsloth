// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// New unsaved chats share NEW_CHAT_DRAFT_ID; callers clear it when a fresh chat starts.
const DRAFT_PREFIX = "chat-draft:";
const PASTE_DRAFT_PREFIX = "chat-draft-pastes:";
const NEW_CHAT_DRAFT_ID = "__new__";

export function composerDraftKey(threadId: string | null | undefined): string {
  return `${DRAFT_PREFIX}${threadId ?? NEW_CHAT_DRAFT_ID}`;
}

// Pastes live in their own slot so typing never rewrites megabytes of pasted text.
export function composerPasteDraftKey(
  threadId: string | null | undefined,
): string {
  return `${PASTE_DRAFT_PREFIX}${threadId ?? NEW_CHAT_DRAFT_ID}`;
}

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

// A quota error here leaves the text draft untouched, hence separate slots.
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

// localStorage throws when unavailable or full, so all access is best-effort.
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

export function clearNewChatDraft(): void {
  clearComposerDraft(null);
}
