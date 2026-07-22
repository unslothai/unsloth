// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-thread composer drafts persisted in localStorage. New (unsaved) chats
// share the NEW_CHAT_DRAFT_ID slot; callers clear it when a fresh chat starts
// so one new chat's draft never bleeds into the next.
const DRAFT_PREFIX = "chat-draft:";
const NEW_CHAT_DRAFT_ID = "__new__";

export function composerDraftKey(threadId: string | null | undefined): string {
  return `${DRAFT_PREFIX}${threadId ?? NEW_CHAT_DRAFT_ID}`;
}

// All storage access is best-effort: localStorage throws when unavailable
// (private mode, blocked storage) or full (quota), so swallow failures.
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
  } catch {
    // ignore
  }
}

// Drop the shared new-chat draft so a freshly started chat opens empty.
export function clearNewChatDraft(): void {
  clearComposerDraft(null);
}
