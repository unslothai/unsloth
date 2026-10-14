// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One composer draft shared by every chat, persisted in localStorage, so whatever was typed last
// follows the user to any chat, new or saved, and across reloads and tabs. The account switch
// clears it with the other chat-draft keys (account-transition.ts).
const DRAFT_KEY = "chat-draft:latest";
const PASTE_DRAFT_KEY = "chat-draft-pastes:latest";
// Drafts used to be kept per chat under these prefixes; only deleting a chat still touches them.
const LEGACY_DRAFT_PREFIX = "chat-draft:";
const LEGACY_PASTE_DRAFT_PREFIX = "chat-draft-pastes:";

export function composerDraftKey(): string {
  return DRAFT_KEY;
}

// Pasted attachments live in their own slot rather than inside the text draft, so typing never
// rewrites a paste that can run to megabytes.
export function composerPasteDraftKey(): string {
  return PASTE_DRAFT_KEY;
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

// Removes a deleted chat's leftover per-chat draft. Never the shared draft, which belongs to no chat.
export function clearComposerDraft(threadId: string): void {
  try {
    window.localStorage.removeItem(`${LEGACY_DRAFT_PREFIX}${threadId}`);
    window.localStorage.removeItem(`${LEGACY_PASTE_DRAFT_PREFIX}${threadId}`);
  } catch {
    // ignore
  }
}
