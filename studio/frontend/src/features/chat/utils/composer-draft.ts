// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One draft shared by every chat, so the last thing typed follows the user. The chat-draft prefix
// is what the account switch clears (account-transition.ts).
const DRAFT_KEY = "chat-draft:latest";
const PASTE_DRAFT_KEY = "chat-draft-pastes:latest";
// Pre-upgrade per-chat drafts: adopted when their chat opens, removed with their chat.
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

// This tab's pastes once it saves any, as `latest` is for its text.
let latestPastes: string[] | null = null;

export function latestPasteDraft(): string[] {
  return latestPastes !== null ? [...latestPastes] : readPasteDraft(PASTE_DRAFT_KEY);
}

// A quota error here leaves the text draft untouched, hence separate slots.
export function writePasteDraft(key: string, pastes: readonly string[]): void {
  if (key === PASTE_DRAFT_KEY) latestPastes = [...pastes];
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

// A draft saved per chat before the shared slot existed becomes the shared draft when that chat
// opens and nothing newer is waiting, so an upgrade never loses unsent text. True if it did.
export function adoptLegacyDraft(threadId: string | null): boolean {
  // Text typed this session and not yet saved is newer than any old slot.
  if (latest) return false;
  const id = threadId ?? "__new__";
  try {
    const storage = window.localStorage;
    const text = storage.getItem(`${LEGACY_DRAFT_PREFIX}${id}`);
    const pastes = storage.getItem(`${LEGACY_PASTE_DRAFT_PREFIX}${id}`);
    if (text === null && pastes === null) return false;
    let adopted = false;
    if (storage.getItem(DRAFT_KEY) === null && storage.getItem(PASTE_DRAFT_KEY) === null) {
      if (text !== null) storage.setItem(DRAFT_KEY, text);
      try {
        if (pastes !== null) storage.setItem(PASTE_DRAFT_KEY, pastes);
      } catch (error) {
        // All or nothing: a text-only copy would make the next try skip, dropping the paste.
        storage.removeItem(DRAFT_KEY);
        throw error;
      }
      adopted = true;
      seeded = null;
      latestPastes = null;
    }
    storage.removeItem(`${LEGACY_DRAFT_PREFIX}${id}`);
    storage.removeItem(`${LEGACY_PASTE_DRAFT_PREFIX}${id}`);
    return adopted;
  } catch {
    return false;
  }
}

// The chat composer stays mounted, hidden, under Compare's, so every composer publishes its text
// here and hears the others; `latestDraft` is ahead of the debounced save. Other open tabs are
// never heard (#15782): the saved slot only seeds a tab once, for a reload or a new tab.
type DraftListener = (text: string, source: symbol) => void;
const draftListeners = new Set<DraftListener>();
let latest: string | null = null;
let seeded: string | null = null;

export function latestDraft(): string {
  if (latest !== null) return latest;
  seeded ??= readComposerDraft(DRAFT_KEY) ?? "";
  return seeded;
}

export function publishDraft(text: string, source: symbol): void {
  if (text === latest) return;
  latest = text;
  for (const listener of draftListeners) listener(text, source);
}

export function subscribeDraft(source: symbol, listener: (text: string) => void): () => void {
  const wrapped: DraftListener = (text, from) => {
    if (from !== source) listener(text);
  };
  draftListeners.add(wrapped);
  return () => draftListeners.delete(wrapped);
}

// A deleted chat's leftover per-chat draft; never the shared one.
export function clearComposerDraft(threadId: string): void {
  try {
    window.localStorage.removeItem(`${LEGACY_DRAFT_PREFIX}${threadId}`);
    window.localStorage.removeItem(`${LEGACY_PASTE_DRAFT_PREFIX}${threadId}`);
  } catch {
    // ignore
  }
}
