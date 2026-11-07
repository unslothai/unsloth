// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sharing is opt-in. Keep its existing slots so opting out never destroys an unsent draft.
// The chat-draft prefix is what account-transition.ts clears on account switches.
const DRAFT_KEY = "chat-draft:latest";
const PASTE_DRAFT_KEY = "chat-draft-pastes:latest";
// Per-chat drafts retain the pre-sharing keys.
const LEGACY_DRAFT_PREFIX = "chat-draft:";
const LEGACY_PASTE_DRAFT_PREFIX = "chat-draft-pastes:";

export function composerDraftKey(threadId: string, shared = false): string {
  return shared ? DRAFT_KEY : `${LEGACY_DRAFT_PREFIX}${threadId}`;
}

// Pasted attachments live in their own slot rather than inside the text draft, so typing never
// rewrites a paste that can run to megabytes.
export function composerPasteDraftKey(threadId: string, shared = false): string {
  return shared ? PASTE_DRAFT_KEY : `${LEGACY_PASTE_DRAFT_PREFIX}${threadId}`;
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
const perChatPastes = new Map<string, string[]>();

export function latestPasteDraft(key = PASTE_DRAFT_KEY): string[] {
  if (key === PASTE_DRAFT_KEY) {
    return latestPastes !== null ? [...latestPastes] : readPasteDraft(key);
  }
  if (!perChatPastes.has(key)) perChatPastes.set(key, readPasteDraft(key));
  return [...perChatPastes.get(key)!];
}

// A quota error here leaves the text draft untouched, hence separate slots.
export function writePasteDraft(key: string, pastes: readonly string[]): void {
  if (key === PASTE_DRAFT_KEY) latestPastes = [...pastes];
  else perChatPastes.set(key, [...pastes]);
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
  if (key !== DRAFT_KEY) perChatDrafts.set(key, text);
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
      // Moved, not copied: a copy left behind would return a prompt sent from the shared slot.
      clearComposerDraft(id);
    }
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
const perChatDrafts = new Map<string, string>();

export function latestDraft(key = DRAFT_KEY): string {
  if (key !== DRAFT_KEY) {
    if (!perChatDrafts.has(key)) perChatDrafts.set(key, readComposerDraft(key) ?? "");
    return perChatDrafts.get(key)!;
  }
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

// Deleting a chat clears only its own draft, never the shared one.
export function clearComposerDraft(threadId: string): void {
  perChatDrafts.delete(composerDraftKey(threadId));
  perChatPastes.delete(composerPasteDraftKey(threadId));
  try {
    window.localStorage.removeItem(`${LEGACY_DRAFT_PREFIX}${threadId}`);
    window.localStorage.removeItem(`${LEGACY_PASTE_DRAFT_PREFIX}${threadId}`);
  } catch {
    // ignore
  }
}
