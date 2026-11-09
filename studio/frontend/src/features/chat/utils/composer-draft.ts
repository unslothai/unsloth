// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The chat-draft prefix is what account-transition.ts clears on account switches.
const DRAFT_KEY = "chat-draft:latest";
const PASTE_DRAFT_KEY = "chat-draft-pastes:latest";
const LEGACY_DRAFT_PREFIX = "chat-draft:";
const LEGACY_PASTE_DRAFT_PREFIX = "chat-draft-pastes:";
// Unsent New chat and Compare ids are minted per navigation; per-id slots would pile up unread.
export const NEW_CHAT_DRAFT_ID = "__new__";
export const NEW_COMPARE_DRAFT_ID = "__compare_new__";

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
  return parsePastes(raw);
}

function parsePastes(raw: string | null): string[] {
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
  else {
    if (deletedDraftIds.has(key.slice(LEGACY_PASTE_DRAFT_PREFIX.length))) return;
    perChatPastes.set(key, [...pastes]);
    if (pastes.length > 0) lastTypedDraftId = key.slice(LEGACY_PASTE_DRAFT_PREFIX.length);
  }
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

// Typing inside the save debounce: this tab's cache shows it at once, so adoption takes it.
export function stageComposerDraft(key: string, text: string): void {
  if (key === DRAFT_KEY || deletedDraftIds.has(key.slice(LEGACY_DRAFT_PREFIX.length))) return;
  perChatDrafts.set(key, text);
  if (text) lastTypedDraftId = key.slice(LEGACY_DRAFT_PREFIX.length);
}

// A staged save whose slot was adopted (or deleted) since must not recreate it.
export function flushComposerDraft(key: string, text: string): void {
  if (key !== DRAFT_KEY && perChatDrafts.get(key) !== text) return;
  writeComposerDraft(key, text);
}

export function writeComposerDraft(key: string, text: string): void {
  if (key !== DRAFT_KEY) {
    if (deletedDraftIds.has(key.slice(LEGACY_DRAFT_PREFIX.length))) return;
    perChatDrafts.set(key, text);
  }
  try {
    if (text.length > 0) window.localStorage.setItem(key, text);
    else window.localStorage.removeItem(key);
  } catch {
    // ignore write failures
  }
}

// A draft saved per chat before the shared slot existed becomes the shared draft when that chat
// opens and nothing newer is waiting, so an upgrade never loses unsent text. True if it did.
export function adoptLegacyDraft(threadId: string | null, overwrite = false): boolean {
  // Text typed this session and not yet saved is newer than any old slot.
  if (latest && !overwrite) return false;
  const id = threadId ?? "__new__";
  try {
    const storage = window.localStorage;
    // This tab's copy first: another window may have written its own draft to the same slot since.
    const textKey = `${LEGACY_DRAFT_PREFIX}${id}`;
    const pasteKey = `${LEGACY_PASTE_DRAFT_PREFIX}${id}`;
    const cachedText = perChatDrafts.get(textKey);
    const cachedPastes = perChatPastes.get(pasteKey);
    const text = cachedText !== undefined ? cachedText || null : storage.getItem(textKey);
    const pastes =
      cachedPastes !== undefined
        ? cachedPastes.length > 0 ? JSON.stringify(cachedPastes) : null
        : storage.getItem(pasteKey);
    if (text === null && pastes === null) return false;
    let adopted = false;
    const prevText = storage.getItem(DRAFT_KEY);
    const prevPastes = storage.getItem(PASTE_DRAFT_KEY);
    if (overwrite || (prevText === null && prevPastes === null)) {
      try {
        if (text !== null) storage.setItem(DRAFT_KEY, text);
        else storage.removeItem(DRAFT_KEY);
        if (pastes !== null) storage.setItem(PASTE_DRAFT_KEY, pastes);
        else storage.removeItem(PASTE_DRAFT_KEY);
      } catch (error) {
        // All or nothing: a text-only copy would make the next try skip, dropping the paste.
        if (prevText === null) storage.removeItem(DRAFT_KEY);
        else storage.setItem(DRAFT_KEY, prevText);
        if (prevPastes === null) storage.removeItem(PASTE_DRAFT_KEY);
        else storage.setItem(PASTE_DRAFT_KEY, prevPastes);
        throw error;
      }
      adopted = true;
      // Seeded from what was adopted: another window's adoption may already have replaced storage.
      latest = text ?? "";
      seeded = null;
      latestPastes = parsePastes(pastes);
      // Moved, not copied: a copy left behind would return a prompt sent from the shared slot.
      dropPerChatDraft(id);
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
// This tab's last-typed per-chat slot: what sharing adopts, even unmounted (Settings hides chat).
let lastTypedDraftId: string | null = null;

export function adoptCurrentDraft(): boolean {
  // What this tab just typed wins over an older shared draft, including another window's (#15782).
  return lastTypedDraftId !== null && adoptLegacyDraft(lastTypedDraftId, true);
}

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

// A deleted chat's pending autosave can land after the delete; its slot stays closed.
const deletedDraftIds = new Set<string>();

export function clearComposerDraft(threadId: string): void {
  deletedDraftIds.add(threadId);
  dropPerChatDraft(threadId);
}

// While a delete is in flight: late writes are dropped, the draft survives a failed delete.
export function closeComposerDraft(threadId: string): void {
  deletedDraftIds.add(threadId);
}

export function reopenComposerDraft(threadId: string): void {
  deletedDraftIds.delete(threadId);
}

function dropPerChatDraft(threadId: string): void {
  if (lastTypedDraftId === threadId) lastTypedDraftId = null;
  perChatDrafts.delete(composerDraftKey(threadId));
  perChatPastes.delete(composerPasteDraftKey(threadId));
  try {
    window.localStorage.removeItem(`${LEGACY_DRAFT_PREFIX}${threadId}`);
    window.localStorage.removeItem(`${LEGACY_PASTE_DRAFT_PREFIX}${threadId}`);
  } catch {
    // ignore
  }
}
