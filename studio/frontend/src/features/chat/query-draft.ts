// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext } from "react";
import { createNavigationNonce } from "../../lib/navigation-nonce.ts";
import {
  QUERY_DRAFT_ID_PREFIX,
  composerDraftKey,
  composerPasteDraftKey,
  writeComposerDraft,
} from "./utils/composer-draft.ts";

// The nonce of the new chat this runtime opens; a `?q=` one names that link's draft slot.
export const NewChatNonceContext = createContext<string | null>(null);

// A temporary chat writes no draft to storage, so its link text waits here, keyed like a draft.
const temporaryQueries = new Map<string, string>();

// Never sends: any page can link to a local Studio, so the user presses Enter themselves.
export function seedQueryDraft(text: string, temporary: boolean): string {
  const nonce = QUERY_DRAFT_ID_PREFIX + createNavigationNonce();
  if (temporary) temporaryQueries.set(composerDraftKey(nonce), text);
  else writeComposerDraft(composerDraftKey(nonce), text);
  return nonce;
}

const RESUME_KEY = "chat-query-resume";

// A query chat interrupted by sign-in reopens after it; an account switch clears session storage.
export function rememberQueryDraftToResume(nonce: string): void {
  try {
    window.sessionStorage.setItem(RESUME_KEY, nonce);
  } catch {
    // Only the reopen is lost.
  }
}

export function takeQueryDraftToResume(): string | null {
  try {
    const nonce = window.sessionStorage.getItem(RESUME_KEY);
    window.sessionStorage.removeItem(RESUME_KEY);
    if (!nonce) return null;
    const saved = [composerDraftKey(nonce), composerPasteDraftKey(nonce)];
    return saved.some((key) => window.sessionStorage.getItem(key) !== null) ? nonce : null;
  } catch {
    return null;
  }
}

export function temporaryQueryDraft(draftKey: string): string {
  return temporaryQueries.get(draftKey) ?? "";
}

export function clearTemporaryQueryDraft(draftKey: string | null): void {
  if (draftKey) temporaryQueries.delete(draftKey);
}
