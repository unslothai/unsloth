// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext } from "react";
import {
  QUERY_DRAFT_ID_PREFIX,
  composerDraftKey,
  writeComposerDraft,
} from "./utils/composer-draft.ts";

// The nonce of the new chat this runtime opens; a `?q=` one names that link's draft slot.
export const NewChatNonceContext = createContext<string | null>(null);

// A temporary chat writes no draft to storage, so its link text waits here, keyed like a draft.
const temporaryQueries = new Map<string, string>();

// Never sends: any page can link to a local Studio, so the user presses Enter themselves.
export function seedQueryDraft(text: string, temporary: boolean): string {
  // Not randomUUID alone: Studio on a LAN address is not a secure context.
  const id =
    typeof globalThis.crypto?.randomUUID === "function"
      ? globalThis.crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const nonce = QUERY_DRAFT_ID_PREFIX + id;
  if (temporary) temporaryQueries.set(composerDraftKey(nonce), text);
  else writeComposerDraft(composerDraftKey(nonce), text);
  return nonce;
}

export function temporaryQueryDraft(draftKey: string): string {
  return temporaryQueries.get(draftKey) ?? "";
}

export function clearTemporaryQueryDraft(draftKey: string | null): void {
  if (draftKey) temporaryQueries.delete(draftKey);
}
