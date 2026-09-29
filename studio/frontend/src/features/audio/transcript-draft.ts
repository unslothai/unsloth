// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { accountDatabaseName } from "../../lib/account-transition.ts";

export interface TranscriptDraft {
  text: string;
  title: string;
  model: string;
}

export function transcriptDraftKey(): string {
  return accountDatabaseName("unsloth:audio:unsaved-transcript");
}

export function readTranscriptDraft(key: string): TranscriptDraft | null {
  try {
    const draft = JSON.parse(sessionStorage.getItem(key) ?? "null");
    if (
      !draft ||
      typeof draft.text !== "string" ||
      !draft.text ||
      typeof draft.title !== "string" ||
      typeof draft.model !== "string"
    ) {
      return null;
    }
    return { text: draft.text, title: draft.title, model: draft.model };
  } catch {
    return null;
  }
}

export function writeTranscriptDraft(
  key: string,
  draft: TranscriptDraft | null,
): boolean {
  try {
    if (draft) sessionStorage.setItem(key, JSON.stringify(draft));
    else sessionStorage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}
