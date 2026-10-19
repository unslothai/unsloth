// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { accountDatabaseName } from "../../lib/account-transition.ts";
import {
  type TranscriptDetails,
  detailsFrom,
  sanitizeSpeakerName,
} from "./transcript-model.ts";

export interface TranscriptDraft {
  text: string;
  title: string;
  model: string;
  /** Absent on drafts written before timestamps existed. */
  details?: TranscriptDetails;
  speakerNames?: Record<string, string>;
}

function speakerNamesFrom(
  value: unknown,
  details: TranscriptDetails,
): Record<string, string> {
  const names: Record<string, string> = {};
  if (!value || typeof value !== "object" || Array.isArray(value)) return names;
  for (const [id, name] of Object.entries(value)) {
    if (typeof name !== "string") continue;
    if (!details.speakers.some((speaker) => speaker.id === id)) continue;
    const clean = sanitizeSpeakerName(name);
    if (clean) names[id] = clean;
  }
  return names;
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
    const restored: TranscriptDraft = {
      text: draft.text,
      title: draft.title,
      model: draft.model,
    };
    if (draft.details && typeof draft.details === "object") {
      const details = detailsFrom(draft.details);
      restored.details = details;
      restored.speakerNames = speakerNamesFrom(draft.speakerNames, details);
    }
    return restored;
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
