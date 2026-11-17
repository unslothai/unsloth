// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function failureLabelFor(candidate: {
  id: string;
  ggufVariant: string | null;
}): string {
  return candidate.ggufVariant
    ? `${candidate.id} (${candidate.ggufVariant})`
    : candidate.id;
}

// Quant names match case-insensitively, as resolveAutoLoadCandidate picks them.
export function sameVariant(a: string | null, b: string | null): boolean {
  return (
    (a?.trim().toLowerCase() || null) === (b?.trim().toLowerCase() || null)
  );
}

const ENDS_SENTENCE_RE = /[.!?]$/;
// The backend appends the llama-server log tail; a toast keeps only the summary before it.
const SERVER_OUTPUT_RE = /\s*llama-server output:/i;
const MAX_REASON_CHARS = 200;

/** Shown on the fallback's success toast, so a stand-in model never reads as the last used one. */
export function rememberedMissNote(
  label: string,
  detail: string | null,
): string {
  let reason = (detail ?? "").split(SERVER_OUTPUT_RE)[0].trim();
  if (reason.length > MAX_REASON_CHARS) {
    reason = `${reason.slice(0, MAX_REASON_CHARS - 1).trimEnd()}…`;
  } else if (reason && !ENDS_SENTENCE_RE.test(reason)) {
    reason = `${reason}.`;
  }
  return reason
    ? `Your last used model ${label} did not load: ${reason}`
    : `Your last used model ${label} did not load.`;
}

/** A fallback load is never a plain success: lead with why the last used model was skipped. */
export function withRememberedMiss<
  T extends { description: string | undefined; degraded: boolean },
>(notice: T, missedNote: string | null): T {
  if (!missedNote) {
    return notice;
  }
  return {
    ...notice,
    description: [missedNote, notice.description].filter(Boolean).join(" "),
    degraded: true,
  };
}
