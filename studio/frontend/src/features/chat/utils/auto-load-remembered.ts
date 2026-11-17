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

/** Shown on the fallback's success toast, so a stand-in model never reads as the last used one. */
export function rememberedMissNote(
  label: string,
  detail: string | null,
): string {
  const reason = detail?.trim();
  return reason
    ? `Your last used model ${label} did not load: ${ENDS_SENTENCE_RE.test(reason) ? reason : `${reason}.`}`
    : `Your last used model ${label} did not load.`;
}

/** A fallback load is never a plain success: lead with why the last used model was skipped. */
export function withRememberedMiss<
  T extends { description: string | undefined; degraded: boolean },
>(notice: T, missedNote: string | null): T {
  if (!missedNote) return notice;
  return {
    ...notice,
    description: [missedNote, notice.description].filter(Boolean).join(" "),
    degraded: true,
  };
}
