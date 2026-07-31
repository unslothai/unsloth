// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether the recording bar's send should submit once dictation ends.
 *
 * Only when this recording actually added text: silence and failed
 * transcription leave a pre-recording draft in place rather than sending it
 * half-finished.
 *
 * @param before composer text captured when send was pressed
 * @param after composer text once the session ended
 */
export function dictationProducedText(before: string, after: string): boolean {
  const trimmed = after.trim();
  return trimmed.length > 0 && trimmed !== before.trim();
}
