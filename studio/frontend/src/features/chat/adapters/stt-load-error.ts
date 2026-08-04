// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Error from an STT endpoint, carrying the HTTP status the caller can act on. */
export type SttRequestError = Error & { status: number };

export function sttRequestError(
  detail: string,
  status: number,
): SttRequestError {
  return Object.assign(new Error(detail), { status });
}

/**
 * Whether a failed model load means transcription cannot succeed later either.
 *
 * 409 (model not downloaded) and 501 (STT support not installed) come back
 * unchanged at stop time, so a recording started after one is already lost.
 * Every other failure may still transcribe, since /transcribe/raw loads the
 * model itself, and must not cost the user their dictation.
 */
export function isUnrecoverableSttLoadError(error: unknown): boolean {
  const status = (error as { status?: unknown } | null)?.status;
  return status === 409 || status === 501;
}
