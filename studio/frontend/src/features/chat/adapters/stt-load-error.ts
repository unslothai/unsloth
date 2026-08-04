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
 * The backend's wording for a model that was never downloaded (stt_sidecar.py
 * and stt_ggml_sidecar.py, pinned by tests/studio/test_stt_load_409_contract.py).
 */
const NOT_DOWNLOADED = /is not downloaded/i;

/**
 * Whether a failed model load means transcription cannot succeed later either.
 *
 * Only two failures are final. 501 means STT support is not installed. 409 is
 * overloaded: it is also how a load cancelled so training could start is
 * reported (routes/inference.py), and /transcribe/raw recovers from that by
 * reloading on CPU, so only the not-downloaded wording counts.
 *
 * Everything else keeps recording, because /transcribe/raw loads the model
 * itself and may still succeed. An unrecognised 409 therefore errs towards
 * keeping the audio: a stale match costs a wasted recording, a wrong one costs
 * the user words they already spoke.
 */
export function isUnrecoverableSttLoadError(error: unknown): boolean {
  const status = (error as { status?: unknown } | null)?.status;
  if (status === 501) return true;
  if (status !== 409) return false;
  return error instanceof Error && NOT_DOWNLOADED.test(error.message);
}
