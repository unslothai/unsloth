// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The live download pollers, one per dictation model. Each STT engine owns its
 * own download state, so a Qwen transfer and a Whisper one really do run at the
 * same time, and starting one must not drop the other's panel row.
 */
export class SttDownloadTrackers {
  private readonly running = new Map<string, () => void>();

  has(model: string): boolean {
    return this.running.has(model);
  }

  /** Stops this model's previous poller, if any, and registers the new one. */
  start(model: string, stop: () => void): void {
    this.stop(model);
    this.running.set(model, stop);
  }

  stop(model: string): void {
    const stop = this.running.get(model);
    this.running.delete(model);
    stop?.();
  }
}

export type SttReplacementAction = "track" | "retry" | "ignore";

/** Decide a confirmed replacement without letting an older response displace a newer one. */
export function sttReplacementAction(
  hasTracker: boolean,
  currentDownloadId: string | null | undefined,
  previousDownloadId: string,
  candidateDownloadId: string,
  backendDownloadId: string | null | undefined,
  candidateCompleted: boolean,
): SttReplacementAction {
  if (
    candidateCompleted &&
    currentDownloadId &&
    currentDownloadId === backendDownloadId &&
    currentDownloadId !== candidateDownloadId
  )
    return "ignore";
  if (!hasTracker || currentDownloadId === previousDownloadId) return "track";
  if (currentDownloadId === candidateDownloadId || candidateCompleted)
    return "ignore";
  return "retry";
}

/** A failed confirmation still needs another status check if no poller owns the candidate. */
export function shouldRecheckSttReplacement(
  currentDownloadId: string | null | undefined,
  candidateDownloadId: string,
): boolean {
  return currentDownloadId !== candidateDownloadId;
}
