// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One poller per model: STT engines run downloads concurrently. */
export class SttDownloadTrackers {
  private readonly running = new Map<string, () => void>();

  has(model: string): boolean {
    return this.running.has(model);
  }

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

export function shouldRecheckSttReplacement(
  currentDownloadId: string | null | undefined,
  candidateDownloadId: string,
): boolean {
  return currentDownloadId !== candidateDownloadId;
}

/** An adopter's attempt is already tracked; a missing id (older backend) matches by row. */
export function isSameSttAttempt(
  hasTracker: boolean,
  trackedDownloadId: string | null | undefined,
  downloadId: string | null | undefined,
): boolean {
  return hasTracker && (!downloadId || trackedDownloadId === downloadId);
}
