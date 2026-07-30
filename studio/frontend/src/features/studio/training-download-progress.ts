// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type TrainingDownloadProgress = {
  downloaded_bytes: number;
  expected_bytes: number;
  progress: number;
  complete_on_disk: boolean;
  cache_path: string | null;
};

export type TrainingDownloadState = {
  downloadedBytes: number;
  totalBytes: number;
  percent: number;
  completeOnDisk: boolean;
  cachePath: string | null;
};

export const EMPTY_TRAINING_DOWNLOAD_STATE: TrainingDownloadState = {
  downloadedBytes: 0,
  totalBytes: 0,
  percent: 0,
  completeOnDisk: false,
  cachePath: null,
};

export function toTrainingDownloadState(
  progress: TrainingDownloadProgress,
): TrainingDownloadState {
  const downloadedBytes = Math.max(0, progress.downloaded_bytes ?? 0);
  const totalBytes = Math.max(0, progress.expected_bytes ?? 0);
  const ratio = Math.max(0, progress.progress ?? 0);
  const completeOnDisk = progress.complete_on_disk === true;
  const percent = completeOnDisk
    ? 100
    : totalBytes > 0
      ? Math.min(99, Math.round(ratio * 100))
      : 0;

  return {
    downloadedBytes,
    totalBytes,
    percent,
    completeOnDisk,
    cachePath: progress.cache_path ?? null,
  };
}
