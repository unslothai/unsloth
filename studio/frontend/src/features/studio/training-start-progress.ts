// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DownloadProgressLike = {
  downloaded_bytes?: number;
  expected_bytes?: number;
  progress?: number;
  cache_path?: string | null;
  complete_on_disk?: boolean;
};

export type DownloadState = {
  downloadedBytes: number;
  totalBytes: number;
  percent: number;
  cachePath: string | null;
  completeOnDisk: boolean;
};

export const EMPTY_DOWNLOAD_STATE: DownloadState = {
  downloadedBytes: 0,
  totalBytes: 0,
  percent: 0,
  cachePath: null,
  completeOnDisk: false,
};

export function downloadStateFromProgress(
  progress: DownloadProgressLike,
): DownloadState {
  const downloadedBytes = progress.downloaded_bytes ?? 0;
  const totalBytes = progress.expected_bytes ?? 0;
  const ratio = progress.progress ?? 0;
  const completeOnDisk = progress.complete_on_disk === true || ratio >= 1;

  return {
    downloadedBytes,
    totalBytes,
    percent: completeOnDisk
      ? 100
      : totalBytes > 0
        ? Math.min(100, Math.round(ratio * 100))
        : 0,
    cachePath: progress.cache_path ?? null,
    completeOnDisk,
  };
}

export function coerceCachedStateReady(
  state: DownloadState,
  assumeReady = false,
): DownloadState {
  if (state.completeOnDisk) {
    const totalBytes =
      state.totalBytes > 0 ? state.totalBytes : state.downloadedBytes;
    return {
      ...state,
      downloadedBytes: totalBytes > 0 ? totalBytes : state.downloadedBytes,
      totalBytes,
      percent: 100,
    };
  }
  if (!state.cachePath) return state;
  const totalBytes =
    state.totalBytes > 0 ? state.totalBytes : state.downloadedBytes;
  if (
    state.downloadedBytes <= 0 ||
    (!assumeReady && state.downloadedBytes < totalBytes)
  ) {
    return state;
  }
  if (totalBytes <= 0) {
    return { ...state, percent: 100, completeOnDisk: true };
  }
  return {
    ...state,
    downloadedBytes: totalBytes,
    totalBytes,
    percent: 100,
    completeOnDisk: true,
  };
}

export function isDownloadComplete(state: DownloadState): boolean {
  return state.completeOnDisk;
}

export function shouldShowPreparationStatus(
  phase: string,
  currentStep: number,
  isStarting: boolean,
): boolean {
  if (phase === "downloading_model" || phase === "downloading_dataset") {
    return false;
  }
  if (isStarting) return true;
  return (
    phase === "loading_model" ||
    phase === "loading_dataset" ||
    phase === "configuring" ||
    (phase === "training" && currentStep <= 0)
  );
}

export function resolvePreparationMessage(
  message: string,
  fallback: string,
): string {
  const trimmed = message.trim();
  return trimmed && !/^download/i.test(trimmed) ? trimmed : fallback;
}

export type PreparationProgress = {
  title: string;
  detail: string | null;
  percent: number | null;
};

const QUANTITATIVE_PREPARATION_RE =
  /^(?<label>.+?)\s+(?<reportedPercent>\d{1,3})%\s+\((?<current>[\d,]+)\s*\/\s*(?<total>[\d,]+)\)\s*$/;

function cleanPreparationTitle(label: string): string {
  return label
    .replace(/^Unsloth:\s*/i, "")
    .replace(/\s+\(num_proc=\d+\)\s*$/i, "")
    .replace(/(?:\.\.\.|…)\s*$/, "")
    .trim();
}

function indeterminatePreparation(title: string): PreparationProgress {
  return { title: cleanPreparationTitle(title), detail: null, percent: null };
}

export function parsePreparationProgress(
  message: string,
  fallback: string,
): PreparationProgress {
  const resolved = resolvePreparationMessage(message, fallback);
  const match = QUANTITATIVE_PREPARATION_RE.exec(resolved);
  if (!match?.groups) return indeterminatePreparation(resolved);

  const current = Number(match.groups.current.replaceAll(",", ""));
  const total = Number(match.groups.total.replaceAll(",", ""));
  if (
    !Number.isFinite(current) ||
    !Number.isFinite(total) ||
    current < 0 ||
    total <= 0 ||
    current > total
  ) {
    return indeterminatePreparation(match.groups.label);
  }

  const percent = Math.min(100, Math.max(0, (current / total) * 100));
  return {
    title: cleanPreparationTitle(match.groups.label),
    detail: `${Math.round(percent)}% (${match.groups.current}/${match.groups.total})`,
    percent,
  };
}
