// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DownloadStartOutcome =
  | "started"
  | "cancelling"
  | "conflict"
  | "busy"
  | "error";

export type PendingStartMap = Map<string, Promise<DownloadStartOutcome>>;

export function hasPendingStartForRepo(
  pendingStarts: PendingStartMap,
  repoKey: string,
): boolean {
  for (const key of pendingStarts.keys()) {
    if (key === repoKey || key.startsWith(`${repoKey}#`)) return true;
  }
  return false;
}

export function runOrJoinPendingStart(
  pendingStarts: PendingStartMap,
  key: string,
  action: () => Promise<DownloadStartOutcome>,
  reportError: (error: unknown) => void,
): Promise<DownloadStartOutcome> {
  const existing = pendingStarts.get(key);
  if (existing) return existing;

  const pending = Promise.resolve()
    .then(action)
    .catch((error: unknown) => {
      reportError(error);
      return "error" as const;
    })
    .finally(() => {
      if (pendingStarts.get(key) === pending) {
        pendingStarts.delete(key);
      }
    });
  pendingStarts.set(key, pending);
  return pending;
}
