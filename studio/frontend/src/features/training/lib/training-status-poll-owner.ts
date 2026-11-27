// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Each entry says whether a mounted poller is fetching /api/train/status on its own timer right now. */
const statusPollers = new Set<() => boolean>();

export function registerTrainingStatusPoller(
  isPolling: () => boolean,
): () => void {
  statusPollers.add(isPolling);
  return () => {
    statusPollers.delete(isPolling);
  };
}

export function isTrainingStatusPolledElsewhere(): boolean {
  for (const isPolling of statusPollers) {
    if (isPolling()) return true;
  }
  return false;
}
