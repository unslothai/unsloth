// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Plain module: the hook imports React and cannot load under node --test.

export type ClearMonitorDeps = {
  clearRemote: () => Promise<void>;
  resetDetails: () => void;
  reload: () => Promise<void>;
  onError: (message: string) => void;
};

export const CLEAR_MONITOR_FAILED = "Failed to clear the monitor";

export async function clearMonitor(deps: ClearMonitorDeps): Promise<void> {
  try {
    await deps.clearRemote();
  } catch (err: unknown) {
    // The click handler discards this promise; rethrowing would be an unhandled rejection with no message.
    deps.onError(err instanceof Error ? err.message : CLEAR_MONITOR_FAILED);
    return;
  }
  deps.resetDetails();
  await deps.reload();
}
