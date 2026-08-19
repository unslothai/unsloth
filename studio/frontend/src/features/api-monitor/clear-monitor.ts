// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the monitor's "Clear log" button runs. Its own plain module because the hook it
// belongs to imports React and cannot be loaded by the node --test suite.

export type ClearMonitorDeps = {
  /** One DELETE /api/inference/monitor. Rejects when the server refuses or is unreachable. */
  clearRemote: () => Promise<void>;
  /** Drops the cached full payloads, whose rows the clear just deleted. */
  resetDetails: () => void;
  /** One refetch, so the emptied log is what the page renders. Settles either way. */
  reload: () => Promise<void>;
  /** The monitor's error banner. */
  onError: (message: string) => void;
};

/** Shown when the DELETE rejects with something that is not an Error. */
export const CLEAR_MONITOR_FAILED = "Failed to clear the monitor";

export async function clearMonitor(deps: ClearMonitorDeps): Promise<void> {
  try {
    await deps.clearRemote();
  } catch (err: unknown) {
    // The click handler discards this promise, so rethrowing is an unhandled rejection and
    // the page shows a cleared selection over an unchanged log with nothing said. Nothing
    // was deleted, so the cached details and the current snapshot both still stand.
    deps.onError(err instanceof Error ? err.message : CLEAR_MONITOR_FAILED);
    return;
  }
  deps.resetDetails();
  await deps.reload();
}
