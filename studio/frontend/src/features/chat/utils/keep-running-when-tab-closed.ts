// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** On: a reply or Deep Research run outlives its tab, which is what a long research run needs. */
export const DEFAULT_KEEP_RUNNING_WHEN_TAB_CLOSED = true;

/**
 * What a durable chat or Deep Research run is admitted with for "Keep running after the tab closes".
 *
 * Off asks the server to stop the run once no tab has followed it for a while (backend
 * state/unattended_runs.py), so a reload or a second tab still keeps it. On sends nothing, which
 * keeps admission exactly what it was before the setting existed.
 */
export function tabCloseRunFields(keepRunningWhenTabClosed: boolean): {
  stopWhenUnattended?: true;
} {
  return keepRunningWhenTabClosed ? {} : { stopWhenUnattended: true };
}
