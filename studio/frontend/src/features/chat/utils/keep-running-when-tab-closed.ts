// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** On: a reply or Deep Research run outlives its tab, which is what a long research run needs. */
export const DEFAULT_KEEP_RUNNING_WHEN_TAB_CLOSED = true;

/**
 * Keep-running admission fields: off sets stopWhenUnattended so the server stops the run once no
 * tab follows it for a while; on sends nothing (backend state/unattended_runs.py).
 */
export function tabCloseRunFields(keepRunningWhenTabClosed: boolean): {
  stopWhenUnattended?: true;
} {
  return keepRunningWhenTabClosed ? {} : { stopWhenUnattended: true };
}
