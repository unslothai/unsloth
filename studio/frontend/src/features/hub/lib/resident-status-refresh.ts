// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An API request can switch the resident model while the Hub is open. Re-read
// only when this tab could have missed a switch, rather than polling.

/** The event targets to listen on; injected so this is testable off a browser. */
export interface ResidentStatusRefreshTargets {
  window: Pick<EventTarget, "addEventListener" | "removeEventListener">;
  document: Pick<EventTarget, "addEventListener" | "removeEventListener"> & {
    readonly hidden: boolean;
  };
}

function browserTargets(): ResidentStatusRefreshTargets {
  return { window, document };
}

/**
 * Call ``refresh`` whenever this tab returns to the foreground: coming back from the
 * client that made the API call is when the resident model may have moved.
 */
export function subscribeResidentStatusRefresh(
  refresh: () => void,
  targets: ResidentStatusRefreshTargets = browserTargets(),
): () => void {
  const onFocus = () => refresh();
  // Focus alone misses a backgrounded tab, visibility alone a window that never hid;
  // a redundant pair of reads is cheaper than a missed switch.
  const onVisibility = () => {
    if (!targets.document.hidden) refresh();
  };
  targets.window.addEventListener("focus", onFocus);
  targets.document.addEventListener("visibilitychange", onVisibility);
  return () => {
    targets.window.removeEventListener("focus", onFocus);
    targets.document.removeEventListener("visibilitychange", onVisibility);
  };
}
