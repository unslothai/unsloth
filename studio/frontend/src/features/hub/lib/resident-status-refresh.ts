// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// When the Hub has to re-read /api/inference/status.
//
// An OpenAI-compatible request can auto-switch the resident model at any moment,
// and nothing else on /hub reads status: the chat runtime hook has no mount sync
// and the chat page is a different route. A mount-only read therefore leaves
// every "loaded" marker -- and, worse, the settings page's live config -- pinned
// to whatever was resident when the Hub opened, so the newly loaded model's
// editor seeds from saved/default values and Apply reloads it with them.
//
// A background timer would keep asking a server that has usually not changed, so
// re-read only on the moments this tab could have missed a switch instead.

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
 * Call ``refresh`` whenever this tab comes back to the foreground: returning from
 * the terminal or client that made the API call is exactly when what is resident
 * may have moved. Returns the unsubscribe.
 */
export function subscribeResidentStatusRefresh(
  refresh: () => void,
  targets: ResidentStatusRefreshTargets = browserTargets(),
): () => void {
  const onFocus = () => refresh();
  // Focus alone misses a tab that was merely backgrounded, and visibility alone
  // misses a window that never went hidden; a redundant pair of reads is cheaper
  // than a missed switch.
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
