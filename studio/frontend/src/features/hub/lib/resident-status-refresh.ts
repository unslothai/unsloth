// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Re-read when the tab regains focus instead of polling: an API call may switch the resident model.

export interface ResidentStatusRefreshTargets {
  window: Pick<EventTarget, "addEventListener" | "removeEventListener">;
  document: Pick<EventTarget, "addEventListener" | "removeEventListener"> & {
    readonly hidden: boolean;
  };
}

function browserTargets(): ResidentStatusRefreshTargets {
  return { window, document };
}

export function subscribeResidentStatusRefresh(
  refresh: () => void,
  targets: ResidentStatusRefreshTargets = browserTargets(),
): () => void {
  const onFocus = () => refresh();
  // Focus alone misses a backgrounded tab, visibility alone a window that never hid.
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
