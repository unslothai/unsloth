// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Run `callback` once the main thread is idle, or after `timeout` at the latest; returns a
 * canceller. Falls back to setTimeout without requestIdleCallback (Safari, the WebKitGTK webview
 * the desktop app embeds on Linux), and runs synchronously with no window.
 */
export function scheduleIdleTask(
  callback: () => void,
  timeout = 250,
): () => void {
  let canceled = false;
  const run = () => {
    if (!canceled) callback();
  };

  if (typeof window === "undefined") {
    run();
    return () => {
      canceled = true;
    };
  }

  const idleWindow = window as Window & {
    requestIdleCallback?: Window["requestIdleCallback"];
    cancelIdleCallback?: Window["cancelIdleCallback"];
  };

  if (idleWindow.requestIdleCallback && idleWindow.cancelIdleCallback) {
    const handle = idleWindow.requestIdleCallback(run, { timeout });
    return () => {
      canceled = true;
      idleWindow.cancelIdleCallback?.(handle);
    };
  }

  const handle = globalThis.setTimeout(run, Math.min(timeout, 120));
  return () => {
    canceled = true;
    globalThis.clearTimeout(handle);
  };
}
