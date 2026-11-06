// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const INPUT_QUIET_MS = 300;
const INPUT_EVENTS = [
  "wheel",
  "scroll",
  "pointerdown",
  "keydown",
  "touchstart",
  "touchmove",
] as const;
let lastInputAt = Number.NEGATIVE_INFINITY;
let watchingInput = false;
const noteInput = (): void => {
  lastInputAt = performance.now();
};
const watchInput = (): void => {
  if (watchingInput || typeof document === "undefined") return;
  watchingInput = true;
  for (const type of INPUT_EVENTS) {
    document.addEventListener(type, noteInput, { capture: true, passive: true });
  }
};

export const inputQuietIn = (): number => {
  watchInput();
  return INPUT_QUIET_MS - (performance.now() - lastInputAt);
};

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

export function scheduleQuietIdleTask(
  callback: () => void,
  timeout = 250,
): () => void {
  if (
    typeof window === "undefined" ||
    typeof window.requestIdleCallback === "function"
  ) {
    return scheduleIdleTask(callback, timeout);
  }

  watchInput();
  const deadline = performance.now() + timeout;
  let handle: ReturnType<typeof setTimeout>;
  const attempt = (): void => {
    const wait = Math.min(inputQuietIn(), deadline - performance.now());
    if (wait > 0) {
      handle = globalThis.setTimeout(attempt, wait);
      return;
    }
    callback();
  };
  handle = globalThis.setTimeout(attempt, Math.min(timeout, 16));
  return () => globalThis.clearTimeout(handle);
}
