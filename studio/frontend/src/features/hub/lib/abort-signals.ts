// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// AbortSignal.timeout / .any are missing on older WebKitGTK that Tauri embeds; these ponyfills fall back.

export interface PollSignal {
  signal: AbortSignal;
  dispose: () => void;
}

// On the ponyfill path the timer pins the controller, so callers settling early MUST dispose.
export function disposableTimeoutSignal(ms: number): PollSignal {
  if (typeof AbortSignal.timeout === "function") {
    return { signal: AbortSignal.timeout(ms), dispose: () => {} };
  }
  const controller = new AbortController();
  const timer = setTimeout(
    () =>
      controller.abort(
        new DOMException("The operation timed out.", "TimeoutError"),
      ),
    ms,
  );
  return { signal: controller.signal, dispose: () => clearTimeout(timer) };
}

// Callers MUST dispose() once the request settles so abort listeners don't pile up.
export function combineAbortSignals(signals: AbortSignal[]): PollSignal {
  if (typeof AbortSignal.any === "function") {
    return { signal: AbortSignal.any(signals), dispose: () => {} };
  }
  const controller = new AbortController();
  const detachers: Array<() => void> = [];
  const dispose = () => {
    while (detachers.length > 0) {
      detachers.pop()?.();
    }
  };
  const abort = (reason: unknown) => {
    if (!controller.signal.aborted) controller.abort(reason);
    dispose();
  };
  for (const input of signals) {
    if (input.aborted) {
      abort(input.reason);
      break;
    }
    const handler = () => abort(input.reason);
    input.addEventListener("abort", handler, { once: true });
    detachers.push(() => input.removeEventListener("abort", handler));
  }
  return { signal: controller.signal, dispose };
}

export function pollSignal(parent: AbortSignal, timeoutMs: number): PollSignal {
  const timeout = disposableTimeoutSignal(timeoutMs);
  const combined = combineAbortSignals([parent, timeout.signal]);
  return {
    signal: combined.signal,
    dispose: () => {
      combined.dispose();
      timeout.dispose();
    },
  };
}

export function abortError(signal: AbortSignal): DOMException {
  return signal.reason instanceof DOMException
    ? signal.reason
    : new DOMException("The operation was aborted.", "AbortError");
}

// Never aborts the wrapped promise, so a shared request keeps running for other callers.
export function withAbort<T>(
  promise: Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortError(signal));
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", onAbort);
    });
  });
}
