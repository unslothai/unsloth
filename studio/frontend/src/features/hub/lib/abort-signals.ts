// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// WebView-compatibility shims. AbortSignal.timeout (WebKitGTK < 2.38) and
// AbortSignal.any (WebKitGTK < 2.44) are missing on older engines Tauri embeds
// and throw synchronously if called. These ponyfills delegate to the native
// impl when present, else fall back to an AbortController.

export interface PollSignal {
  signal: AbortSignal;
  dispose: () => void;
}

// Timeout signal paired with a disposer. On the ponyfill path the setTimeout
// pins the controller until it fires, so callers settling early MUST dispose.
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

export function timeoutSignal(ms: number): AbortSignal {
  return disposableTimeoutSignal(ms).signal;
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

// Rejects the returned promise on abort but never aborts the wrapped promise,
// so a request shared across callers keeps running when one caller's signal fires.
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
