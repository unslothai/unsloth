// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// WebView-compatibility shims for AbortSignal helpers. AbortSignal.timeout
// (WebKitGTK < 2.38, Safari < 16) and AbortSignal.any (WebKitGTK < 2.44,
// Safari < 17.4) are missing on the older engines Tauri can embed; calling
// them directly throws synchronously before a request even starts. These
// ponyfills delegate to the native implementation when present and fall back
// to an AbortController otherwise.

export interface PollSignal {
  signal: AbortSignal;
  dispose: () => void;
}

// Returns a timeout signal paired with a disposer. On engines without native
// AbortSignal.timeout the ponyfill's setTimeout would otherwise pin the
// controller until it fires; callers that settle early MUST dispose to clear it.
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

// Callers MUST invoke dispose() once the request settles so the abort
// listeners don't pile up until the inputs themselves are GC'd.
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

// Rejects the returned promise when *signal* aborts, but never aborts the
// wrapped *promise* itself, so a request shared across callers keeps running
// when one caller's signal fires.
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
