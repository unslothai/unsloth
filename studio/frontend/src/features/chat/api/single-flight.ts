// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

function abortReason(signal: AbortSignal): unknown {
  return (
    signal.reason ?? new DOMException("Operation cancelled.", "AbortError")
  );
}

function waitForPromiseOrAbort<T>(
  promise: Promise<T>,
  signal: AbortSignal,
): Promise<T> {
  if (signal.aborted) {
    return Promise.reject(abortReason(signal));
  }
  return new Promise((resolve, reject) => {
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    const onAbort = () => {
      cleanup();
      reject(abortReason(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      },
    );
  });
}

export type SingleFlight<T> = {
  run: (signal: AbortSignal, start: () => Promise<T>) => Promise<T>;
};

export function createSingleFlight<T>(): SingleFlight<T> {
  let active: Promise<T> | null = null;

  return {
    run(signal, start) {
      if (signal.aborted) {
        return Promise.reject(abortReason(signal));
      }
      if (!active) {
        let created: Promise<T>;
        try {
          created = Promise.resolve(start());
        } catch (error) {
          created = Promise.reject(error);
        }
        active = created;
        const clear = () => {
          if (active === created) {
            active = null;
          }
        };
        created.then(clear, clear);
      }
      return waitForPromiseOrAbort(active, signal);
    },
  };
}
