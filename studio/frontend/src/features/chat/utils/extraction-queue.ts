// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";

// Module-level FIFO gate mirroring the backend `_EXTRACT_SEMAPHORE`
// (default 2) so the frontend never outruns the worker pool (503 busy).
// The limit is re-read from the chat store at acquire/release time.

let activeCount = 0;
let backendLimit: number | null = null;
const waitQueue: Array<() => void> = [];

function getLimit(): number {
  const value = useChatRuntimeStore.getState().docExtract.extractConcurrency;
  const requested = Number.isFinite(value) && value > 0 ? Math.floor(value) : 1;
  return backendLimit === null ? requested : Math.min(requested, backendLimit);
}

function pump(): void {
  while (activeCount < getLimit() && waitQueue.length > 0) {
    const next = waitQueue.shift()!;
    activeCount += 1;
    next();
  }
}

export function setExtractionBackendLimit(value: number | null | undefined): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    backendLimit = null;
  } else {
    backendLimit = Math.max(1, Math.floor(value));
  }
  pump();
}

/**
 * Reserve an extraction slot. Resolves with a `release` function that must
 * be called exactly once (use try/finally); rejects with an AbortError
 * DOMException if `signal` aborts while waiting.
 */
export function acquireExtractionSlot(
  signal?: AbortSignal,
): Promise<() => void> {
  return new Promise<() => void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }

    let granted = false;
    let released = false;

    const release = (): void => {
      if (released) return;
      released = true;
      activeCount -= 1;
      pump();
    };

    const grant = (): void => {
      granted = true;
      if (signal) signal.removeEventListener("abort", onAbort);
      resolve(release);
    };

    const onAbort = (): void => {
      if (granted) return;
      const idx = waitQueue.indexOf(grant);
      if (idx !== -1) waitQueue.splice(idx, 1);
      reject(new DOMException("Aborted", "AbortError"));
    };

    if (signal) signal.addEventListener("abort", onAbort, { once: true });

    if (activeCount < getLimit()) {
      activeCount += 1;
      grant();
    } else {
      waitQueue.push(grant);
    }
  });
}
