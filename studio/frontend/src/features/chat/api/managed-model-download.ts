// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ManagedModelDownloadResult =
  | "complete"
  | "cancelled"
  | "conflict"
  | "busy"
  | "error";

export type ManagedModelDownloadRequest = {
  repoId: string;
  variant: string | null;
  expectedBytes: number;
};

type ManagedDownloadRequest = ManagedModelDownloadRequest & { kind: "model" };
type DownloadStartOutcome =
  | "started"
  | "existing"
  | "conflict"
  | "busy"
  | "error";
type JobListeners = {
  onComplete?: (variant: string | null, bytes: number) => unknown;
  onCancelled?: (variant: string | null) => unknown;
  onError?: (variant: string | null) => unknown;
};

export interface ManagedModelDownloadDependencies {
  requestStart: (
    request: ManagedDownloadRequest,
  ) => Promise<DownloadStartOutcome>;
  cancel: (key: string) => void | Promise<void>;
  subscribe: (
    kind: "model",
    repoId: string,
    listeners: JobListeners,
  ) => () => void;
  jobKey: (kind: "model", repoId: string, variant: string | null) => string;
}

type SharedDownloadLifecycle = {
  consumers: Set<symbol>;
  startPromise: Promise<DownloadStartOutcome> | null;
  startSettled: boolean;
  owned: boolean;
  cancelWhenOwned: boolean;
  cancellation: Promise<void> | null;
};

const sharedDownloads = new Map<string, SharedDownloadLifecycle>();

function sameVariant(left: string | null, right: string | null): boolean {
  return (
    (left?.trim().toLowerCase() ?? "") === (right?.trim().toLowerCase() ?? "")
  );
}

function abortReason(signal: AbortSignal): unknown {
  return (
    signal.reason ?? new DOMException("Model download cancelled.", "AbortError")
  );
}

function waitForPromiseOrAbort(
  promise: Promise<void>,
  signal: AbortSignal,
): Promise<void> {
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
      () => {
        cleanup();
        resolve();
      },
      () => {
        cleanup();
        resolve();
      },
    );
  });
}

function beginCancellation(
  key: string,
  shared: SharedDownloadLifecycle,
  dependencies: ManagedModelDownloadDependencies,
): Promise<void> {
  if (shared.cancellation) {
    return shared.cancellation;
  }
  const cancellation = Promise.resolve(dependencies.cancel(key))
    .catch(() => undefined)
    .then(() => undefined)
    .finally(() => {
      if (sharedDownloads.get(key) === shared) {
        sharedDownloads.delete(key);
      }
    });
  shared.cancellation = cancellation;
  return cancellation;
}

/**
 * Start (or attach to) a Download Manager model job and wait for its terminal
 * event. Concurrent consumers of the exact job share ownership: one consumer
 * leaving cannot cancel work another still needs, and a successor cannot start
 * until cancellation of the previous owned generation has finished.
 */
export async function coordinateManagedModelDownload(
  request: ManagedModelDownloadRequest,
  signal: AbortSignal,
  dependencies: ManagedModelDownloadDependencies,
): Promise<ManagedModelDownloadResult> {
  const managedRequest: ManagedDownloadRequest = { kind: "model", ...request };
  const key = dependencies.jobKey(
    managedRequest.kind,
    managedRequest.repoId,
    managedRequest.variant,
  );
  const prior = sharedDownloads.get(key);
  if (prior?.cancellation) {
    await waitForPromiseOrAbort(prior.cancellation, signal);
    return coordinateManagedModelDownload(request, signal, dependencies);
  }

  const existing = sharedDownloads.get(key);
  const shared: SharedDownloadLifecycle = existing ?? {
    consumers: new Set<symbol>(),
    startPromise: null,
    startSettled: false,
    owned: false,
    cancelWhenOwned: false,
    cancellation: null,
  };
  if (!existing) {
    sharedDownloads.set(key, shared);
  }
  const consumer = Symbol(key);
  shared.consumers.add(consumer);

  return new Promise((resolve, reject) => {
    let settled = false;
    let unsubscribe: (() => void) | null = null;

    const cleanup = () => {
      signal.removeEventListener("abort", onAbort);
      unsubscribe?.();
      unsubscribe = null;
    };
    const detach = (cancelIfLast: boolean): Promise<void> | null => {
      shared.consumers.delete(consumer);
      if (shared.consumers.size > 0) {
        return null;
      }
      if (!shared.startSettled) {
        shared.cancelWhenOwned ||= cancelIfLast;
        return null;
      }
      if (cancelIfLast && shared.owned) {
        return beginCancellation(key, shared, dependencies);
      }
      if (sharedDownloads.get(key) === shared) {
        sharedDownloads.delete(key);
      }
      return null;
    };
    const settle = (result: ManagedModelDownloadResult) => {
      if (settled) return;
      settled = true;
      cleanup();
      detach(false);
      resolve(result);
    };
    const fail = (error: unknown) => {
      if (settled) return;
      settled = true;
      cleanup();
      detach(false);
      reject(error);
    };
    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      const cancellation = detach(true);
      if (cancellation) {
        cancellation.finally(() => reject(abortReason(signal)));
      } else {
        reject(abortReason(signal));
      }
    };

    if (signal.aborted) {
      fail(abortReason(signal));
      return;
    }

    unsubscribe = dependencies.subscribe(
      managedRequest.kind,
      managedRequest.repoId,
      {
        onComplete: (variant) => {
          if (sameVariant(variant, managedRequest.variant)) settle("complete");
        },
        onCancelled: (variant) => {
          if (sameVariant(variant, managedRequest.variant)) settle("cancelled");
        },
        onError: (variant) => {
          if (sameVariant(variant, managedRequest.variant)) settle("error");
        },
      },
    );
    signal.addEventListener("abort", onAbort, { once: true });

    shared.startPromise ??= dependencies.requestStart(managedRequest);
    shared.startPromise
      .then((outcome) => {
        shared.startSettled = true;
        shared.owned = outcome === "started";
        if (shared.consumers.size === 0) {
          if (shared.owned && shared.cancelWhenOwned) {
            beginCancellation(key, shared, dependencies);
          } else if (sharedDownloads.get(key) === shared) {
            sharedDownloads.delete(key);
          }
        }
        if (!settled && outcome !== "started" && outcome !== "existing") {
          settle(outcome);
        }
      })
      .catch(() => {
        shared.startSettled = true;
        shared.owned = false;
        if (
          shared.consumers.size === 0 &&
          sharedDownloads.get(key) === shared
        ) {
          sharedDownloads.delete(key);
        }
        settle("error");
      });
  });
}
