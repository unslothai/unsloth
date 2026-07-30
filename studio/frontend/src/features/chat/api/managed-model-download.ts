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
type DownloadStartOutcome = "started" | "conflict" | "busy" | "error";
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

/**
 * Start (or attach to) a Download Manager model job and wait for its terminal
 * event. The listener is registered before requestStart so a fast backend
 * completion cannot be missed. Aborting the chat cancels the exact managed job.
 */
export function coordinateManagedModelDownload(
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

  return new Promise((resolve, reject) => {
    let settled = false;
    let unsubscribe: (() => void) | null = null;

    const cleanup = () => {
      signal.removeEventListener("abort", onAbort);
      unsubscribe?.();
      unsubscribe = null;
    };
    const settle = (result: ManagedModelDownloadResult) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve(result);
    };
    const fail = (error: unknown) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(error);
    };
    const onAbort = () => {
      Promise.resolve(dependencies.cancel(key)).catch(() => undefined);
      fail(abortReason(signal));
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
          if (sameVariant(variant, managedRequest.variant)) {
            settle("complete");
          }
        },
        onCancelled: (variant) => {
          if (sameVariant(variant, managedRequest.variant)) {
            settle("cancelled");
          }
        },
        onError: (variant) => {
          if (sameVariant(variant, managedRequest.variant)) {
            settle("error");
          }
        },
      },
    );
    signal.addEventListener("abort", onAbort, { once: true });

    dependencies
      .requestStart(managedRequest)
      .then((outcome) => {
        // The abort may have landed while requestStart was still running its
        // async preflight, before a cancellable job existed. If that preflight
        // subsequently starts this job, cancel it again now that it is live.
        if (signal.aborted) {
          if (outcome === "started") {
            Promise.resolve(dependencies.cancel(key)).catch(() => undefined);
          }
          return;
        }
        if (outcome !== "started") {
          settle(outcome);
        }
      })
      .catch(() => settle("error"));
  });
}
