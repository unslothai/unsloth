// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { disposableTimeoutSignal } from "../lib/abort-signals";
import { getActiveModelDownloads } from "./api";
import { TRANSPORT, type TransportMode } from "./constants";
import {
  apiTransportStatusWithRetry,
  effectiveTransportMode,
} from "./download-api-adapter";
import type { DownloadRequest } from "./download-manager-types";
import {
  findActiveJobForRepo,
  getState,
  hasVariantRepoActivity,
  jobKeyOf,
  repoKeyOf,
  setConflict,
} from "./download-manager-state";
import { startJob } from "./poll-loop";
import { runtimeRegistry } from "./runtime-registry";
import { getTransportMode } from "./transport-preference";
import { ACTIVE_STATES, TRANSPORT_STATUS_TIMEOUT_MS } from "./download-manager-config";

function reportConflictStartError(error: unknown): void {
  const description =
    error instanceof Error ? error.message : String(error || "Unknown error");
  toast.error("Couldn't start download", { description });
}

function pendingStartKey(req: DownloadRequest): string {
  return jobKeyOf(req.kind, req.repoId, req.variant);
}

function hasPendingStartForRepo(repoKey: string): boolean {
  for (const key of runtimeRegistry.pendingStartRepoKeys) {
    if (key === repoKey || key.startsWith(`${repoKey}#`)) return true;
  }
  return false;
}

function hasActiveOrPendingStart(req: DownloadRequest): boolean {
  const key = pendingStartKey(req);
  if (req.kind === "model" && req.variant) {
    return hasVariantRepoActivity(req.kind, req.repoId, key, {
      includeOwnRuntime: true,
      includePending: true,
    });
  }
  if (runtimeRegistry.pendingStartRepoKeys.has(key)) return true;
  const repoKey = repoKeyOf(req.kind, req.repoId);
  if (hasPendingStartForRepo(repoKey)) return true;
  return (
    Boolean(findActiveJobForRepo(getState().jobs, req.kind, req.repoId)) ||
    Boolean(runtimeRegistry.runtimes.get(repoKey))
  );
}

function asTransportMode(value: unknown): TransportMode | null {
  return value === TRANSPORT.HTTP || value === TRANSPORT.XET ? value : null;
}

async function activeSiblingTransport(
  req: DownloadRequest,
): Promise<TransportMode | null> {
  if (req.kind !== "model" || !req.variant) return null;
  const timeout = disposableTimeoutSignal(TRANSPORT_STATUS_TIMEOUT_MS);
  const downloads = await getActiveModelDownloads(req.repoId, timeout.signal, {
    fresh: true,
  }).finally(() => timeout.dispose());
  const variant = req.variant.trim().toLowerCase();
  for (const download of downloads) {
    const siblingVariant = download.variant?.trim().toLowerCase();
    if (!siblingVariant || siblingVariant === variant) continue;
    if (!ACTIVE_STATES.has(download.state)) continue;
    const transport = asTransportMode(download.transport);
    if (transport) return transport;
  }
  return null;
}

// Outcome of a start request so callers can tell whether a transfer for this
// exact request is actually live before telling the user it began. "started"
// means a running/cancelling job exists for this key (a fresh start or an
// already-active one). "conflict" means a transport partial conflict was
// recorded and must be resolved from the Hub download card; "busy" means the
// repo is occupied by a sibling variant/snapshot/pending start that is not this
// transfer; "error" means the start failed or was refused.
export type DownloadStartOutcome = "started" | "conflict" | "busy" | "error";

// A start can no-op without throwing: the backend can refuse it (startJob
// finalizes "error"), startJob's peer guard can skip it, or
// hasActiveOrPendingStart can trip on a snapshot/peer/pending that is not this
// request. Derive the outcome from the actual job state of this exact key so
// callers never claim a download began when it did not.
function isJobActiveFor(req: DownloadRequest): boolean {
  const job = getState().jobs[jobKeyOf(req.kind, req.repoId, req.variant)];
  return Boolean(job && ACTIVE_STATES.has(job.state));
}

async function runWithPendingStartGuard(
  req: DownloadRequest,
  action: () => Promise<DownloadStartOutcome>,
): Promise<DownloadStartOutcome> {
  const startKey = pendingStartKey(req);
  // Already active or pending for the repo: only report "started" when this
  // exact request is the live transfer; a peer/snapshot/pending start has not.
  if (hasActiveOrPendingStart(req)) {
    return isJobActiveFor(req) ? "started" : "busy";
  }
  runtimeRegistry.pendingStartRepoKeys.add(startKey);
  try {
    return await action();
  } catch (error) {
    reportConflictStartError(error);
    return "error";
  } finally {
    runtimeRegistry.pendingStartRepoKeys.delete(startKey);
  }
}

export async function requestStart(
  req: DownloadRequest,
): Promise<DownloadStartOutcome> {
  return runWithPendingStartGuard(req, async () => {
    let mode: TransportMode = getTransportMode();
    try {
      mode = await effectiveTransportMode(mode);
    } catch (err) {
      console.warn(
        "Transport capability check failed; using the selected transport.",
        err,
      );
    }
    let siblingTransport: TransportMode | null = null;
    let siblingProbed = false;
    try {
      siblingTransport = await activeSiblingTransport(req);
      siblingProbed = true;
      if (siblingTransport && siblingTransport !== mode) {
        toast.info("Another variant is already downloading", {
          description:
            siblingTransport === TRANSPORT.XET
              ? "This repository is currently downloading with Xet. Switch to Xet or wait for it to finish."
              : "This repository is currently downloading with HTTP. Switch to HTTP or wait for it to finish.",
        });
        return "busy";
      }
    } catch (err) {
      console.warn("Active download transport check failed.", err);
    }
    try {
      const status = await apiTransportStatusWithRetry(req);
      if (
        status.has_partial &&
        status.last_transport &&
        status.last_transport !== mode
      ) {
        setConflict(jobKeyOf(req.kind, req.repoId, req.variant), {
          info: {
            previous: status.last_transport,
            next: mode,
            resumable: status.resumable,
          },
          pending: req,
        });
        return "conflict";
      }
      if (status.has_partial && !status.last_transport) {
        toast.info("Restarting this download", {
          description:
            "An earlier partial download can't be resumed, so it will start again from the beginning.",
        });
      }
    } catch (err) {
      console.warn(
        "Transport status check failed; starting without partial-conflict preflight.",
        err,
      );
      // Fail safe: Xet purges any partial unconditionally, so when the partial
      // can't be verified we downgrade this one start to HTTP (resumes an HTTP
      // partial, harmless for a fresh download); the Xet preference is kept for
      // next time. Only downgrade once we confirmed no sibling variant is
      // downloading, since a live sibling may be mid-transfer on Xet.
      if (mode === TRANSPORT.XET && siblingProbed && !siblingTransport) {
        toast.warning("Couldn't verify existing partial download", {
          description:
            "Starting with HTTP so an existing partial is not discarded. Switch transport to retry with Xet.",
        });
        await startJob(req, { useXet: false });
        return isJobActiveFor(req) ? "started" : "error";
      }
      toast.warning("Couldn't verify existing partial download", {
        description:
          "Starting with the selected transport. If a partial from another transport exists, it may be restarted from the beginning.",
      });
    }
    await startJob(req, { useXet: mode === TRANSPORT.XET });
    return isJobActiveFor(req) ? "started" : "error";
  });
}

export function resumeConflict(conflictKey: string): void {
  const entry = getState().conflicts[conflictKey];
  if (!entry) return;
  setConflict(conflictKey, null);
  void runWithPendingStartGuard(entry.pending, async () => {
    await startJob(entry.pending, {
      useXet: entry.info.previous === TRANSPORT.XET,
    });
    return "started";
  });
}

export function restartConflict(conflictKey: string): void {
  const entry = getState().conflicts[conflictKey];
  if (!entry) return;
  setConflict(conflictKey, null);
  void runWithPendingStartGuard(entry.pending, async () => {
    await startJob(entry.pending, {
      useXet: entry.info.next === TRANSPORT.XET,
    });
    return "started";
  });
}

export function cancelConflict(conflictKey: string): void {
  setConflict(conflictKey, null);
}
