// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { checkDiskSpace } from "@/features/settings/low-disk-check";
import { toast } from "@/lib/toast";
import { disposableTimeoutSignal } from "../lib/abort-signals";
import { getActiveModelDownloads } from "./api";
import {
  mismatchStartAction,
  TRANSPORT,
  type ResolvedTransport,
  type TransportMode,
} from "./constants";
import {
  apiTransportStatusWithRetry,
  effectiveTransportMode,
} from "./download-api-adapter";
import type {
  DownloadRequest,
  ManagedDownload,
} from "./download-manager-types";
import {
  findActiveJobForRepo,
  getState,
  hasVariantRepoActivity,
  jobKeyOf,
  repoKeyOf,
  setConflict,
} from "./download-manager-state";
import { startJob } from "./poll-loop";
import {
  currentRoute,
  currentStartToastSelectionEpoch,
  showCallerToast,
} from "./start-toast";
import { runtimeRegistry } from "./runtime-registry";
import { resolveTransportMode } from "./transport-preference";
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

function asTransportMode(value: unknown): ResolvedTransport | null {
  return value === TRANSPORT.HTTP || value === TRANSPORT.XET ? value : null;
}

async function activeSiblingTransport(
  req: DownloadRequest,
): Promise<ResolvedTransport | null> {
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

// "started": a live job exists for this key; "conflict": resolve from the Hub card;
// "busy": a sibling occupies the repo; "error": failed or refused.
export type DownloadStartOutcome = "started" | "conflict" | "busy" | "error";

// A start can no-op without throwing, so derive the outcome from this exact key's job state.
function isJobActiveFor(req: DownloadRequest): boolean {
  const job = getState().jobs[jobKeyOf(req.kind, req.repoId, req.variant)];
  if (!job || !ACTIVE_STATES.has(job.state)) return false;
  return !scopedFileSetDiffers(job, req);
}

// A shared scope slot counts as this transfer only if it fetches the same files; a job with
// no file list is adoptable only for an unscoped request.
function scopedFileSetDiffers(
  job: ManagedDownload,
  req: DownloadRequest,
): boolean {
  if (!req.files || req.files.length === 0) return false;
  if (!job.scopedFiles) return true;
  const live = [...new Set(job.scopedFiles)].sort();
  const wanted = [...new Set(req.files)].sort();
  return (
    live.length !== wanted.length || live.some((f, i) => f !== wanted[i])
  );
}

async function runWithPendingStartGuard(
  req: DownloadRequest,
  action: () => Promise<DownloadStartOutcome>,
): Promise<DownloadStartOutcome> {
  const startKey = pendingStartKey(req);
  if (hasActiveOrPendingStart(req)) {
    if (!isJobActiveFor(req)) return "busy";
    // Returns "started" without running the action, so this is the only feedback.
    showCallerToast(
      jobKeyOf(req.kind, req.repoId, req.variant),
      req.callerToast,
    );
    return "started";
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
  // Read before the preflight round trips, during which the user can navigate.
  const originRoute = currentRoute();
  const originSelectionEpoch = currentStartToastSelectionEpoch();
  // Fire-and-forget, never a gate; the check throttles itself.
  void checkDiskSpace();
  return runWithPendingStartGuard(req, async () => {
    const preferred: TransportMode = await resolveTransportMode();
    let mode: TransportMode = preferred;
    try {
      mode = await effectiveTransportMode(preferred);
    } catch (err) {
      console.warn(
        "Transport capability check failed; using the selected transport.",
        err,
      );
    }
    let siblingTransport: ResolvedTransport | null = null;
    let siblingProbed = false;
    try {
      siblingTransport = await activeSiblingTransport(req);
      siblingProbed = true;
      if (
        siblingTransport &&
        siblingTransport !== mode &&
        preferred !== TRANSPORT.AUTO
      ) {
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
    let restartDisclosure = false;

    try {
      const status = await apiTransportStatusWithRetry(req);
      const last = asTransportMode(status.last_transport);
      const resolved = asTransportMode(mode);
      if (status.has_partial && last && resolved && last !== resolved) {
        const action = mismatchStartAction(
          preferred,
          resolved,
          last,
          status.resumable,
        );
        if (action === "conflict") {
          setConflict(jobKeyOf(req.kind, req.repoId, req.variant), {
            info: {
              previous: last,
              next: resolved,
              resumable: status.resumable,
            },
            // Drop the caller's line: resolved later from the Hub, chat's auto-load promise no longer holds.
            pending: { ...req, callerToast: undefined },
          });
          return "conflict";
        }
        mode = action;
      }
      if (
        status.has_partial &&
        (status.resumable === false || !status.last_transport)
      ) {
        // Do not raise during preflight: the backend may still reject or attach this start.
        restartDisclosure = true;
      }
    } catch (err) {
      console.warn(
        "Transport status check failed; starting without partial-conflict preflight.",
        err,
      );
      // Xet purges partials unconditionally, so an unverifiable partial downgrades this start to HTTP,
      // but only once no sibling variant is live (it may be on Xet).
      if (mode === TRANSPORT.XET && siblingProbed && !siblingTransport) {
        toast.warning("Couldn't verify existing partial download", {
          description:
            "Starting with HTTP so an existing partial is not discarded. Switch transport to retry with Xet.",
        });
        await startJob(req, {
          useXet: false,
          originRoute,
          originSelectionEpoch,
          restartDisclosure,
        });
        return isJobActiveFor(req) ? "started" : "error";
      }
      toast.warning("Couldn't verify existing partial download", {
        description:
          "Starting with the selected transport. If a partial from another transport exists, it may be restarted from the beginning.",
      });
    }
    if (siblingProbed && siblingTransport && siblingTransport !== mode) {
      toast.info("Another variant is already downloading", {
        description:
          siblingTransport === TRANSPORT.XET
            ? "This repository is currently downloading with Xet. Switch to Xet or wait for it to finish."
            : "This repository is currently downloading with HTTP. Switch to HTTP or wait for it to finish.",
      });
      return "busy";
    }

    await startJob(req, {
      useXet: mode === TRANSPORT.XET,
      originRoute,
      originSelectionEpoch,
      restartDisclosure,
    });
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
