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

async function runWithPendingStartGuard(
  req: DownloadRequest,
  action: () => Promise<void>,
): Promise<void> {
  const startKey = pendingStartKey(req);
  if (hasActiveOrPendingStart(req)) return;
  runtimeRegistry.pendingStartRepoKeys.add(startKey);
  try {
    await action();
  } catch (error) {
    reportConflictStartError(error);
  } finally {
    runtimeRegistry.pendingStartRepoKeys.delete(startKey);
  }
}

export async function requestStart(req: DownloadRequest): Promise<void> {
  await runWithPendingStartGuard(req, async () => {
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
        return;
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
        return;
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
      // Fail safe, not open: Xet purges any partial unconditionally, so when we
      // could not verify the partial we downgrade this one start to HTTP, which
      // resumes an HTTP partial and is harmless for a fresh download. The user
      // keeps their Xet preference for the next attempt. Only downgrade once we
      // positively confirmed no sibling variant is downloading: a live sibling
      // (or one we couldn't probe) means the repo may be mid-transfer on Xet,
      // and HTTP would only force a transport conflict.
      if (mode === TRANSPORT.XET && siblingProbed && !siblingTransport) {
        toast.warning("Couldn't verify existing partial download", {
          description:
            "Starting with HTTP so an existing partial is not discarded. Switch transport to retry with Xet.",
        });
        await startJob(req, { useXet: false });
        return;
      }
      toast.warning("Couldn't verify existing partial download", {
        description:
          "Starting with the selected transport. If a partial from another transport exists, it may be restarted from the beginning.",
      });
    }
    await startJob(req, { useXet: mode === TRANSPORT.XET });
  });
}

export function resumeConflict(conflictKey: string): void {
  const entry = getState().conflicts[conflictKey];
  if (!entry) return;
  setConflict(conflictKey, null);
  void runWithPendingStartGuard(entry.pending, () =>
    startJob(entry.pending, {
      useXet: entry.info.previous === TRANSPORT.XET,
    }),
  );
}

export function restartConflict(conflictKey: string): void {
  const entry = getState().conflicts[conflictKey];
  if (!entry) return;
  setConflict(conflictKey, null);
  void runWithPendingStartGuard(entry.pending, () =>
    startJob(entry.pending, {
      useXet: entry.info.next === TRANSPORT.XET,
    }),
  );
}

export function cancelConflict(conflictKey: string): void {
  setConflict(conflictKey, null);
}
