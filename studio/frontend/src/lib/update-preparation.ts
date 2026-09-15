// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type PrefetchStatus, sameUpdateVersion } from "@/lib/tauri-updater";

export type ShellPreparation = "pending" | "downloading" | "done" | "failed";

/** `skipped` and `failed` are fine outcomes: the prefetch only warms a cache. */
export type BackendPreparation =
  | "pending"
  | "prefetching"
  | "ready"
  | "failed"
  | "skipped";

export interface UpdatePreparation {
  shell: ShellPreparation;
  backend: BackendPreparation;
  shellProgress: number;
}

export type PreparationStatus = "preparing" | "ready" | "available";

export const INITIAL_PREPARATION: UpdatePreparation = {
  shell: "pending",
  backend: "pending",
  shellProgress: 0,
};

/** A failed bundle download puts the offer back to "available"; the backend half never blocks. */
export function preparationStatus(
  preparation: UpdatePreparation,
): PreparationStatus {
  if (preparation.shell === "failed") return "available";
  if (preparation.shell !== "done") return "preparing";
  switch (preparation.backend) {
    case "ready":
    case "failed":
    case "skipped":
      return "ready";
    default:
      return "preparing";
  }
}

export type DesktopDownloadDecision = "ready" | "wait" | "download";

export function desktopDownloadDecision(
  status: { version: string | null; downloaded: boolean; downloading: boolean },
  offeredVersion: string,
): DesktopDownloadDecision {
  if (status.downloaded && sameUpdateVersion(status.version, offeredVersion))
    return "ready";
  return status.downloading ? "wait" : "download";
}

/**
 * `prefetch` start one; `adopt` join the run already preparing this offer;
 * `restart` stop a run preparing an older offer and start over; `already-ready`
 * the cache is warm for this offer; `skip` nothing to prepare here.
 */
export type PrefetchDecision =
  | "prefetch"
  | "adopt"
  | "restart"
  | "already-ready"
  | "skip";

export function prefetchDecision(args: {
  inApp: boolean;
  isExternalServer: boolean;
  offeredVersion: string;
  prefetch: PrefetchStatus;
}): PrefetchDecision {
  if (!args.inApp || args.isExternalServer) return "skip";
  const prefetch = args.prefetch;
  if (prefetch.running) {
    return sameUpdateVersion(prefetch.runningShellVersion, args.offeredVersion)
      ? "adopt"
      : "restart";
  }
  const usable =
    prefetch.state === "ready" ||
    prefetch.state === "noop" ||
    prefetch.state === "partial";
  if (usable && sameUpdateVersion(prefetch.shellVersion, args.offeredVersion))
    return "already-ready";
  return "prefetch";
}

export function preparationShortLabel(preparation: UpdatePreparation): string {
  if (preparation.shell === "downloading")
    return `downloading ${preparation.shellProgress}%`;
  if (preparation.backend === "prefetching") return "preparing packages";
  return "starting";
}
