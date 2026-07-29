// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which rows of a monitor snapshot are traffic this session has not shown yet.
// Split out of the overlay so it can be driven without a browser.

import type { ApiMonitorEntry } from "@/features/chat/types/api";

export type WatchedEntry = Pick<
  ApiMonitorEntry,
  "id" | "status" | "via_api_key" | "started_at"
>;

export interface WatchedResponse {
  entries: readonly WatchedEntry[];
  // biome-ignore lint/style/useNamingConvention: API schema
  server_time?: number | null;
}

export interface ApiMonitorWatch {
  /** The first snapshot has been folded in and its backlog written off. */
  seeded: boolean;
  /** Ids already shown. A set, not "the newest id": finishing moves an entry to
   * the front, so the head flips without any new traffic. */
  seenIds: Set<string>;
  /** performance.now() when this watch began; monotonic, so a client clock step
   * mid-session cannot move it. */
  watchStartedAt: number;
}

export function createWatch(nowMs: number): ApiMonitorWatch {
  return { seeded: false, seenIds: new Set(), watchStartedAt: nowMs };
}

/**
 * Re-anchor as the poll stands up, and only while still unseeded: the first
 * snapshot can land long after the overlay mounted (a hidden tab issues no
 * fetch, a backend still coming up fails one), and dating the backlog from
 * mount would write that whole gap off as history.
 */
export function startWatching(watch: ApiMonitorWatch, nowMs: number): void {
  if (!watch.seeded) {
    watch.watchStartedAt = nowMs;
  }
}

/** The full page took over; what it showed is not new traffic on the way back. */
export function rearmWatch(watch: ApiMonitorWatch): void {
  watch.seeded = false;
}

/**
 * When this watch began, on the server's clock.
 *
 * The server's own ``time.time()`` minus a browser *duration*, never minus a
 * browser timestamp, so a browser clock that disagrees with the server's -- a
 * Studio behind a tunnel, in a container, on a host that has not run NTP --
 * cancels instead of skewing the answer. Null on a backend with no clock field,
 * which keeps the old behaviour.
 */
function historyCutoff(
  watch: ApiMonitorWatch,
  response: WatchedResponse,
  nowMs: number,
): number | null {
  const serverTime = response.server_time;
  if (typeof serverTime !== "number" || !Number.isFinite(serverTime)) {
    return null;
  }
  return serverTime - Math.max(0, nowMs - watch.watchStartedAt) / 1000;
}

function isHistory(entry: WatchedEntry, cutoff: number | null): boolean {
  // Still running at the first snapshot: it started while Studio was loading, so
  // it is unseen live traffic.
  if (entry.status === "running") {
    return false;
  }
  if (cutoff == null || !Number.isFinite(entry.started_at)) {
    return true;
  }
  // Finished before the first snapshot is not the same as started before we did:
  // a call made while the tab was hidden is already terminal when the poll
  // finally runs, and writing it off is how the panel misses the first request.
  return entry.started_at <= cutoff;
}

/**
 * Fold a snapshot in and report whether it holds API-key traffic not shown yet.
 */
export function observeResponse(
  watch: ApiMonitorWatch,
  response: WatchedResponse,
  nowMs: number,
): boolean {
  const { entries } = response;
  if (!watch.seeded) {
    watch.seeded = true;
    const cutoff = historyCutoff(watch, response, nowMs);
    watch.seenIds = new Set(
      entries.filter((entry) => isHistory(entry, cutoff)).map((e) => e.id),
    );
  }
  const seen = watch.seenIds;
  // Only API-key traffic counts: Studio's own chat uses these same endpoints, and
  // this panel is about serving other clients.
  const hasNewTraffic = entries.some(
    (entry) => entry.via_api_key && !seen.has(entry.id),
  );
  // Re-seed each poll so the set stays bounded by the server's ring buffer.
  watch.seenIds = new Set(entries.map((entry) => entry.id));
  return hasNewTraffic;
}
