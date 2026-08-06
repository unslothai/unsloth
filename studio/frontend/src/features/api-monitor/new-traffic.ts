


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
  /** Ids already shown. A set, not "the newest id": finishing moves an entry to the
   * front, so the head flips without any new traffic. */
  seenIds: Set<string>;
  /** performance.now() when this watch began; monotonic, so a clock step cannot move it. */
  watchStartedAt: number;
  /** This seed follows a stay on the full page rather than starting a session. */
  resumed: boolean;
  /** The snapshot already folded in. The overlay's observer re-runs on any store change,
   * not only on a new poll, and a second fold would spend a pending re-arm on a snapshot
   * from before the stand down. */
  lastFolded: WatchedResponse | null;
}

export function createWatch(nowMs: number): ApiMonitorWatch {
  return {
    seeded: false,
    seenIds: new Set(),
    watchStartedAt: nowMs,
    resumed: false,
    lastFolded: null,
  };
}

/**
 * Re-anchor as the poll stands up, and only while unseeded: the first snapshot can land
 * long after mount (a hidden tab issues no fetch), and dating the backlog from mount
 * would write that whole gap off as history.
 */
export function startWatching(watch: ApiMonitorWatch, nowMs: number): void {
  if (!watch.seeded) {
    watch.watchStartedAt = nowMs;
  }
}

/** The full page took over; what it showed is not new traffic on the way back. */
export function rearmWatch(watch: ApiMonitorWatch): void {
  watch.seeded = false;
  watch.resumed = true;
}

/**
 * The poll also stands down for the auto-open opt out, and calls still land behind it.
 * By the time the user turns automatic opening back on those are backlog, not a reason to
 * pop the panel over the composer, so write them off exactly as a stay on the full page
 * does. An unseeded watch has no backlog to write off: re-arming it would silence the
 * first snapshot of a session that merely started out opted out.
 */
export function standDownWatch(watch: ApiMonitorWatch): void {
  if (!watch.seeded) {
    return;
  }
  rearmWatch(watch);
}

/**
 * When this watch began, on the server's clock. Server ``time.time()`` minus a browser
 * *duration*, never minus a browser timestamp, so a clock disagreeing with the server's cancels
 * instead of skewing the answer. Null on a backend with no clock field.
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
  // Still running at the first snapshot: it started while Studio loaded, so it is unseen.
  if (entry.status === "running") {
    return false;
  }
  if (cutoff == null || !Number.isFinite(entry.started_at)) {
    return true;
  }
  // Finished before the first snapshot is not the same as started before we did: a call
  // made while the tab was hidden is already terminal when the poll finally runs.
  return entry.started_at <= cutoff;
}

/** Fold a snapshot in and report whether it holds API-key traffic not shown yet. */
export function observeResponse(
  watch: ApiMonitorWatch,
  response: WatchedResponse,
  nowMs: number,
): boolean {
  // Fold each snapshot once. The observer re-runs when the store changes, with the same
  // snapshot in hand; folding it again would seed a re-arm from before the stand down and
  // leave the backlog that landed behind it counting as new traffic.
  if (watch.lastFolded === response) {
    return false;
  }
  watch.lastFolded = response;
  const { entries } = response;
  if (!watch.seeded) {
    watch.seeded = true;
    const { resumed } = watch;
    watch.resumed = false;
    const cutoff = historyCutoff(watch, response, nowMs);
    // A rearm is not a fresh watch. isHistory keeps a running row on purpose: at a first
    // snapshot nobody has seen it. Off the full page the opposite holds, since that page
    // was showing this same feed, so mark everything it could show as read.
    watch.seenIds = new Set(
      entries
        .filter((entry) => resumed || isHistory(entry, cutoff))
        .map((e) => e.id),
    );
  }
  const seen = watch.seenIds;
  // Only API-key traffic counts: Studio's own chat uses these same endpoints.
  const hasNewTraffic = entries.some(
    (entry) => entry.via_api_key && !seen.has(entry.id),
  );
  // Re-seed each poll so the set stays bounded by the server's ring buffer.
  watch.seenIds = new Set(entries.map((entry) => entry.id));
  return hasNewTraffic;
}
