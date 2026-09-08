// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Asking the backend for one of the three Xet notices. The count was in per-origin
// localStorage, and an Unsloth origin moves whenever port 8888 is taken, so every new
// origin handed out a fresh three and the notice never stopped.

import { authFetch } from "@/features/auth/api";

const LEGACY_COUNT_KEY = "unsloth.studio.xetNoticeCount";
const LEGACY_MIGRATED_KEY = "unsloth.studio.xetNoticeMigrated";

export interface XetNoticeReservation {
  granted: boolean;
  shown: number;
  limit: number;
}

/** A count recorded before the tally moved server-side. Sent until the server
 * confirms it, and it can only raise the stored value. */
function readLegacyCount(): number {
  if (typeof window === "undefined") return 0;
  try {
    if (window.localStorage.getItem(LEGACY_MIGRATED_KEY)) return 0;
    const parsed = Number.parseInt(
      window.localStorage.getItem(LEGACY_COUNT_KEY) ?? "",
      10,
    );
    return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
  } catch {
    // Private mode, or storage disabled. Nothing to migrate.
    return 0;
  }
}

/** Only once a response proves the server took the hint. Marking it on the way out
 * dropped the floor on any failed POST, handing three fresh notices to someone who
 * had already spent them. */
function markLegacyMigrated(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LEGACY_MIGRATED_KEY, "1");
  } catch {
    // Nothing to record it in; the hint is re-sent next time, which is harmless.
  }
}

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

/** Take one of the remaining notices, or report that none are left.
 *
 * FAILS CLOSED: an unreachable, older or erroring backend shows nothing. Falling back
 * to the browser count would restore the resetting bug on any failed request. */
export async function reserveXetNoticeFromServer(): Promise<XetNoticeReservation> {
  const denied: XetNoticeReservation = { granted: false, shown: 0, limit: 0 };
  try {
    const res = await authFetch(
      "/api/settings/xet-notice/reserve",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seen_hint: readLegacyCount() }),
      },
      // This POST increments a counter, so a retry whose predecessor reached the
      // backend spends a second notice. Every other mutation here opts out too.
      { retryNetworkErrors: false },
    );
    if (!res.ok) return denied;
    const body = (await res.json()) as Partial<XetNoticeReservation>;
    // A proxy or older backend can answer this unknown route with a 200 and other
    // JSON, so `res.ok` is no proof the hint was stored. Only a reply reporting the
    // cap it enforced came from the reservation endpoint.
    if (typeof body.granted !== "boolean" || !isCount(body.limit)) return denied;
    markLegacyMigrated();
    return {
      granted: body.granted,
      shown: isCount(body.shown) ? body.shown : 0,
      limit: body.limit,
    };
  } catch {
    return denied;
  }
}
