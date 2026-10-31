// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The notice count lives on the backend: per-origin localStorage reset whenever the port moved.

import { authFetch } from "@/features/auth/api";

const LEGACY_COUNT_KEY = "unsloth.studio.xetNoticeCount";
const LEGACY_MIGRATED_KEY = "unsloth.studio.xetNoticeMigrated";

export interface XetNoticeReservation {
  granted: boolean;
  shown: number;
  limit: number;
}

/** Legacy browser count, sent until the server confirms; it can only raise the stored value. */
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
    return 0;
  }
}

/** Only after the server confirms, or a failed POST would hand out three fresh notices. */
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

/** Fails closed: falling back to the browser count would restore the resetting bug. */
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
      // This POST increments a counter, so a retry could spend a second notice.
      { retryNetworkErrors: false },
    );
    if (!res.ok) return denied;
    const body = (await res.json()) as Partial<XetNoticeReservation>;
    // A proxy can answer an unknown route with 200; only a reply with the cap is genuine.
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
