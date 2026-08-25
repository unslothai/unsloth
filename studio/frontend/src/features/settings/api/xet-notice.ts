// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Asking the backend for one of the three Xet notices.
//
// The count used to live in localStorage, which is per-origin, and a Studio origin is
// not stable: the server falls back past port 8888 when Jupyter already has it, and
// Colab and the tunnel are different origins again. Every one of those handed the user
// a fresh set of three, so the notice never actually stopped.

import { authFetch } from "@/features/auth/api";

const LEGACY_COUNT_KEY = "unsloth.studio.xetNoticeCount";
const LEGACY_MIGRATED_KEY = "unsloth.studio.xetNoticeMigrated";

export interface XetNoticeReservation {
  granted: boolean;
  shown: number;
  limit: number;
}

/** A count this browser recorded before the tally moved server-side.
 *
 * Sent once, and only ever raises the stored count, so someone who already spent
 * their three does not get three more the first time they run this build. Marked as
 * migrated straight away: sending it repeatedly is harmless (the server takes the
 * max) but pointless, and the local value is meaningless afterwards.
 */
function takeLegacyCount(): number {
  if (typeof window === "undefined") return 0;
  try {
    if (window.localStorage.getItem(LEGACY_MIGRATED_KEY)) return 0;
    const raw = window.localStorage.getItem(LEGACY_COUNT_KEY);
    window.localStorage.setItem(LEGACY_MIGRATED_KEY, "1");
    const parsed = Number.parseInt(raw ?? "", 10);
    return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
  } catch {
    // Private mode, or storage disabled. Nothing to migrate.
    return 0;
  }
}

/** Take one of the remaining notices, or report that none are left.
 *
 * FAILS CLOSED. If the endpoint is unreachable, older than these routes, or errors,
 * this reports "not granted" and no toast is shown. The alternative, falling back to
 * the browser count, would reintroduce exactly the resetting behaviour this replaced,
 * every time a request happened to fail. Missing the explanation once is a smaller
 * cost than a notice that comes back forever.
 */
export async function reserveXetNoticeFromServer(): Promise<XetNoticeReservation> {
  const denied: XetNoticeReservation = { granted: false, shown: 0, limit: 0 };
  try {
    const res = await authFetch("/api/settings/xet-notice/reserve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seen_hint: takeLegacyCount() }),
    });
    if (!res.ok) return denied;
    const body = (await res.json()) as Partial<XetNoticeReservation>;
    return {
      granted: body.granted === true,
      shown: typeof body.shown === "number" ? body.shown : 0,
      limit: typeof body.limit === "number" ? body.limit : 0,
    };
  } catch {
    return denied;
  }
}
