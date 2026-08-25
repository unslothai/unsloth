// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Asking the backend for one of the three Xet notices. The count was in per-origin
// localStorage, and a Studio origin moves whenever port 8888 is taken, so every new
// origin handed out a fresh three and the notice never stopped.

import { authFetch } from "@/features/auth/api";

const LEGACY_COUNT_KEY = "unsloth.studio.xetNoticeCount";
const LEGACY_MIGRATED_KEY = "unsloth.studio.xetNoticeMigrated";

export interface XetNoticeReservation {
  granted: boolean;
  shown: number;
  limit: number;
}

/** A count recorded before the tally moved server-side. Sent once, and only raises
 * the stored value, so someone who spent their three does not get three more. */
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
 * FAILS CLOSED: an unreachable, older or erroring backend shows nothing. Falling back
 * to the browser count would restore the resetting bug on any failed request. */
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
