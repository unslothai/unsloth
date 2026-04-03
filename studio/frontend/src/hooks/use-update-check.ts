// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

export interface UpdateCheckInfo {
  critical: boolean;
  announcementBadge: string | null;
  announcementMessage: string | null;
  announcementUrl: string | null;
  manifestFetched: boolean;
}

const DEFAULT: UpdateCheckInfo = {
  critical: false,
  announcementBadge: null,
  announcementMessage: null,
  announcementUrl: null,
  manifestFetched: false,
};

// Module-level cache so multiple components share one fetch.
let cached: UpdateCheckInfo | null = null;
let fetchPromise: Promise<UpdateCheckInfo> | null = null;

async function fetchOnce(): Promise<UpdateCheckInfo> {
  if (cached) return cached;
  if (fetchPromise) return fetchPromise;

  fetchPromise = (async () => {
    try {
      // Unauthenticated endpoint -- use plain fetch, not authFetch.
      const res = await fetch("/api/update-check");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const info: UpdateCheckInfo = {
        critical: data?.critical ?? false,
        announcementBadge: data?.announcement_badge ?? null,
        announcementMessage: data?.announcement_message ?? null,
        announcementUrl: data?.announcement_url ?? null,
        manifestFetched: data?.manifest_fetched ?? false,
      };
      cached = info;
      return info;
    } catch {
      fetchPromise = null;
      return DEFAULT;
    }
  })();

  return fetchPromise;
}

/**
 * Fetch update-check info from `GET /api/update-check`.
 *
 * The result is cached at module level -- only one network request is made
 * regardless of how many components call this hook.
 */
export function useUpdateCheck(): UpdateCheckInfo {
  const [info, setInfo] = useState<UpdateCheckInfo>(cached ?? DEFAULT);

  useEffect(() => {
    if (cached) return;

    let cancelled = false;
    fetchOnce().then((status) => {
      if (!cancelled) setInfo(status);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return info;
}
