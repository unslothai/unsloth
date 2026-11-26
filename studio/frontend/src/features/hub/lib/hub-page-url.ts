// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { loadHubSettings } from "@/features/settings/api/hub-settings";
import { setHubPageBase, useHubWebUrl, useMissingHubPageBase } from "@/lib/hf-endpoint";

// Nothing is kept past a lookup: the store holds the current mirror's base, and links that
// mount together share one request.
const pending = new Set<string>();

/** `/api/health` hides an endpoint saved in Settings; the owner route names it where this browser
 * can reach it. Others keep the relay. */
async function loadSavedPageBase(endpoint: string): Promise<void> {
  if (pending.has(endpoint)) return;
  pending.add(endpoint);
  try {
    const { pageBase, pageBaseRelay } = await loadHubSettings();
    // The route reads the current settings: a save that landed first answers for another relay.
    if (pageBase && pageBaseRelay && endpoint.endsWith(pageBaseRelay)) {
      setHubPageBase(endpoint, pageBase);
    }
  } catch {
    // Not the owner, or offline: the relay URL stays until links next mount.
  } finally {
    pending.delete(endpoint);
  }
}

/** The hub web page for a URL built on the Hub endpoint, for links a person opens. */
export function useHubPageUrl(url: string): string {
  const missing = useMissingHubPageBase();
  useEffect(() => {
    if (missing) void loadSavedPageBase(missing);
  }, [missing]);
  return useHubWebUrl(url);
}
