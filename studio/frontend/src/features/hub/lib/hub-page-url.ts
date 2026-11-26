// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { loadHubSettings } from "@/features/settings/api/hub-settings";
import { setHubPageBase, useHubWebUrl, useMissingHubPageBase } from "@/lib/hf-endpoint";

const asked = new Set<string>();

/** `/api/health` hides an endpoint saved in Settings; the owner route names it where this browser
 * can reach it. Others keep the relay. */
async function loadSavedPageBase(endpoint: string): Promise<void> {
  if (asked.has(endpoint)) return;
  asked.add(endpoint);
  try {
    const settings = await loadHubSettings();
    if (settings.pageBase) setHubPageBase(endpoint, settings.pageBase);
  } catch {
    // Not the owner, or offline: the relay URL still redirects for a local browser.
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
