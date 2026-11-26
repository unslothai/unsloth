// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { loadHubSettings } from "@/features/settings/api/hub-settings";
import { setHubPageBase, useHubWebUrl, useMissingHubPageBase } from "@/lib/hf-endpoint";

// Per endpoint: the store holds one base, so switching back to a mirror reapplies its answer.
const answers = new Map<string, string | null>();

/** `/api/health` hides an endpoint saved in Settings; the owner route names it where this browser
 * can reach it. Others keep the relay. */
async function loadSavedPageBase(endpoint: string): Promise<void> {
  if (answers.has(endpoint)) {
    const known = answers.get(endpoint);
    if (known) setHubPageBase(endpoint, known);
    return;
  }
  answers.set(endpoint, null);
  try {
    const { pageBase, pageBaseRelay } = await loadHubSettings();
    // A save that landed first answers for the new endpoint; leave this one to ask again.
    if (pageBase && !(pageBaseRelay && endpoint.endsWith(pageBaseRelay))) {
      answers.delete(endpoint);
      return;
    }
    answers.set(endpoint, pageBase);
    if (pageBase) setHubPageBase(endpoint, pageBase);
  } catch {
    // Not the owner, or offline: keep the relay URL now, and ask again when next needed.
    answers.delete(endpoint);
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
