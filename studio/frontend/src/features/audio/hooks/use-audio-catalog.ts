// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useSyncExternalStore } from "react";
import { fetchAudioCatalog } from "../api";
import {
  type AudioCatalogSnapshot,
  createAudioCatalogStore,
} from "../audio-catalog-rows";

const store = createAudioCatalogStore(fetchAudioCatalog);
const CATALOG_POLL_MS = 30_000;

/** The backend's audio.cpp catalog, shared by every Audio page; the static seed until it lands. */
export function useAudioCatalog(enabled: boolean): AudioCatalogSnapshot {
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot);
  useEffect(() => {
    if (!enabled) return;
    void store.refresh();
    // The page stays mounted, so retries and TTL refreshes need a timer; refresh is a no-op until due.
    const timer = setInterval(() => void store.refresh(), CATALOG_POLL_MS);
    return () => clearInterval(timer);
  }, [enabled]);
  return snapshot;
}
