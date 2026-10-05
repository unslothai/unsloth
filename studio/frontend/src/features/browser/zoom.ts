// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type BrowserTab, currentEntry, useBrowserStore } from "./store";

export function canZoom(tab: BrowserTab | undefined): tab is BrowserTab {
  const kind = tab ? currentEntry(tab).kind : null;
  return kind === "web" || kind === "file";
}

export const ZOOM_STEPS = [0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5];

export function stepZoom(zoom: number, direction: 1 | -1): number {
  if (direction > 0) return ZOOM_STEPS.find((step) => step > zoom + 0.001) ?? zoom;
  return [...ZOOM_STEPS].reverse().find((step) => step < zoom - 0.001) ?? zoom;
}

/** Zooms a tab's page a step in (1), out (-1), or to 100% (0), apart from the interface's zoom. */
export function zoomTab(tabId: string, direction: 1 | -1 | 0): void {
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === tabId);
  if (!canZoom(tab)) return;
  store.setZoom(tabId, direction === 0 ? 1 : stepZoom(tab.zoom, direction));
}
