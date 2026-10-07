// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { defaultZoom } from "./prefs-store";
import { type BrowserTab, currentEntry, pageDownload, useBrowserStore } from "./store";

export function canZoom(tab: BrowserTab | undefined): tab is BrowserTab {
  const kind = tab ? currentEntry(tab).kind : null;
  return kind === "web" || kind === "file";
}

export const ZOOM_STEPS = [0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5];

export function stepZoom(zoom: number, direction: 1 | -1): number {
  if (direction > 0) return ZOOM_STEPS.find((step) => step > zoom + 0.001) ?? zoom;
  return [...ZOOM_STEPS].reverse().find((step) => step < zoom - 0.001) ?? zoom;
}

/** Reset target: 100% for files (attached or fetched), the default zoom for web pages. */
export function homeZoom(tab: BrowserTab, preferred = defaultZoom()): number {
  return currentEntry(tab).kind === "file" || pageDownload(tab.id) ? 1 : preferred;
}

// Tabs moved to 100% for a fetched file, so the next web page goes back to the default.
const fittedForFile = new Set<string>();

/** Show a fetched file at 100% and restore the default zoom after it, unless the reader zoomed. */
export function fitZoomToPage(tabId: string, isFile: boolean): void {
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === tabId);
  if (!tab) return;
  const preferred = defaultZoom();
  const at = (zoom: number) => Math.abs(tab.zoom - zoom) < 0.001;
  if (isFile) {
    // Marked even when the default is 100%, so a default changed while it shows applies after.
    if (!at(preferred)) return;
    fittedForFile.add(tabId);
    if (!at(1)) store.setZoom(tabId, 1);
  } else if (fittedForFile.delete(tabId) && at(1)) {
    store.setZoom(tabId, preferred);
  }
}

/** Zoom a tab's page in (1), out (-1), or back to its default (0), apart from the interface's zoom. */
export function zoomTab(tabId: string, direction: 1 | -1 | 0): void {
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === tabId);
  if (!canZoom(tab)) return;
  store.setZoom(tabId, direction === 0 ? homeZoom(tab) : stepZoom(tab.zoom, direction));
}
