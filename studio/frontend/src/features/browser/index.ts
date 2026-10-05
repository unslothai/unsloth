// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { browserPanelAvailable } from "./panel-availability";
import { useBrowserPrefsStore } from "./prefs-store";
import { type OpenFileInput, useBrowserStore } from "./store";

export { ClearBrowsingDataDialog } from "./clear-data-dialog";
export { BrowserToggleButton } from "./browser-toggle";
export { saveLinkAs } from "./downloads";
export { browserTabType, textFileKind } from "./file-kind";
export { SEARCH_ENGINES, type SearchEngineId } from "./address";
export { useBrowserHistoryStore } from "./history-store";
export { useNativeBrowser } from "./native-support";
export { pinBrowserPage } from "./resize-pin";
export { type BookmarksToolbarMode, useBrowserPrefsStore } from "./prefs-store";
export { type OpenFileInput, useBrowserStore } from "./store";

export { browserPanelAvailable, setBrowserPanelAvailable } from "./panel-availability";
export { PinnedPageRows, usePinnedPageCount } from "./pinned-page-row";

/** Open a link in the browser panel; false if unavailable or links go to the system browser. */
export function openUrlInBrowser(url: string): boolean {
  if (!browserPanelAvailable() || !useBrowserPrefsStore.getState().openLinksInBrowser) return false;
  useBrowserStore.getState().openUrl(url);
  return true;
}

/** Open a link in the browser panel whatever the preference; false while the panel can't show. */
export function openUrlInBrowserPanel(url: string): boolean {
  if (!browserPanelAvailable() || !/^https?:\/\//i.test(url)) return false;
  useBrowserStore.getState().openUrl(url);
  return true;
}

export function filesOpenInBrowser(): boolean {
  return browserPanelAvailable() && useBrowserPrefsStore.getState().openFilesInBrowser;
}

export function openFileInBrowser(input: OpenFileInput): void {
  useBrowserStore.getState().openFile(input);
}
