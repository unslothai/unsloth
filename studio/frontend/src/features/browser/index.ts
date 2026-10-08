// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { browserPanelAvailable } from "./panel-availability";
import { REACT_PREVIEW_KEY_PREFIX, REACT_PREVIEW_TYPE } from "./react-preview-type";
import { useBrowserPrefsStore } from "./prefs-store";
import { type FileViewMode, type OpenFileInput, useBrowserStore } from "./store";

export { ClearBrowsingDataDialog } from "./clear-data-dialog";
export { BrowserToggleButton } from "./browser-toggle";
export { canAskWhereToSave, saveLinkAs } from "./downloads";
export { DownloadApprovalDialog } from "./download-approval";
export { type DownloadSiteDecision, useDownloadSitesStore } from "./download-sites-store";
export {
  type DownloadFolder,
  nativeDownloadFolder,
  pickNativeDownloadFolder,
  resetNativeDownloadFolder,
} from "./native-downloads";
export { BookmarksFileError, exportBookmarksFile, importBookmarksFile } from "./bookmarks-io";
export { MAX_BOOKMARKS } from "./bookmarks-store";
export { canScreenshot } from "./screenshot-support";
export { browserTabType, fileBarKind, textFileKind } from "./file-kind";
export { SEARCH_ENGINES, type SearchEngineId } from "./address";
export { useBrowserHistoryStore } from "./history-store";
export { useNativeBrowser } from "./native-support";
export { pinBrowserPage } from "./resize-pin";
export {
  type AnnotationScreenshots,
  type BookmarksToolbarMode,
  DEFAULT_ZOOM_STEPS,
  HISTORY_RETENTION_DAYS,
  useBrowserPrefsStore,
} from "./prefs-store";
export { type FileViewMode, type OpenFileInput, useBrowserStore } from "./store";

export { browserPanelAvailable, setBrowserPanelAvailable } from "./panel-availability";
export { PinnedPageRow, usePinnedPages } from "./pinned-page-row";
export type { PinnedPage } from "./pinned-pages-store";

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

const htmlOpenKey = (key: string) => `file:html:${key}`;

export function openHtmlInBrowser({
  key,
  name,
  code,
  view = "preview",
}: {
  key: string;
  name: string;
  code: string;
  view?: FileViewMode;
}): void {
  const store = useBrowserStore.getState();
  store.openFile({ blob: new Blob([code], { type: "text/html" }), name, contentType: "text/html", key: `html:${key}` });
  const tabId = useBrowserStore.getState().activeTabId;
  if (tabId) store.setFileView(tabId, { mode: view });
}

/** Opens a React component from chat as a preview tab; `key` is the artifact's id. */
export function openReactInBrowser({
  key,
  name,
  code,
  view = "preview",
}: {
  key: string;
  name: string;
  code: string;
  view?: FileViewMode;
}): void {
  const store = useBrowserStore.getState();
  store.openFile({
    blob: new Blob([code], { type: REACT_PREVIEW_TYPE }),
    name,
    contentType: REACT_PREVIEW_TYPE,
    key: `${REACT_PREVIEW_KEY_PREFIX}${key}`,
  });
  const tabId = useBrowserStore.getState().activeTabId;
  if (tabId) store.setFileView(tabId, { mode: view });
}

/** The view showing `key`'s HTML, or null if it is not on screen. */
export function useShownHtmlView(key: string): FileViewMode | null {
  return useBrowserStore((state) => {
    if (!state.open) return null;
    const tab = state.tabs.find((candidate) => candidate.id === state.activeTabId);
    if (tab?.openKey !== htmlOpenKey(key)) return null;
    return state.fileViews[tab.id]?.mode ?? "preview";
  });
}
