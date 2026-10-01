// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBrowserPrefsStore } from "./prefs-store";
import { type OpenFileInput, useBrowserStore } from "./store";

export { BrowserPanel, ClearBrowsingDataDialog } from "./browser-panel";
export { BrowserToggleButton } from "./browser-toggle";
export { saveLinkAs } from "./downloads";
export { browserTabType, textFileKind } from "./file-view";
export { FullViewChatBar, FullViewChatButton } from "./full-view-chat";
export { SEARCH_ENGINES, type SearchEngineId } from "./address";
export { useBrowserHistoryStore } from "./history-store";
export { nativeBrowser } from "./native-view";
export { useBrowserPrefsStore } from "./prefs-store";
export { type OpenFileInput, useBrowserStore } from "./store";

// Set by the chat page while the panel can be shown (not on mobile).
let panelAvailable = false;

export function setBrowserPanelAvailable(available: boolean): void {
  panelAvailable = available;
}

export function browserPanelAvailable(): boolean {
  return panelAvailable;
}

/** Open a link in the browser panel; false if unavailable or links go to the system browser. */
export function openUrlInBrowser(url: string): boolean {
  if (!panelAvailable || !useBrowserPrefsStore.getState().openLinksInBrowser) return false;
  useBrowserStore.getState().openUrl(url);
  return true;
}

/** Open a link in the browser panel whatever the preference; false while the panel can't show. */
export function openUrlInBrowserPanel(url: string): boolean {
  if (!panelAvailable || !/^https?:\/\//i.test(url)) return false;
  useBrowserStore.getState().openUrl(url);
  return true;
}

/** Whether opened files go to the browser panel instead of the preview dialog. */
export function filesOpenInBrowser(): boolean {
  return panelAvailable && useBrowserPrefsStore.getState().openFilesInBrowser;
}

export function openFileInBrowser(input: OpenFileInput): void {
  useBrowserStore.getState().openFile(input);
}
