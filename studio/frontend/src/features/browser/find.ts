// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The active tab's page as a find-bar target, matched by the frame script or the native view's own find.

import {
  EMPTY_FIND_RESULT,
  type FindTargetResult,
  notifyFindTargets,
  registerFindTarget,
} from "@/features/find-in-page";
import { hasNativeView, nativeFind, refreshCoveredPage } from "./native-view";
import { sendFrameCommand } from "./page-frame";
import { type BrowserTab, currentEntry, useBrowserStore } from "./store";

export const BROWSER_FIND_TARGET = "browser";

let query = "";
let searchedTabId: string | null = null;
let result: FindTargetResult = EMPTY_FIND_RESULT;

function activeTab(): BrowserTab | undefined {
  const { tabs, activeTabId } = useBrowserStore.getState();
  return tabs.find((tab) => tab.id === activeTabId);
}

/** A web page, not a PDF or other document shown in its place; a loading one is searched once loaded. */
function searchable(tab: BrowserTab | undefined): tab is BrowserTab {
  return Boolean(tab && currentEntry(tab).kind === "web" && !tab.documentType);
}

function setResult(next: FindTargetResult): void {
  if (next.count === result.count && next.active === result.active) return;
  result = next;
  notifyFindTargets();
}

function nativeStep(tabId: string, backwards: boolean): void {
  const asked = query;
  void nativeFind(tabId, asked, backwards).then((found) => {
    // The native view walks matches without counting them: null keeps the walk open.
    if (asked === query && searchedTabId === tabId) setResult({ count: found ? null : 0, active: found ? 0 : -1 });
    // Show the new match on a covered page's snapshot.
    refreshCoveredPage(tabId);
  });
}

function run(): void {
  const tab = activeTab();
  const tabId = searchable(tab) ? tab.id : null;
  if (searchedTabId && searchedTabId !== tabId) sendFrameCommand(searchedTabId, { command: "find", query: "" });
  searchedTabId = query ? tabId : null;
  if (!tabId || !query) {
    if (tabId) sendFrameCommand(tabId, { command: "find", query: "" });
    setResult(EMPTY_FIND_RESULT);
    return;
  }
  if (hasNativeView(tabId)) nativeStep(tabId, false);
  else if (!sendFrameCommand(tabId, { command: "find", query })) setResult(EMPTY_FIND_RESULT);
}

export function receiveFindResult(tabId: string, count: number, active: number): void {
  if (tabId === searchedTabId && query) setResult({ count, active });
}

export function pageLoadedForFind(tabId: string): void {
  if (query && tabId === activeTab()?.id) run();
}

export function registerBrowserFind(contains: (node: Node) => boolean): () => void {
  const unregister = registerFindTarget({
    id: BROWSER_FIND_TARGET,
    available: () => searchable(activeTab()),
    contains,
    search: (next) => {
      query = next;
      run();
    },
    step: (delta) => {
      if (!searchedTabId || !query) return;
      if (hasNativeView(searchedTabId)) nativeStep(searchedTabId, delta < 0);
      else sendFrameCommand(searchedTabId, { command: "findStep", delta });
    },
    result: () => result,
  });
  // Another tab, or a page turning into a document, moves or drops the search.
  let tabId = activeTab()?.id;
  let available = searchable(activeTab());
  const unsubscribe = useBrowserStore.subscribe(() => {
    const tab = activeTab();
    const now = searchable(tab);
    if (tab?.id === tabId && now === available) return;
    tabId = tab?.id;
    available = now;
    notifyFindTargets();
    if (query) run();
  });
  return () => {
    unsubscribe();
    unregister();
  };
}
