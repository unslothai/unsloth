// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { currentEntry, onChatAdopted, useBrowserStore } from "../store";
import { useNativeBrowser } from "../native-support";
import { keepInBackground, releaseBackground } from "../native-view";
import { adoptAgentChat, agentChat, liveChat, notifyAgentTargetChange, patchAgentChat, setBrowserAgentEnabled, stopBrowserAgent, useBrowserAgentStore } from "./agent-store";

onChatAdopted(adoptAgentChat);

export function shareAgentTab(chat: string, tabId: string | null): boolean {
  const key = liveChat(chat);
  const browser = useBrowserStore.getState();
  if (key !== browser.chat) return false;
  if (tabId) {
    if (browser.activeTabId !== tabId) return false;
    const tab = browser.tabs.find((item) => item.id === tabId && item.chat === key);
    if (!tab || currentEntry(tab).kind !== "web") return false;
    if (Object.entries(useBrowserAgentStore.getState().chats).some(([owner, agent]) => liveChat(owner) !== key && agent.tabId === tabId)) return false;
    setBrowserAgentEnabled(key, true);
  }
  const current = agentChat(key);
  if (current.tabId === tabId && current.shared === Boolean(tabId)) return true;
  stopBrowserAgent(key);
  patchAgentChat(key, { tabId, shared: Boolean(tabId), targetVersion: current.targetVersion + 1 });
  if (useNativeBrowser.getState().enabled) {
    const previous = browser.tabs.find((item) => item.id === current.tabId && liveChat(item.chat) === key);
    if (previous && previous.id !== tabId) releaseBackground(previous.id);
    if (tabId) keepInBackground(tabId);
  }
  notifyAgentTargetChange(key, current.tabId);
  return true;
}
