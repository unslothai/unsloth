// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { currentEntry, onChatAdopted, useBrowserStore } from "../store";
import { adoptAgentChat, agentChat, liveChat, notifyAgentTargetChange, patchAgentChat, stopBrowserAgent, useBrowserAgentStore } from "./agent-store";

onChatAdopted(adoptAgentChat);

export function shareAgentTab(chat: string, tabId: string | null): boolean {
  const key = liveChat(chat);
  const browser = useBrowserStore.getState();
  if (key !== browser.chat) return false;
  if (tabId) {
    if (browser.activeTabId !== tabId) return false;
    const tab = browser.tabs.find((item) => item.id === tabId && item.chat === key);
    if (!tab || currentEntry(tab).kind !== "web") return false;
    useBrowserAgentStore.getState().setEnabled(true);
  }
  const current = agentChat(key);
  if (current.tabId === tabId && current.shared === Boolean(tabId)) return true;
  stopBrowserAgent(key);
  patchAgentChat(key, { tabId, shared: Boolean(tabId), targetVersion: current.targetVersion + 1 });
  notifyAgentTargetChange(key, current.tabId);
  return true;
}
