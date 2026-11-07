// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { agentChat } from "./agent/agent-store";
import { useBrowserStore } from "./store";

export function agentUsingBrowser(): boolean {
  const state = useBrowserStore.getState();
  const agent = agentChat(state.chat);
  return (
    agent.runKey !== null &&
    agent.stoppedRun !== agent.runKey &&
    state.tabs.some((tab) => tab.id === agent.tabId && tab.chat === state.chat)
  );
}

/** Mid-run, closing floats the model's tab instead (a web browser keeps its page loaded); closing that closes. */
export function closeOrFloatBrowser(): void {
  const state = useBrowserStore.getState();
  if (state.floating || !agentUsingBrowser()) {
    state.closePanel();
    return;
  }
  const tabId = agentChat(state.chat).tabId;
  if (tabId && tabId !== state.activeTabId) state.activateTab(tabId);
  state.floatForAgent();
}
