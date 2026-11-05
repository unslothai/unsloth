// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { agentChat } from "./agent/agent-store";
import { useBrowserStore } from "./store";

/** True while a chat run is still using this chat's browser: one the user has not stopped, with its tab still open. */
export function agentUsingBrowser(): boolean {
  const state = useBrowserStore.getState();
  const agent = agentChat(state.chat);
  return (
    agent.runKey !== null &&
    agent.stoppedRun !== agent.runKey &&
    state.tabs.some((tab) => tab.id === agent.tabId && tab.chat === state.chat)
  );
}

/** The reader closing the browser beside a chat. While the model is still at work in it, it floats
 *  over the chat instead, showing the model's tab, so the run stays in view (and, in a web browser,
 *  keeps its page loaded). Closing the preview then closes it. */
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
