// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AGENT_DEFAULT_ROWS,
  isAgentWindowSize,
  type AgentWindowSize,
} from "../../../components/assistant-ui/agent-history-config.ts";

import { agentBufferRowLimit, isAgentBufferMode, type AgentDomLevel } from "../../../components/assistant-ui/agent-history-config.ts";

export type AgentHistoryPreferenceSnapshot = {
  globalDomLevel?: AgentDomLevel;
  domLevelByThreadId?: Record<string, AgentDomLevel>;
  globalEnabled: boolean;
  globalWindowRows: AgentWindowSize;
  enabledThreads: Record<string, true>;
  disabledThreads: Record<string, true>;
  windowRowsByThreadId: Record<string, AgentWindowSize>;
};

/** One canonical effective-mode decision, shared by renderer and UI helpers. */
export function agentHistoryEnabledFor(
  state: AgentHistoryPreferenceSnapshot,
  threadId: string | null | undefined,
): boolean {
  return agentHistoryLevelFor(state, threadId) !== "off";
}

export function agentHistoryWindowFor(
  state: AgentHistoryPreferenceSnapshot,
  threadId: string | null | undefined,
): AgentWindowSize {
  const active = agentHistoryLevelFor(state, threadId);
  if (isAgentBufferMode(active)) return agentBufferRowLimit(active);
  if (active === "low") return 5;
  if (active === "high") return 32;
  const specific = threadId ? state.windowRowsByThreadId[threadId] : undefined;
  if (isAgentWindowSize(specific)) return specific;
  return isAgentWindowSize(state.globalWindowRows)
    ? state.globalWindowRows
    : AGENT_DEFAULT_ROWS;
}

/** Explicit per-chat settings win. Legacy v1 preferences remain readable. */
export function agentHistoryLevelFor(
  state: AgentHistoryPreferenceSnapshot,
  threadId: string | null | undefined,
): AgentDomLevel | "legacy" {
  // There is no chat to virtualize before a thread ID is assigned.
  if (!threadId) return "off";
  const specific = state.domLevelByThreadId?.[threadId];
  if (specific === "off" || specific === "low" || specific === "high" ||
      isAgentBufferMode(specific)) return specific;
  // A pre-existing explicit per-chat opt-out/in is a local exception too.
  // Do not overwrite it when adopting the new global policy.
  if (threadId && state.disabledThreads[threadId] === true) return "off";
  if (threadId && state.enabledThreads[threadId] === true) {
    return (isAgentBufferMode(state.globalDomLevel) ||
      state.globalDomLevel === "low" || state.globalDomLevel === "high")
      ? state.globalDomLevel : "legacy";
  }
  if (state.globalDomLevel === "off") return "off";
  if (state.globalDomLevel === "low" || state.globalDomLevel === "high") {
    return state.globalDomLevel; // v1 dynamic setting was implicitly enabled
  }
  if (isAgentBufferMode(state.globalDomLevel)) {
    return state.globalEnabled ? state.globalDomLevel : "off";
  }
  // The legacy enabled/disabled maps are consulted only after explicit
  // per-thread and global modes have been resolved. Do not call the public
  // enabled selector here: it delegates to this canonical mode decision.
  return threadId && state.disabledThreads[threadId] !== true &&
    (state.globalEnabled || state.enabledThreads[threadId] === true)
    ? "legacy" : "off";
}


/** Public UI has three choices; old High/Low values stay readable on disk. */
export type AgentHistoryUiMode = "off" | "dynamic" | "fixed";
export function agentHistoryUiModeFor(level: AgentDomLevel | "legacy"): AgentHistoryUiMode {
  if (level === "off") return "off";
  if (level === "low" || level === "high" ||
      level === "dynamic" || level === "dynamic-low" || level === "dynamic-high") return "dynamic";
  return "fixed";
}
export function agentHistoryStorageModeForUi(mode: AgentHistoryUiMode): AgentDomLevel {
  if (mode === "dynamic") return "dynamic";
  if (mode === "fixed") return "fixed-low";
  return "off";
}


/** Normalize historical UI variants without changing persisted message content. */
export function normalizeAgentHistoryDomMode(level: unknown): AgentDomLevel | undefined {
  if (level === "off") return "off";
  if (level === "low" || level === "high" || level === "dynamic" ||
      level === "dynamic-low" || level === "dynamic-high") return "dynamic";
  if (level === "fixed-low" || level === "fixed-high") return "fixed-low";
  return undefined;
}
