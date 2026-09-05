// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useChatRuntimeStore } from "../stores/chat-runtime-store";

const QUEUED_SETTING_KEYS = [
  "activeGgufVariant",
  "supportsTools",
  "supportsReasoning",
  "reasoningAlwaysOn",
  "reasoningStyle",
  "supportsReasoningOff",
  "reasoningEffortLevels",
  "supportsPreserveThinking",
  "reasoningEnabled",
  "reasoningEffort",
  "preserveThinking",
  "toolsEnabled",
  "codeToolsEnabled",
  "imageToolsEnabled",
  "artifactsEnabled",
  "mcpEnabledForChat",
  "confirmToolCalls",
  "bypassPermissions",
  "permissionMode",
  "webFetchToolsEnabled",
  "deepResearchEnabled",
  "researchWebsitePolicy",
  "researchModelTimeoutSeconds",
  "ragEnabled",
  "ragSource",
  "ragMode",
  "ragTopK",
  "ragAutoInject",
  "ragAutoInjectMinScore",
  "loadedContextLength",
  "autoHealToolCalls",
  "nudgeToolCalls",
  "maxToolCallsPerMessage",
  "toolCallTimeout",
  "autoCompactEnabled",
  "contextPolicy",
  "compactionHeadroomRatio",
] as const;

type ChatRuntimeState = ReturnType<typeof useChatRuntimeStore.getState>;

export type QueuedChatRunSettings = Pick<
  ChatRuntimeState,
  (typeof QUEUED_SETTING_KEYS)[number]
> & {
  params: ChatRuntimeState["params"];
};

type PendingSettings = {
  id: number;
  threadIds: Set<string>;
  settings: QueuedChatRunSettings;
};

let nextPendingSettingsId = 1;
const pendingSettings: PendingSettings[] = [];

export function snapshotQueuedChatRunSettings(
  state: ChatRuntimeState,
): QueuedChatRunSettings {
  const snapshot = {
    params: { ...state.params },
  } as QueuedChatRunSettings;
  for (const key of QUEUED_SETTING_KEYS) {
    Object.assign(snapshot, { [key]: state[key] });
  }
  return snapshot;
}

/** A queued send may only fill in the model of a row that was written without one. */
export function shouldPersistResolvedQueuedModel(
  storedThread: { modelId?: string | null } | null | undefined,
): boolean {
  return Boolean(storedThread && !storedThread.modelId);
}

export function registerQueuedChatRunSettings(
  threadIds: string[],
  settings: QueuedChatRunSettings,
): number {
  const id = nextPendingSettingsId++;
  pendingSettings.push({
    id,
    threadIds: new Set(threadIds),
    settings,
  });
  return id;
}

export function addQueuedChatRunSettingsThreadIds(
  id: number,
  threadIds: string[],
): void {
  const entry = pendingSettings.find((candidate) => candidate.id === id);
  if (!entry) {
    return;
  }
  for (const threadId of threadIds) {
    if (threadId) {
      entry.threadIds.add(threadId);
    }
  }
}

export function discardQueuedChatRunSettings(id: number): void {
  const index = pendingSettings.findIndex((entry) => entry.id === id);
  if (index >= 0) {
    pendingSettings.splice(index, 1);
  }
}

export function discardQueuedChatRunSettingsForThread(
  threadId?: string | null,
): void {
  if (!threadId) {
    return;
  }
  for (let index = pendingSettings.length - 1; index >= 0; index -= 1) {
    if (pendingSettings[index].threadIds.has(threadId)) {
      pendingSettings.splice(index, 1);
    }
  }
}

export function hasQueuedChatRunSettings(threadId?: string | null): boolean {
  return Boolean(
    threadId && pendingSettings.some((entry) => entry.threadIds.has(threadId)),
  );
}

export function consumeQueuedChatRunSettings(
  threadId?: string | null,
): QueuedChatRunSettings | null {
  const index = threadId
    ? pendingSettings.findIndex((entry) => entry.threadIds.has(threadId))
    : -1;
  // Never consume another chat's snapshot as a fallback. Multiple queued chats can start
  // concurrently, so a "sole pending entry" is not proof that it belongs to this adapter run.
  if (index < 0) {
    return null;
  }
  // Tool calls can invoke the adapter multiple times for one assistant run. Keep the snapshot
  // available until the owning prompt queue observes that the whole run is idle, then discard it
  // through its registration id.
  return pendingSettings[index].settings;
}
