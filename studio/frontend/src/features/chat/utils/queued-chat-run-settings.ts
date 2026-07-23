// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";

const QUEUED_SETTING_KEYS = [
  "supportsTools",
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
  "ragEnabled",
  "ragSource",
  "ragMode",
  "ragTopK",
  "ragAutoInject",
  "ragAutoInjectMinScore",
  "ggufContextLength",
  "autoHealToolCalls",
  "nudgeToolCalls",
  "maxToolCallsPerMessage",
  "toolCallTimeout",
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

export function discardQueuedChatRunSettings(id: number): void {
  const index = pendingSettings.findIndex((entry) => entry.id === id);
  if (index >= 0) pendingSettings.splice(index, 1);
}

export function consumeQueuedChatRunSettings(
  threadId?: string | null,
): QueuedChatRunSettings | null {
  let index = threadId
    ? pendingSettings.findIndex((entry) => entry.threadIds.has(threadId))
    : -1;
  // Assistant UI can replace its optimistic local id with a persisted remote
  // id between append() and adapter startup. Global concurrency is one, so a
  // sole pending snapshot is unambiguously the queued run being started.
  if (index < 0 && pendingSettings.length === 1) index = 0;
  if (index < 0) return null;
  return pendingSettings.splice(index, 1)[0].settings;
}
