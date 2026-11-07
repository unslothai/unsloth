// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ObserverTarget = { id: string; chat: string };
type AgentTarget = { shared?: boolean; tabId: string | null };

export function observedTarget<T extends ObserverTarget>(
  viewerChat: string,
  selection: { chat: string; tabId: string } | null,
  tabs: readonly T[],
  agents: Record<string, AgentTarget>,
): T | undefined {
  if (selection?.chat !== viewerChat) return undefined;
  return tabs.find((tab) => tab.id === selection.tabId && tab.chat !== viewerChat &&
    agents[tab.chat]?.tabId === tab.id);
}
