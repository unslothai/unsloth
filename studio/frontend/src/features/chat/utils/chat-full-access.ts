// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ChatFullAccessGrant {
  runId: string;
  threadId: string;
  sessionId: string;
  /** chatPermissionChoiceVersion when granted: a later explicit choice voids an unadopted grant. */
  choiceVersion?: number;
}

export type ChatFullAccessGrants = Readonly<Record<string, ChatFullAccessGrant>>;

/** Workspace sessions can be shared by multiple chats; they never identify a grant by themselves. */
export function hasChatFullAccess(
  grants: ChatFullAccessGrants,
  threadId: string | null | undefined,
  runId?: string,
): boolean {
  const ownRun = runId && Object.hasOwn(grants, runId) ? grants[runId] : undefined;
  return !!(
    (ownRun && (!ownRun.threadId || ownRun.threadId === threadId)) ||
    (threadId && Object.values(grants).some((grant) => grant.threadId === threadId))
  );
}

type BasePermissionMode = "auto" | "ask" | "off";
interface PermissionState {
  permissionMode: BasePermissionMode | "full";
  bypassPermissions: boolean;
  confirmToolCalls: boolean;
  deepResearchEnabled: boolean;
  /** Base mode displaced by this chat shortcut; null means no active chat overlay. */
  chatFullAccessPreviousMode: BasePermissionMode | null;
}

/** Overlay a live chat grant on queued or current settings without inheriting another chat's bypass. */
export function withChatFullAccess<T extends PermissionState>(
  state: T,
  grants: ChatFullAccessGrants,
  threadId: string | null | undefined,
): T {
  if (state.permissionMode === "full" && state.chatFullAccessPreviousMode === null) return state;
  if (hasChatFullAccess(grants, threadId)) {
    return {
      ...state,
      chatFullAccessPreviousMode: state.permissionMode === "full"
        ? state.chatFullAccessPreviousMode
        : state.permissionMode,
      permissionMode: "full",
      bypassPermissions: true,
      confirmToolCalls: false,
      deepResearchEnabled: false,
    };
  }
  if (state.chatFullAccessPreviousMode !== null) {
    const mode = state.permissionMode === "full" ? state.chatFullAccessPreviousMode : state.permissionMode;
    return {
      ...state,
      chatFullAccessPreviousMode: null,
      permissionMode: mode,
      bypassPermissions: false,
      confirmToolCalls: mode === "ask" || mode === "auto",
    };
  }
  return state;
}

export function revokeChatFullAccess(
  grants: ChatFullAccessGrants,
  threadId: string | null,
  runId?: string | null,
): ChatFullAccessGrants {
  return Object.fromEntries(Object.entries(grants).filter(([key, grant]) =>
    threadId ? grant.threadId !== threadId : !runId || key !== runId,
  ));
}
