


"use client";

import { useChatRuntimeStore } from "./stores/chat-runtime-store";

/**
 * True while this card's call is parked on the Allow / Deny prompt, so it can say it is
 * waiting rather than counting up "Running". Set when the backend gates the call.
 */
export function useToolAwaitingApproval(toolCallId?: string): boolean {
  return useChatRuntimeStore(
    (s) =>
      !!toolCallId &&
      Object.prototype.hasOwnProperty.call(s.toolConfirmations, toolCallId),
  );
}
