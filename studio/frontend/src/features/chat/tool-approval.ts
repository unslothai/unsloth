// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useChatRuntimeStore } from "./stores/chat-runtime-store";

/**
 * True while this card's call is parked on the Allow / Deny prompt.
 *
 * The entry is registered when the backend gates the call and cleared once the
 * decision lands, so a card can say it is waiting on a human rather than
 * counting up "Running" against a prompt nobody has answered yet.
 */
export function useToolAwaitingApproval(toolCallId?: string): boolean {
  return useChatRuntimeStore(
    (s) =>
      !!toolCallId &&
      Object.prototype.hasOwnProperty.call(s.toolConfirmations, toolCallId),
  );
}
