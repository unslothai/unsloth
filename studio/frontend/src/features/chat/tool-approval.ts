// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useChatRuntimeStore } from "./stores/chat-runtime-store";

export function useToolAwaitingApproval(toolCallId?: string): boolean {
  return useChatRuntimeStore(
    (s) =>
      !!toolCallId &&
      Object.prototype.hasOwnProperty.call(s.toolConfirmations, toolCallId),
  );
}
