// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useStopRunningChatsDialogStore } from "@/features/chat/stores/stop-running-chats-dialog-store";

export type ChatModelStatus = "ready" | "switching" | "none";

export function useChatModelStatus(): ChatModelStatus {
  const status = useChatRuntimeStore((s): ChatModelStatus =>
    s.modelLoading || s.loadingModelPick !== null
      ? "switching"
      : s.params.checkpoint
        ? "ready"
        : "none",
  );
  const asking = useStopRunningChatsDialogStore((s) => s.open);
  return asking && status === "ready" ? "switching" : status;
}
