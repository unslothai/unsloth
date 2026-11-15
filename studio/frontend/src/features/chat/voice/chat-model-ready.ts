// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useStopRunningChatsDialogStore } from "@/features/chat/stores/stop-running-chats-dialog-store";

/** The chat model as voice mode sees it. "switching": being loaded, unloaded or swapped, or a swap is
 *  still asking to stop running chats; a turn sent then would wait on a load that has not started. */
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
