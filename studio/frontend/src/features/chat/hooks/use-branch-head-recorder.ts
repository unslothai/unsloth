// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAuiState } from "@assistant-ui/react";
import { useEffect } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { writeBranchHead } from "../utils/branch-head";
import { isThreadIncognito } from "../utils/chat-history-storage";

// empty threads are loading; `incognito` changes when a temporary chat gets its persistent id.
export function useBranchHeadRecorder(): void {
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const headId = useAuiState(({ thread }) => thread.messages.at(-1)?.id);
  const incognito = useChatRuntimeStore((s) => s.incognito);

  useEffect(() => {
    if (!remoteId || !headId || isThreadIncognito(remoteId)) return;
    writeBranchHead(remoteId, headId);
  }, [headId, incognito, remoteId]);
}
