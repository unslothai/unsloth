// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { errorMessage } from "@/lib/error-message";
import { toast } from "@/lib/toast";
import { useCharactersStore } from "./store";
import type { ChatView } from "./types";

export function chatViewOf(settings: { view?: ChatView } | undefined): ChatView {
  return settings?.view ?? "chat";
}

/** Switches a chat between the plain chat, stage view and visual novel view; remembered per chat. */
export function setChatView(threadId: string, view: ChatView): void {
  void useCharactersStore
    .getState()
    .patchScene(threadId, { settings: { view } })
    .catch((error) => toast.error(errorMessage(error)));
}
