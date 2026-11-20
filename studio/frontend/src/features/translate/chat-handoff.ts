// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatArtifactsStore } from "@/features/chat/artifacts/store";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  type TranslateHistoryEntry,
  buildChatHandoff,
} from "./translate-history";

export const CHAT_HANDOFF_ENTRIES = 5;

type HandoffEntry = Pick<
  TranslateHistoryEntry,
  "at" | "source" | "output" | "target" | "from"
>;

export function chatAboutTranslations(
  entries: readonly HandoffEntry[],
  navigate: (options: {
    to: "/chat";
    search: { new: string };
  }) => Promise<void>,
): boolean {
  const handoff = buildChatHandoff(entries);
  if (!handoff) return false;
  const runtime = useChatRuntimeStore.getState();
  runtime.setActiveThreadId(null);
  runtime.setActiveProjectId(null);
  runtime.setIncognito(false);
  // Staged once the new chat is up: opening one empties the message box, staged text included.
  void navigate({ to: "/chat", search: { new: crypto.randomUUID() } }).then(
    () => useChatArtifactsStore.getState().stageFixPrompt(handoff),
  );
  return true;
}
