// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type { ClearStoredChatsResult } from "./chat-history-storage";

type Message = { key: TranslationKey; values?: Record<string, number> };

export interface ClearAllChatsSummary {
  ok: boolean;
  title: Message;
  description?: Message;
}

// A failed store reports no failed ids, so an empty failure list alone is not success.
export function clearAllChatsSucceeded(result: ClearStoredChatsResult): boolean {
  return (
    result.backend !== "failed" &&
    result.legacy !== "failed" &&
    result.failedThreadIds.length === 0
  );
}

/** `remaining` is read only when a store failed without naming the chats it kept. */
export function summarizeClearAllChats(
  result: ClearStoredChatsResult,
  remaining: number,
): ClearAllChatsSummary {
  const cleared = result.deletedThreadIds.length;
  if (clearAllChatsSucceeded(result)) {
    return {
      ok: true,
      title:
        cleared === 0
          ? { key: "settings.chat.clearedAllChats" }
          : cleared === 1
            ? { key: "settings.chat.clearedOneChat" }
            : { key: "settings.chat.clearedChatCount", values: { count: cleared } },
    };
  }
  const failed = result.failedThreadIds.length;
  let description: Message;
  if (failed === 0) {
    description =
      remaining === 1
        ? { key: "settings.chat.storageClearFailedOne" }
        : { key: "settings.chat.storageClearFailed", values: { count: remaining } };
  } else if (cleared === 1) {
    description =
      failed === 1
        ? { key: "settings.chat.oneChatClearedRemainOne" }
        : { key: "settings.chat.oneChatClearedRemain", values: { remainingCount: failed } };
  } else {
    description =
      failed === 1
        ? { key: "settings.chat.chatsClearedRemainOne", values: { clearedCount: cleared } }
        : {
            key: "settings.chat.chatsClearedRemain",
            values: { clearedCount: cleared, remainingCount: failed },
          };
  }
  return {
    ok: false,
    title: { key: "settings.chat.someChatsCouldNotBeCleared" },
    description,
  };
}
