// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { translate } from "@/i18n";
import type { ClearStoredChatsResult } from "./chat-history-storage";
import { clearAllChats, countAllChats } from "./clear-all-chats";
import {
  clearAllChatsSucceeded,
  summarizeClearAllChats,
} from "./clear-all-chats-summary";
import { offerToDeleteKeptSandboxes } from "./offer-kept-sandbox-files";

/** `result` is null when the clear threw; `remaining` is null when it could not be counted. */
export async function clearAllChatsWithToast(
  options: { deleteFiles?: boolean },
  knownCount: number | null = null,
): Promise<{ result: ClearStoredChatsResult | null; remaining: number | null }> {
  try {
    const result = await clearAllChats(options);
    // After a clear there is no row left to reach a kept sandbox from.
    offerToDeleteKeptSandboxes(result.sandboxesKept);
    const remaining = clearAllChatsSucceeded(result)
      ? 0
      : await countAllChats().catch(() =>
          result.failedThreadIds.length > 0
            ? result.failedThreadIds.length
            : (knownCount ?? 0),
        );
    const { ok, title, description } = summarizeClearAllChats(result, remaining);
    (ok ? toast.success : toast.warning)(translate(title.key, title.values), {
      description: description
        ? translate(description.key, description.values)
        : undefined,
    });
    return { result, remaining };
  } catch (error) {
    const remaining = await countAllChats().catch(() => knownCount);
    toast.error(translate("settings.chat.failedToClearChats"), {
      description: error instanceof Error ? error.message : undefined,
    });
    return { result: null, remaining };
  }
}
