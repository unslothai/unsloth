// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { deleteStoredChatThreads } from "./chat-history-storage";

/**
 * Offer to delete sandbox files a delete kept.
 *
 * Once the chats are gone there is no card left to reach those folders from,
 * so every delete surface that cannot ask up front makes the offer here.
 */
export function offerToDeleteKeptSandboxes(keptThreadIds: string[]): void {
  if (keptThreadIds.length === 0) return;
  toast(
    keptThreadIds.length === 1
      ? "Files from this chat were kept."
      : `Files from ${keptThreadIds.length} chats were kept.`,
    {
      description:
        keptThreadIds.length === 1
          ? "Its sandbox folder is no longer reachable from Studio."
          : "Their sandbox folders are no longer reachable from Studio.",
      action: {
        label: "Delete files",
        onClick: () => {
          void deleteStoredChatThreads(keptThreadIds, {
            deleteFiles: true,
          }).catch(() => {
            toast.error("Could not delete the files.");
          });
        },
      },
    },
  );
}
