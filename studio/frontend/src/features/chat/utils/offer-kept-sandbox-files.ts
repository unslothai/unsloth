// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { deleteStoredChatThreads } from "./chat-history-storage";

export function offerToDeleteKeptSandboxes(keptThreadIds: string[]): void {
  if (keptThreadIds.length === 0) return;
  toast(
    keptThreadIds.length === 1
      ? "Files from this chat were kept."
      : `Files from ${keptThreadIds.length} chats were kept.`,
    {
      description:
        keptThreadIds.length === 1
          ? "Its sandbox folder is no longer reachable from Unsloth."
          : "Their sandbox folders are no longer reachable from Unsloth.",
      action: {
        label: "Delete files",
        onClick: () => {
          void deleteStoredChatThreads(keptThreadIds, { deleteFiles: true })
            .then((stillKept) => {
              if (stillKept.length > 0) offerToDeleteKeptSandboxes(stillKept);
            })
            .catch(() => {
              toast.error("Could not delete the files.");
            });
        },
      },
    },
  );
}
