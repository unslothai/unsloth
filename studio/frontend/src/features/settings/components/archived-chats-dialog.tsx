// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  deleteChatItem,
  unarchiveChatItem,
  useChatSidebarItems,
  type SidebarItem,
} from "@/features/chat";
import { toast } from "@/lib/toast";
import { ArchiveRestoreIcon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

function formatCreatedAt(ms: number): string {
  return new Date(ms).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export function ArchivedChatsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { archivedItems } = useChatSidebarItems({ requireMessages: false });
  const navigate = useNavigate();
  const closeSettings = useSettingsDialogStore((s) => s.closeDialog);

  // Open an archived chat: leave it archived, just navigate to it.
  function openChat(item: SidebarItem) {
    navigate({
      to: "/chat",
      search:
        item.type === "single" ? { thread: item.id } : { compare: item.id },
    });
    onOpenChange(false);
    closeSettings();
  }

  async function handleUnarchive(item: SidebarItem) {
    try {
      await unarchiveChatItem(item);
      toast.success("Chat unarchived");
    } catch (err) {
      toast.error("Failed to unarchive chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function handleDelete(item: SidebarItem) {
    try {
      await deleteChatItem(item, undefined, () => {});
      toast.success("Chat deleted");
    } catch (err) {
      toast.error("Failed to delete chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Archived chats</DialogTitle>
        </DialogHeader>

        {archivedItems.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No archived chats.
          </p>
        ) : (
          <div className="max-h-[60vh] overflow-y-auto">
            <div className="flex items-center gap-4 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
              <span className="flex-1">Name</span>
              <span className="w-32 shrink-0">Date created</span>
              <span className="w-16 shrink-0" />
            </div>
            {archivedItems.map((item) => (
              <div
                key={item.id}
                className="group flex items-center gap-4 border-b border-border/40 px-1 py-2.5 text-sm last:border-0"
              >
                <button
                  type="button"
                  onClick={() => openChat(item)}
                  className="min-w-0 flex-1 truncate text-left text-primary hover:underline"
                  title={item.title}
                >
                  {item.title}
                </button>
                <span className="w-32 shrink-0 text-muted-foreground tabular-nums">
                  {formatCreatedAt(item.createdAt)}
                </span>
                <span className="flex w-16 shrink-0 items-center justify-end gap-1">
                  <button
                    type="button"
                    onClick={() => void handleUnarchive(item)}
                    aria-label="Unarchive chat"
                    title="Unarchive"
                    className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={ArchiveRestoreIcon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleDelete(item)}
                    aria-label="Delete chat"
                    title="Delete"
                    className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                  >
                    <HugeiconsIcon
                      icon={Delete02Icon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                  </button>
                </span>
              </div>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
