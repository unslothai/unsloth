// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  type SidebarItem,
  deleteChatItem,
  unarchiveChatItem,
  useChatPreferencesStore,
  useChatRuntimeStore,
  useChatSidebarItems,
} from "@/features/chat";
import { toast } from "@/lib/toast";
import { ArchiveRestoreIcon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useState } from "react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Archived chats shown per page; "Show more" reveals the next page. */
const ARCHIVED_PAGE_SIZE = 20;

function formatCreatedAt(ms: number): string {
  return new Date(ms).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export function ArchivedChatsView() {
  const { archivedItems } = useChatSidebarItems({ requireMessages: false });
  const navigate = useNavigate();
  const closeSettings = useSettingsDialogStore((s) => s.closeDialog);
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  // Open chat id from the route. Compare panes do not write the store, so the
  // pair id only lives in the search params; mirror how the sidebar reads it.
  const openChatId = useRouterState({
    select: (s) => {
      if (!s.location.pathname.startsWith("/chat")) return undefined;
      const search = s.location.search as Record<string, string | undefined>;
      return search.thread ?? search.compare ?? storeThreadId ?? undefined;
    },
  });
  const confirmDeleteChats = useChatPreferencesStore(
    (s) => s.confirmDeleteChats,
  );
  const [confirmingDelete, setConfirmingDelete] = useState<SidebarItem | null>(
    null,
  );
  // Pagination: the view remounts with its settings tab, so plain state
  // restarts from the first page on each visit.
  const [visibleCount, setVisibleCount] = useState(ARCHIVED_PAGE_SIZE);

  // Open an archived chat: leave it archived, just navigate to it.
  function openChat(item: SidebarItem) {
    navigate({
      to: "/chat",
      search:
        item.type === "single" ? { thread: item.id } : { compare: item.id },
    });
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
      // Pass the open chat id (single or compare) so deleting it resets nav.
      await deleteChatItem(item, openChatId, (view) => {
        navigate({
          to: "/chat",
          search: item.projectId
            ? { project: item.projectId }
            : { new: view.newThreadNonce },
        });
      });
      toast.success("Chat deleted");
    } catch (err) {
      toast.error("Failed to delete chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  function requestDelete(item: SidebarItem) {
    if (confirmDeleteChats) setConfirmingDelete(item);
    else void handleDelete(item);
  }

  return (
    <div className="flex flex-col gap-4">
      {archivedItems.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No archived chats.
        </p>
      ) : (
        <div>
          <div className="flex items-center gap-4 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
            <span className="flex-1">Name</span>
            <span className="w-32 shrink-0">Date created</span>
            <span className="w-16 shrink-0" />
          </div>
          {archivedItems.slice(0, visibleCount).map((item) => (
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
                  onClick={() => requestDelete(item)}
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
          {archivedItems.length > visibleCount ? (
            <div className="flex justify-center pt-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  setVisibleCount(visibleCount + ARCHIVED_PAGE_SIZE)
                }
              >
                Show more ({archivedItems.length - visibleCount})
              </Button>
            </div>
          ) : null}
        </div>
      )}

      <AlertDialog
        open={confirmingDelete !== null}
        onOpenChange={(o) => {
          if (!o) setConfirmingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete chat</AlertDialogTitle>
            <AlertDialogDescription>
              Delete{" "}
              <span className="font-medium text-foreground">
                &quot;{confirmingDelete?.title}&quot;
              </span>
              ? This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const item = confirmingDelete;
                setConfirmingDelete(null);
                if (item) void handleDelete(item);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
