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
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  COMBINED_EXPORT_FORMATS_LIST,
  type ConvExportFormat,
  EXPORT_FORMATS_LIST,
  type SidebarItem,
  archiveChatItems,
  deleteChatItems,
  exportBulkConversationsMerged,
  exportBulkConversationsSeparate,
  moveChatItemToProject,
  rangeBetween,
  useChatPreferencesStore,
  useChatProjects,
  useChatRuntimeStore,
  useChatSidebarItems,
  usePinnedChatsStore,
} from "@/features/chat";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import {
  Archive02Icon,
  Delete02Icon,
  Download01Icon,
  Folder01Icon,
  PinIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

const MANAGE_PAGE_SIZE = 20;

function formatCreatedAt(ms: number): string {
  return new Date(ms).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function chatCount(n: number): string {
  return n === 1 ? "1 chat" : `${n} chats`;
}

export function ManageChatsView() {
  const { items } = useChatSidebarItems({ requireMessages: false });
  const { projects } = useChatProjects();
  const navigate = useNavigate();
  const closeSettings = useSettingsDialogStore((s) => s.closeDialog);
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  // Open chat id from the route: compare panes only live in the search params.
  const openChatId = useRouterState({
    select: (s) => {
      if (!s.location.pathname.startsWith("/chat")) return undefined;
      const search = s.location.search as Record<string, string | undefined>;
      return search.thread ?? search.compare ?? storeThreadId ?? undefined;
    },
  });
  const pinnedIds = usePinnedChatsStore((s) => s.pinnedIds);
  const setPinned = usePinnedChatsStore((s) => s.setPinned);
  const alwaysDeleteChatFiles = useChatPreferencesStore(
    (s) => s.alwaysDeleteChatFiles,
  );

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [visibleCount, setVisibleCount] = useState(MANAGE_PAGE_SIZE);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [busy, setBusy] = useState(false);
  const lastToggledId = useRef<string | null>(null);

  const visible = items.slice(0, visibleCount);
  const selectedItems = items.filter((item) => selectedIds.has(item.id));
  const selectedCount = selectedItems.length;
  const allVisibleSelected =
    visible.length > 0 && visible.every((item) => selectedIds.has(item.id));
  const allSelectedPinned =
    selectedCount > 0 &&
    selectedItems.every((item) => pinnedIds.includes(item.id));
  const projectNames = new Map(projects.map((p) => [p.id, p.name]));

  function toggleRow(index: number, shiftKey: boolean) {
    const rowId = visible[index].id;
    const target = !selectedIds.has(rowId);
    const anchorId = lastToggledId.current;
    // Anchor on the chat id, not the row index: the list re-sorts by updatedAt,
    // so an index anchor would range over rows the user never picked.
    const ids =
      shiftKey && anchorId !== null
        ? rangeBetween(
            visible.map((item) => item.id),
            anchorId,
            rowId,
          )
        : [rowId];
    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (const id of ids) {
        if (target) next.add(id);
        else next.delete(id);
      }
      return next;
    });
    lastToggledId.current = rowId;
  }

  function toggleAllVisible() {
    setSelectedIds(
      allVisibleSelected ? new Set() : new Set(visible.map((item) => item.id)),
    );
    lastToggledId.current = null;
  }

  function openChat(item: SidebarItem) {
    // Carry the row's project, as the sidebar does: without it ChatPage briefly
    // runs the chat under the project it was already on.
    const project = item.projectId ? { project: item.projectId } : {};
    navigate({
      to: "/chat",
      search:
        item.type === "single"
          ? { thread: item.id, ...project }
          : { compare: item.id, ...project },
    });
    closeSettings();
  }

  function resetView(view: { newThreadNonce: string }) {
    navigate({ to: "/chat", search: { new: view.newThreadNonce } });
  }

  async function run(
    action: () => Promise<void>,
    success: string,
    failure: string,
  ) {
    setBusy(true);
    try {
      await action();
      setSelectedIds(new Set());
      toast.success(success);
    } catch (err) {
      toast.error(failure, {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setBusy(false);
    }
  }

  const handleArchive = () =>
    run(
      () => archiveChatItems(selectedItems, openChatId, resetView),
      `Archived ${chatCount(selectedCount)}`,
      "Failed to archive chats",
    );

  const handleDelete = () =>
    run(
      () =>
        // Honour the same preference as the sidebar delete, or the selected
        // chats leave orphan sandbox folders behind.
        deleteChatItems(selectedItems, openChatId, resetView, {
          deleteFiles: alwaysDeleteChatFiles,
        }),
      `Deleted ${chatCount(selectedCount)}`,
      "Failed to delete chats",
    );

  const handleMove = (projectId: string | null) =>
    run(
      async () => {
        await Promise.all(
          selectedItems.map((item) => moveChatItemToProject(item, projectId)),
        );
      },
      projectId
        ? `Moved ${chatCount(selectedCount)} to ${projectNames.get(projectId) ?? "project"}`
        : `Moved ${chatCount(selectedCount)} to Recents`,
      "Failed to move chats",
    );

  function handleTogglePin() {
    const ids = selectedItems.map((item) => item.id);
    setPinned(ids, !allSelectedPinned);
    toast.success(
      `${allSelectedPinned ? "Unpinned" : "Pinned"} ${chatCount(ids.length)}`,
    );
    setSelectedIds(new Set());
  }

  async function handleExport(format: ConvExportFormat, merged: boolean) {
    const threadIds = selectedItems.flatMap(
      (item) => item.threadIds ?? [item.id],
    );
    const basename = `selected-chats-${new Date().toISOString().slice(0, 10)}`;
    try {
      if (merged) {
        await exportBulkConversationsMerged(threadIds, format, basename);
      } else {
        await exportBulkConversationsSeparate(threadIds, format, basename);
      }
    } catch (error) {
      if (!isDownloadCancelled(error)) toast.error("Export failed.");
    }
  }

  if (items.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No chats.
      </p>
    );
  }

  const actionsDisabled = busy || selectedCount === 0;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-1 items-center gap-3 px-1">
          <Checkbox
            checked={
              allVisibleSelected
                ? true
                : selectedCount > 0
                  ? "indeterminate"
                  : false
            }
            onCheckedChange={toggleAllVisible}
            aria-label="Select all visible chats"
            title="Select all visible"
          />
          <span className="text-xs text-muted-foreground">
            {selectedCount > 0
              ? `${chatCount(selectedCount)} selected`
              : chatCount(items.length)}
          </span>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild={true}>
            <Button variant="outline" size="sm" disabled={actionsDisabled}>
              <HugeiconsIcon
                icon={Folder01Icon}
                strokeWidth={1.75}
                className="size-3.5 mr-1.5"
              />
              Move
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-52">
            <DropdownMenuItem
              disabled={selectedItems.every((item) => !item.projectId)}
              onSelect={() => void handleMove(null)}
            >
              Recents
            </DropdownMenuItem>
            {projects.map((project) => (
              <DropdownMenuItem
                key={project.id}
                onSelect={() => void handleMove(project.id)}
              >
                <HugeiconsIcon
                  icon={Folder01Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
                <span className="truncate">{project.name}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <Button
          variant="outline"
          size="sm"
          disabled={actionsDisabled}
          onClick={handleTogglePin}
        >
          <HugeiconsIcon
            icon={PinIcon}
            strokeWidth={1.75}
            className="size-3.5 mr-1.5"
          />
          {allSelectedPinned ? "Unpin" : "Pin"}
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={actionsDisabled}
          onClick={() => void handleArchive()}
        >
          <HugeiconsIcon
            icon={Archive02Icon}
            strokeWidth={1.75}
            className="size-3.5 mr-1.5"
          />
          Archive
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild={true}>
            <Button variant="outline" size="sm" disabled={actionsDisabled}>
              <HugeiconsIcon
                icon={Download01Icon}
                strokeWidth={1.75}
                className="size-3.5 mr-1.5"
              />
              Export
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            {COMBINED_EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
              <DropdownMenuItem
                key={`m-${fmt}`}
                onSelect={() => void handleExport(fmt, true)}
              >
                {label} (combined)
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
              <DropdownMenuItem
                key={`s-${fmt}`}
                onSelect={() => void handleExport(fmt, false)}
              >
                {label} (per chat)
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <Button
          variant="outline"
          size="sm"
          disabled={actionsDisabled}
          onClick={() => setConfirmingDelete(true)}
          className="text-destructive hover:text-destructive hover:border-destructive/60"
        >
          <HugeiconsIcon
            icon={Delete02Icon}
            strokeWidth={1.75}
            className="size-3.5 mr-1.5"
          />
          Delete
        </Button>
      </div>

      <div>
        <div className="flex items-center gap-4 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
          <span className="w-4 shrink-0" />
          <span className="flex-1">Name</span>
          <span className="w-28 shrink-0">Project</span>
          <span className="w-32 shrink-0">Date created</span>
        </div>
        {visible.map((item, index) => (
          <div
            key={item.id}
            className="group flex items-center gap-4 border-b border-border/40 px-1 py-2.5 text-sm last:border-0 hover:bg-muted/40"
          >
            <Checkbox
              checked={selectedIds.has(item.id)}
              onClick={(e) => toggleRow(index, e.shiftKey)}
              aria-label={`Select "${item.title}"`}
            />
            <button
              type="button"
              onClick={() => openChat(item)}
              className="min-w-0 flex-1 truncate text-left hover:underline"
              title={item.title}
            >
              {item.title}
            </button>
            <span className="w-28 shrink-0 truncate text-muted-foreground">
              {item.projectId ? (projectNames.get(item.projectId) ?? "") : ""}
            </span>
            <span className="w-32 shrink-0 text-muted-foreground tabular-nums">
              {formatCreatedAt(item.createdAt)}
            </span>
          </div>
        ))}
        {items.length > visibleCount ? (
          <div className="flex justify-center pt-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setVisibleCount(visibleCount + MANAGE_PAGE_SIZE)}
            >
              Show more ({items.length - visibleCount})
            </Button>
          </div>
        ) : null}
      </div>

      <AlertDialog
        open={confirmingDelete}
        onOpenChange={(o) => {
          if (!o) setConfirmingDelete(false);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete {chatCount(selectedCount)}
            </AlertDialogTitle>
            <AlertDialogDescription>
              Delete the{" "}
              {selectedCount === 1
                ? "selected chat"
                : `${selectedCount} selected chats`}
              ? This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmingDelete(false);
                void handleDelete();
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
