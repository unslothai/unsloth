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
import {
  type SidebarItem,
  type ProjectRecord,
  listStoredChatProjects,
  DeleteChatFilesSwitch,
  deleteChatItem,
  unarchiveChatItem,
  useChatPreferencesStore,
  useChatProjects,
  useChatRuntimeStore,
  useChatSidebarItems,
} from "@/features/chat";
import { toast } from "@/lib/toast";
import { Delete02Icon, Message01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import {
  DEFAULT_LIBRARY_FILTERS,
  filterLibraryItems,
  type LibraryFilters,
} from "./data-library";
import {
  ChatLibraryGroups,
  LibraryRow,
  LibraryToolbar,
} from "./data-library-controls";

const ARCHIVED_PAGE_SIZE = 20;

export function ArchivedChatsView() {
  const { archivedItems } = useChatSidebarItems({ requireMessages: false });
  const { projects: activeProjects } = useChatProjects();
  const [projects, setProjects] = useState<ProjectRecord[]>(activeProjects);
  useEffect(() => {
    let cancelled = false;
    void listStoredChatProjects({ includeArchived: true })
      .then((all) => {
        if (!cancelled) setProjects(all);
      })
      .catch((error) => {
        if (!cancelled)
          toast.error("Failed to load archived projects", {
            description: error instanceof Error ? error.message : undefined,
          });
      });
    return () => {
      cancelled = true;
    };
  }, [activeProjects]);
  const navigate = useNavigate();
  const closeSettings = useSettingsDialogStore((s) => s.closeDialog);
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
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
  const alwaysDeleteChatFiles = useChatPreferencesStore(
    (s) => s.alwaysDeleteChatFiles,
  );
  const [confirmingDelete, setConfirmingDelete] = useState<SidebarItem[]>([]);
  const [deleteFilesOnDelete, setDeleteFilesOnDelete] = useState(false);
  const [visibleCount, setVisibleCount] = useState(ARCHIVED_PAGE_SIZE);
  const [filters, setFilters] = useState(DEFAULT_LIBRARY_FILTERS);
  const [removed, setRemoved] = useState<ReadonlySet<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const running = useRef(false);
  const projectNames = useMemo(
    () => new Map(projects.map((p) => [p.id, p.name])),
    [projects],
  );
  const filtered = useMemo(
    () =>
      filterLibraryItems(
        archivedItems.filter((item) => !removed.has(item.id)),
        filters,
        projectNames,
      ),
    [archivedItems, removed, filters, projectNames],
  );
  const narrowed =
    filters.query.trim() !== "" ||
    filters.type !== "all" ||
    filters.project !== "all";

  function changeFilters(next: LibraryFilters) {
    setFilters(next);
    setVisibleCount(ARCHIVED_PAGE_SIZE);
  }

  function openChat(item: SidebarItem) {
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

  async function handleDelete(item: SidebarItem, deleteFiles: boolean) {
    await deleteChatItem(
      item,
      openChatId,
      (view) => {
        navigate({
          to: "/chat",
          search: item.projectId
            ? { project: item.projectId }
            : { new: view.newThreadNonce },
        });
      },
      { deleteFiles },
    );
  }

  async function run(
    items: SidebarItem[],
    action: "delete" | "restore",
    deleteFiles = false,
  ) {
    if (running.current || items.length === 0) return;
    running.current = true;
    setBusy(true);
    let completed = 0;
    try {
      for (const item of items) {
        if (action === "delete") await handleDelete(item, deleteFiles);
        else await unarchiveChatItem(item);
        setRemoved((previous) => new Set([...previous, item.id]));
        completed += 1;
      }
      toast.success(
        `${completed} ${completed === 1 ? "chat" : "chats"} ${action === "delete" ? "deleted" : "unarchived"}`,
      );
    } catch (err) {
      toast.error(
        `Could not ${action} ${items.length - completed} ${items.length - completed === 1 ? "chat" : "chats"}`,
        {
          description: err instanceof Error ? err.message : undefined,
        },
      );
    } finally {
      running.current = false;
      setBusy(false);
    }
  }

  function requestDelete(items: SidebarItem[], bulk = false) {
    if (running.current) return;
    if (bulk || confirmDeleteChats) {
      setDeleteFilesOnDelete(alwaysDeleteChatFiles);
      setConfirmingDelete([...items]);
    } else void run(items, "delete", alwaysDeleteChatFiles);
  }

  return (
    <div className="flex flex-col gap-4">
      <LibraryToolbar
        filters={filters}
        onChange={changeFilters}
        placeholder="Search archived chats or projects"
        projects={projectNames}
        disabled={busy}
      />
      <div className="flex flex-wrap items-center gap-2">
        <span role="status" className="flex-1 text-xs text-muted-foreground">
          {filtered.length} {filtered.length === 1 ? "chat" : "chats"}
        </span>
        <Button
          variant="ghost"
          size="sm"
          disabled={busy || filtered.length === 0}
          onClick={() => void run([...filtered], "restore")}
        >
          {narrowed ? "Unarchive results" : "Unarchive all"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="text-destructive hover:bg-destructive/10 hover:text-destructive"
          disabled={busy || filtered.length === 0}
          onClick={() => requestDelete(filtered, true)}
        >
          <HugeiconsIcon icon={Delete02Icon} className="mr-1.5 size-4" />
          {narrowed ? "Delete results" : "Delete all"}
        </Button>
      </div>
      {filtered.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          {narrowed
            ? "No archived chats match your search."
            : "No archived chats."}
        </p>
      ) : (
        <ChatLibraryGroups
          items={filtered.slice(0, visibleCount)}
          projects={projectNames}
        >
          {(item) => (
            <LibraryRow
              title={item.title}
              date={
                filters.sort === "updated" ? item.updatedAt : item.createdAt
              }
              onOpen={() => openChat(item)}
              leading={
                <HugeiconsIcon
                  icon={Message01Icon}
                  className="size-4 shrink-0 text-muted-foreground"
                />
              }
              actions={
                <>
                  <Button
                    variant="ghost"
                    size="icon"
                    disabled={busy}
                    onClick={() => requestDelete([item])}
                    aria-label={`Delete chat: ${item.title}`}
                    title="Delete chat"
                    className="text-muted-foreground hover:text-destructive"
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={busy}
                    onClick={() => void run([item], "restore")}
                    aria-label={`Unarchive chat: ${item.title}`}
                    className="rounded-xl bg-muted/60 hover:bg-muted"
                  >
                    Unarchive
                  </Button>
                </>
              }
            />
          )}
        </ChatLibraryGroups>
      )}
      {filtered.length > visibleCount && (
        <div className="flex justify-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setVisibleCount((count) => count + ARCHIVED_PAGE_SIZE)
            }
          >
            Show more ({filtered.length - visibleCount})
          </Button>
        </div>
      )}
      <AlertDialog
        open={confirmingDelete.length > 0}
        onOpenChange={(open) => {
          if (!open && !busy) setConfirmingDelete([]);
        }}
      >
        <AlertDialogContent
          onEscapeKeyDown={(event) => {
            if (busy) event.preventDefault();
          }}
        >
          <AlertDialogHeader>
            <AlertDialogTitle>
              {confirmingDelete.length === 1
                ? "Delete chat"
                : `Delete ${confirmingDelete.length} archived chats`}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {confirmingDelete.length === 1
                ? `Delete "${confirmingDelete[0]?.title}"?`
                : `Permanently delete these ${confirmingDelete.length} archived chats?`}{" "}
              This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <fieldset disabled={busy}>
            <DeleteChatFilesSwitch
              id="archived-chat-delete-files"
              checked={deleteFilesOnDelete}
              onCheckedChange={setDeleteFilesOnDelete}
            />
          </fieldset>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={busy}
              onClick={(event) => {
                event.preventDefault();
                void run(confirmingDelete, "delete", deleteFilesOnDelete).then(
                  () => setConfirmingDelete([]),
                );
              }}
            >
              {busy ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
