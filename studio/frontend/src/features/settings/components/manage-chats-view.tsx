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
import { useLocale, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  Archive02Icon,
  Delete02Icon,
  Download01Icon,
  Folder01Icon,
  PinIcon,
} from "@hugeicons/core-free-icons";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

import { useLibraryProjectLabels } from "./use-library-project-labels";
import {
  DEFAULT_LIBRARY_FILTERS,
  filterLibraryItems,
  groupLibraryItems,
  type LibraryFilters,
} from "./data-library";
import {
  ChatLibraryGroups,
  LibraryRow,
  LibraryToolbar,
} from "./data-library-controls";

const MANAGE_PAGE_SIZE = 20;

export function ManageChatsView() {
  const t = useT();
  const locale = useLocale();
  const labels = useLibraryProjectLabels();
  const chatCount = (count: number) =>
    t(
      count === 1
        ? "settings.data.library.oneChat"
        : "settings.data.library.chatCount",
      { count },
    );
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

  const [filters, setFilters] = useState(DEFAULT_LIBRARY_FILTERS);
  const projectNames = useMemo(
    () => new Map(projects.map((p) => [p.id, p.name])),
    [projects],
  );
  const filtered = useMemo(
    () => filterLibraryItems(items, filters, projectNames, labels, locale),
    [items, filters, projectNames, labels, locale],
  );
  const visible = groupLibraryItems(
    filtered.slice(0, visibleCount),
    projectNames,
    labels,
  ).flatMap((group) => group.items);
  const selectedItems = items.filter((item) => selectedIds.has(item.id));

  function changeFilters(next: LibraryFilters) {
    setFilters(next);
    setSelectedIds(new Set());
    lastToggledId.current = null;
    setVisibleCount(MANAGE_PAGE_SIZE);
  }
  const selectedCount = selectedItems.length;
  const visibleSelectedCount = visible.filter((item) =>
    selectedIds.has(item.id),
  ).length;
  const allVisibleSelected =
    visible.length > 0 && visibleSelectedCount === visible.length;
  const allSelectedPinned =
    selectedCount > 0 &&
    selectedItems.every((item) => pinnedIds.includes(item.id));

  function toggleRow(index: number, shiftKey: boolean) {
    if (busy) return;
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
    if (busy) return;
    setSelectedIds((previous) => {
      const next = new Set(previous);
      for (const item of visible) {
        if (allVisibleSelected) next.delete(item.id);
        else next.add(item.id);
      }
      return next;
    });
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
      t(
        selectedCount === 1
          ? "settings.data.archivedOneChat"
          : "settings.data.archivedChatCount",
        { count: selectedCount },
      ),
      t("settings.data.failedToArchiveChats"),
    );

  const handleDelete = () =>
    run(
      () =>
        // Honour the same preference as the sidebar delete, or the selected
        // chats leave orphan sandbox folders behind.
        deleteChatItems(selectedItems, openChatId, resetView, {
          deleteFiles: alwaysDeleteChatFiles,
        }),
      t("settings.data.library.deletedChats", { count: selectedCount }),
      t("settings.data.library.deleteFailed"),
    );

  const handleMove = (projectId: string | null) =>
    run(
      async () => {
        await Promise.all(
          selectedItems.map((item) => moveChatItemToProject(item, projectId)),
        );
      },
      projectId
        ? t("settings.data.library.movedChatsToProject", {
            count: selectedCount,
            project: projectNames.get(projectId) ?? labels.unavailableProject,
          })
        : t("settings.data.library.movedChatsToRecents", {
            count: selectedCount,
          }),
      t("settings.data.library.moveFailed"),
    );

  function handleTogglePin() {
    const ids = selectedItems.map((item) => item.id);
    setPinned(ids, !allSelectedPinned);
    toast.success(
      t(
        allSelectedPinned
          ? "settings.data.library.unpinnedChats"
          : "settings.data.library.pinnedChats",
        { count: ids.length },
      ),
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
      if (!isDownloadCancelled(error))
        toast.error(t("settings.data.exportFailed"));
    }
  }

  const actionsDisabled = busy || selectedCount === 0;

  return (
    <div className="flex flex-col gap-4">
      <LibraryToolbar
        filters={filters}
        onChange={changeFilters}
        placeholder={t("settings.data.library.searchChats")}
        projects={projectNames}
        disabled={busy}
      />
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-1 items-center gap-3 px-1">
          <Checkbox
            checked={
              allVisibleSelected
                ? true
                : visibleSelectedCount > 0
                  ? "indeterminate"
                  : false
            }
            disabled={busy || visible.length === 0}
            onCheckedChange={toggleAllVisible}
            aria-label={t("settings.data.library.selectAll")}
            title={t("settings.data.library.selectAll")}
          />
          <span className="text-xs text-muted-foreground">
            {selectedCount > 0
              ? t("settings.data.library.selectedChats", {
                  count: selectedCount,
                })
              : chatCount(filtered.length)}
          </span>
        </div>
        {selectedCount > 0 && (
          <>
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <Button variant="outline" size="sm" disabled={actionsDisabled}>
                  <HugeiconsIcon
                    icon={Folder01Icon}
                    strokeWidth={1.75}
                    className="size-3.5 mr-1.5"
                  />
                  {t("settings.data.library.move")}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52">
                <DropdownMenuItem
                  disabled={selectedItems.every((item) => !item.projectId)}
                  onSelect={() => void handleMove(null)}
                >
                  {t("shell.navigation.recents")}
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
              {allSelectedPinned
                ? t("settings.data.library.unpin")
                : t("settings.data.library.pin")}
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
              {t("settings.data.library.archive")}
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <Button variant="outline" size="sm" disabled={actionsDisabled}>
                  <HugeiconsIcon
                    icon={Download01Icon}
                    strokeWidth={1.75}
                    className="size-3.5 mr-1.5"
                  />
                  {t("common.export")}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                {COMBINED_EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                  <DropdownMenuItem
                    key={`m-${fmt}`}
                    onSelect={() => void handleExport(fmt, true)}
                  >
                    {label} {t("settings.chat.exportCombinedSuffix")}
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                  <DropdownMenuItem
                    key={`s-${fmt}`}
                    onSelect={() => void handleExport(fmt, false)}
                  >
                    {label} {t("settings.chat.exportPerChatSuffix")}
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
              {t("common.delete")}
            </Button>
          </>
        )}
      </div>

      {filtered.length === 0 ? (
        <p
          role="status"
          className="py-8 text-center text-sm text-muted-foreground"
        >
          {t("settings.data.library.noChats")}
        </p>
      ) : (
        <ChatLibraryGroups items={visible} projects={projectNames}>
          {(item) => (
            <LibraryRow
              title={item.title}
              date={
                filters.sort === "updated" ? item.updatedAt : item.createdAt
              }
              onOpen={() => openChat(item)}
              leading={
                <>
                  <Checkbox
                    checked={selectedIds.has(item.id)}
                    disabled={busy}
                    onClick={(event) =>
                      toggleRow(
                        visible.findIndex((row) => row.id === item.id),
                        event.shiftKey,
                      )
                    }
                    aria-label={t("settings.data.library.selectItem", {
                      title: item.title,
                    })}
                  />
                  <HugeiconsIcon
                    icon={MessageCircleIcon}
                    className="size-4 shrink-0 text-muted-foreground"
                  />
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
            onClick={() => setVisibleCount((count) => count + MANAGE_PAGE_SIZE)}
          >
            {t("settings.voice.recents.showMore", {
              count: filtered.length - visibleCount,
            })}
          </Button>
        </div>
      )}

      <AlertDialog
        open={confirmingDelete}
        onOpenChange={(o) => {
          if (!o) setConfirmingDelete(false);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.data.library.deleteChatsTitle", {
                count: selectedCount,
              })}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("settings.data.library.deleteChatsWarning", {
                count: selectedCount,
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmingDelete(false);
                void handleDelete();
              }}
            >
              {t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
