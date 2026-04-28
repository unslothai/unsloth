// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  BookOpen02Icon,
  ColumnInsertIcon,
  Delete02Icon,
  MoreHorizontalIcon,
  NewReleasesIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import type { ChatView } from "./types";
import {
  deleteChatItem,
  renameChatItem,
  useChatSidebarItems,
} from "./hooks/use-chat-sidebar-items";
import type { SidebarItem } from "./hooks/use-chat-sidebar-items";

export function ThreadSidebar({
  view,
  onSelect,
  onNewThread,
  onNewCompare,
  showCompare,
}: {
  view: ChatView;
  onSelect: (view: ChatView) => void;
  onNewThread: () => void;
  onNewCompare: () => void;
  showCompare: boolean;
}) {
  const { items } = useChatSidebarItems();
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const activeId =
    view.mode === "single" ? (view.threadId ?? storeThreadId) : view.pairId;
  const [editingTarget, setEditingTarget] = useState<SidebarItem | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [optimisticTitles, setOptimisticTitles] = useState<Record<string, string>>({});
  const [renaming, setRenaming] = useState(false);
  const editInputRef = useRef<HTMLInputElement | null>(null);
  const skipMenuAutoFocusRef = useRef(false);

  useEffect(() => {
    if (!editingTarget) return;
    window.requestAnimationFrame(() => {
      editInputRef.current?.focus();
      editInputRef.current?.select();
    });
  }, [editingTarget]);

  function viewForItem(item: SidebarItem): ChatView {
    return item.type === "single"
      ? { mode: "single", threadId: item.id }
      : { mode: "compare", pairId: item.id };
  }

  async function handleDelete(item: SidebarItem) {
    // Directly set a new view with a nonce rather than going through
    // onNewThread(), which may return early if the guard sees no
    // threadId and no activeThreadId (after we just cleared it).
    await deleteChatItem(item, activeId ?? undefined, onSelect);
  }

  function startRename(item: SidebarItem) {
    skipMenuAutoFocusRef.current = true;
    setEditingTarget(item);
    setEditingTitle(optimisticTitles[item.id] ?? item.title);
  }

  function cancelRename() {
    setEditingTarget(null);
    setEditingTitle("");
  }

  function clearOptimisticTitle(id: string, title: string) {
    window.setTimeout(() => {
      setOptimisticTitles((current) => {
        if (current[id] !== title) return current;
        const next = { ...current };
        delete next[id];
        return next;
      });
    }, 1200);
  }

  async function commitRename() {
    if (!editingTarget || renaming) return;

    const nextTitle = editingTitle.trim();
    if (!nextTitle || nextTitle === editingTarget.title) {
      cancelRename();
      return;
    }

    setRenaming(true);
    const targetId = editingTarget.id;
    setOptimisticTitles((current) => ({
      ...current,
      [targetId]: nextTitle,
    }));
    cancelRename();
    try {
      await renameChatItem(editingTarget, nextTitle);
    } finally {
      setRenaming(false);
      clearOptimisticTitle(targetId, nextTitle);
    }
  }

  return (
    <>
      <SidebarHeader className="px-4 py-3">
        <span className="text-base font-semibold tracking-tight">Playground</span>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup className="px-4 pt-1">
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton onClick={onNewThread}>
                  <HugeiconsIcon icon={PencilEdit02Icon} />
                  <span>New Chat</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              {showCompare ? (
                <SidebarMenuItem>
                  <SidebarMenuButton data-tour="chat-compare" onClick={onNewCompare}>
                    <HugeiconsIcon icon={ColumnInsertIcon} />
                    <span>Compare</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ) : null}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup className="flex-1 px-4">
          <SidebarGroupLabel className="text-xs font-medium text-muted-foreground/80">Your Chats</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.id} className="group/thread-item relative">
                  {(() => {
                    const displayTitle = optimisticTitles[item.id] ?? item.title;
                    return editingTarget?.id === item.id ? (
                    <div className="flex h-8 items-center rounded-md bg-accent px-2 pr-8">
                      <input
                        ref={editInputRef}
                        value={editingTitle}
                        onChange={(event) => setEditingTitle(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            void commitRename();
                          } else if (event.key === "Escape") {
                            event.preventDefault();
                            cancelRename();
                          }
                        }}
                        disabled={renaming}
                        maxLength={80}
                        aria-label="Chat title"
                        className="h-full min-w-0 flex-1 bg-transparent p-0 text-sm text-foreground outline-none selection:bg-sky-200 dark:selection:bg-sky-500/45"
                      />
                    </div>
                  ) : (
                    <SidebarMenuButton
                      isActive={activeId === item.id}
                      onClick={() => onSelect(viewForItem(item))}
                      className="pr-8"
                    >
                      <span>{displayTitle}</span>
                    </SidebarMenuButton>
                  );
                  })()}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        type="button"
                        onClick={(event) => event.stopPropagation()}
                        aria-label={`Open options for ${optimisticTitles[item.id] ?? item.title}`}
                        className="absolute right-1 top-1/2 flex size-7 -translate-y-1/2 items-center justify-center text-sidebar-foreground/45 opacity-0 transition-colors duration-150 hover:text-sidebar-foreground/80 focus-visible:opacity-100 focus-visible:text-sidebar-foreground/80 focus-visible:outline-none data-[state=open]:text-sidebar-foreground/80 data-[state=open]:opacity-100 group-hover/thread-item:opacity-100"
                      >
                        <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={2.15} className="size-[18px]" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                      align="end"
                      sideOffset={6}
                      onCloseAutoFocus={(event) => {
                        if (!skipMenuAutoFocusRef.current) return;
                        event.preventDefault();
                        skipMenuAutoFocusRef.current = false;
                      }}
                      className="w-40"
                    >
                      <DropdownMenuItem onSelect={() => startRename(item)}>
                        <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} />
                        <span>Rename</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        variant="destructive"
                        onSelect={() => {
                          void handleDelete(item);
                        }}
                      >
                        <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} />
                        <span>Delete</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
            {items.length === 0 && (
              <p className="px-2 py-6 text-center text-xs text-muted-foreground">
                No threads yet
              </p>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="space-y-1 px-4 pb-3">
        <a
          href="https://unsloth.ai/docs/new/studio/chat"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 corner-squircle rounded-md px-2 py-1.5 text-xs font-medium text-primary bg-primary/10 transition-colors hover:bg-primary/20"
        >
          <HugeiconsIcon icon={BookOpen02Icon} className="size-4 shrink-0" strokeWidth={2} />
          <span>Learn more in docs</span>
        </a>
        <a
          href="https://unsloth.ai/docs/new/changelog"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <HugeiconsIcon icon={NewReleasesIcon} className="size-4 shrink-0" strokeWidth={2} />
          <span>What&apos;s new</span>
        </a>
      </SidebarFooter>
    </>
  );
}
