// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import {
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  BookOpen02Icon,
  ColumnInsertIcon,
  Delete02Icon,
  Download01Icon,
  MoreHorizontalIcon,
  NewReleasesIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "sonner";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import type { ChatView } from "./types";
import {
  deleteChatItem,
  renameChatItem,
  useChatSidebarItems,
} from "./hooks/use-chat-sidebar-items";
import type { SidebarItem } from "./hooks/use-chat-sidebar-items";
import {
  exportConversationRawJsonl,
  exportConversationCsv,
  exportConversationShareGPT,
  exportBulkConversationsMerged,
  exportBulkConversationsSeparate,
  EXPORT_FORMATS_LIST,
  type ConvExportFormat,
} from "./prompt-storage/prompt-storage-dialog";
import {
  listStoredChatThreads,
} from "./utils/chat-history-storage";

const EXPORT_FORMATS = [
  { label: "Raw JSONL", fn: exportConversationRawJsonl },
  { label: "CSV", fn: exportConversationCsv },
  { label: "ShareGPT JSONL", fn: exportConversationShareGPT },
] as const;

async function getThreadIdsForItem(item: SidebarItem): Promise<string[]> {
  if (item.type === "single") return [item.id];
  const threads = await listStoredChatThreads({ pairId: item.id });
  return threads.map((t) => t.id);
}

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
    view.mode === "single"
      ? (view.threadId ?? storeThreadId)
      : view.mode === "compare"
        ? view.pairId
        : view.projectId;

  const [renamingItem, setRenamingItem] = useState<SidebarItem | null>(null);
  const [renameDraft, setRenameDraft] = useState("");

  function viewForItem(item: SidebarItem): ChatView {
    return item.type === "single"
      ? { mode: "single", threadId: item.id }
      : { mode: "compare", pairId: item.id };
  }

  async function handleDelete(item: SidebarItem) {
    await deleteChatItem(item, activeId ?? undefined, onSelect);
  }

  function openRename(item: SidebarItem) {
    setRenameDraft(item.title);
    setRenamingItem(item);
  }

  async function commitRename() {
    if (!renamingItem) return;
    try {
      await renameChatItem(renamingItem, renameDraft);
    } catch {
      toast.error("Failed to rename chat.");
    } finally {
      setRenamingItem(null);
    }
  }

  async function handleExport(
    item: SidebarItem,
    fn: (threadId: string) => Promise<void>,
  ) {
    try {
      const ids = await getThreadIdsForItem(item);
      for (const id of ids) {
        await fn(id);
      }
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error("Export failed.");
      }
    }
  }

  async function getBulkThreadIds(scope: "recents" | "all"): Promise<string[]> {
    const threads = await listStoredChatThreads({
      includeArchived: false,
      ...(scope === "recents" ? { projectId: null } : {}),
    });
    return [...new Set(threads.map((t) => t.id))];
  }

  async function handleBulkExport(
    scope: "recents" | "all",
    fmt: ConvExportFormat,
    merged: boolean,
  ) {
    try {
      const ids = await getBulkThreadIds(scope);
      if (ids.length === 0) { toast.info("No conversations to export."); return; }
      const ts = new Date().toISOString().slice(0, 10);
      const basename = `${scope === "all" ? "all-chats" : "recents"}-${ts}`;
      if (merged) {
        await exportBulkConversationsMerged(ids, fmt, basename);
      } else {
        await exportBulkConversationsSeparate(ids, fmt, basename);
      }
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error("Export failed.");
      }
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
          {/* Recents label with export-all menu */}
          <div className="flex items-center justify-between px-2 py-1.5">
            <span className="text-xs font-medium text-muted-foreground/80">Recents</span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  className="flex items-center justify-center rounded-sm p-0.5 text-muted-foreground hover:bg-accent focus:outline-none focus-visible:ring-0"
                  title="Export options"
                >
                  <HugeiconsIcon icon={MoreHorizontalIcon} className="size-3.5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="bottom" align="end" className="w-56">
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger>
                    <HugeiconsIcon icon={Download01Icon} className="mr-2 size-4" />
                    Export Recents
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent avoidCollisions={false} className="w-52">
                    {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`r-m-${fmt}`} onSelect={() => void handleBulkExport("recents", fmt, true)}>
                        {label} — combined
                      </DropdownMenuItem>
                    ))}
                    <DropdownMenuSeparator />
                    {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`r-s-${fmt}`} onSelect={() => void handleBulkExport("recents", fmt, false)}>
                        {label} — per chat
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger>
                    <HugeiconsIcon icon={Download01Icon} className="mr-2 size-4" />
                    Export Recents + Projects
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent avoidCollisions={false} className="w-52">
                    {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`a-m-${fmt}`} onSelect={() => void handleBulkExport("all", fmt, true)}>
                        {label} — combined
                      </DropdownMenuItem>
                    ))}
                    <DropdownMenuSeparator />
                    {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
                      <DropdownMenuItem key={`a-s-${fmt}`} onSelect={() => void handleBulkExport("all", fmt, false)}>
                        {label} — per chat
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.id}>
                  <SidebarMenuButton
                    isActive={activeId === item.id}
                    onClick={() => onSelect(viewForItem(item))}
                  >
                    {item.isFork ? (
                      <span
                        className="mr-1 rounded-sm bg-primary/10 px-1.5 py-0.5 text-[0.625rem] font-semibold uppercase tracking-wide text-primary"
                        title="Forked from another chat"
                      >
                        fork
                      </span>
                    ) : null}
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <SidebarMenuAction showOnHover className="focus:outline-none focus-visible:ring-0" onClick={(e) => e.stopPropagation()}>
                        <HugeiconsIcon icon={MoreHorizontalIcon} className="size-4" />
                        <span className="sr-only">More options</span>
                      </SidebarMenuAction>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent side="bottom" align="end" className="w-44">
                      <DropdownMenuItem onSelect={() => openRename(item)}>
                        <HugeiconsIcon icon={PencilEdit02Icon} className="mr-2 size-4" />
                        Rename
                      </DropdownMenuItem>
                      <DropdownMenuSub>
                        <DropdownMenuSubTrigger>
                          <HugeiconsIcon icon={Download01Icon} className="mr-2 size-4" />
                          Export
                        </DropdownMenuSubTrigger>
                        <DropdownMenuSubContent avoidCollisions={false} className="w-52">
                          {EXPORT_FORMATS.map(({ label, fn }) => (
                            <DropdownMenuItem
                              key={label}
                              onSelect={() => handleExport(item, fn)}
                            >
                              {label}
                            </DropdownMenuItem>
                          ))}
                        </DropdownMenuSubContent>
                      </DropdownMenuSub>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive focus:text-destructive"
                        onSelect={() => void handleDelete(item)}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="mr-2 size-4" />
                        Delete
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

      {/* Rename dialog */}
      <Dialog open={renamingItem !== null} onOpenChange={(open) => { if (!open) setRenamingItem(null); }}>
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Rename chat</DialogTitle>
          </DialogHeader>
          <Input
            value={renameDraft}
            onChange={(e) => setRenameDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void commitRename(); }}
            autoFocus
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenamingItem(null)}>Cancel</Button>
            <Button
              onClick={() => void commitRename()}
              disabled={!renameDraft.trim() || renameDraft.trim() === renamingItem?.title}
            >
              Rename
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
