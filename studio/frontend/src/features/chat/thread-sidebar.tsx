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
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import {
  BookOpen02Icon,
  ColumnInsertIcon,
  Delete02Icon,
  NewReleasesIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ChevronRightIcon,
  DownloadIcon,
  FolderIcon,
  FolderPlusIcon,
  MoreHorizontalIcon,
  PinIcon,
  PinOffIcon,
  SearchIcon,
  XIcon,
} from "lucide-react";
import { useCallback, useMemo, useRef, useState } from "react";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
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
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { db, useLiveQuery } from "./db";
import type { ChatView, FolderRecord, ThreadRecord } from "./types";
import {
  exportAsJSON,
  exportAsJSONL,
  exportAsMarkdown,
  getExportThreadIds,
} from "./lib/thread-export";

interface SidebarItem {
  type: "single" | "compare";
  id: string;
  title: string;
  createdAt: number;
  folderId?: string;
  pinned?: boolean;
}

function groupThreads(threads: ThreadRecord[]): SidebarItem[] {
  const items: SidebarItem[] = [];
  const seenPairs = new Set<string>();

  for (const t of threads) {
    if (t.archived) {
      continue;
    }
    if (t.pairId) {
      if (seenPairs.has(t.pairId)) {
        continue;
      }
      seenPairs.add(t.pairId);
      items.push({
        type: "compare",
        id: t.pairId,
        title: t.title,
        createdAt: t.createdAt,
        folderId: t.folderId,
        pinned: t.pinned,
      });
    } else if (!t.pairId) {
      items.push({
        type: "single",
        id: t.id,
        title: t.title,
        createdAt: t.createdAt,
        folderId: t.folderId,
        pinned: t.pinned,
      });
    }
  }

  return items.sort((a, b) => b.createdAt - a.createdAt);
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
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const debouncedQuery = useDebouncedValue(searchQuery, 150);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const allThreads = useLiveQuery(
    () => db.threads.orderBy("createdAt").reverse().toArray(),
    [],
  );
  const allFolders = useLiveQuery(
    () => db.folders.orderBy("createdAt").toArray(),
    [],
  );

  const items = useMemo(() => groupThreads(allThreads ?? []), [allThreads]);
  const activeId = view.mode === "single" ? view.threadId : view.pairId;

  const filteredItems = useMemo(() => {
    if (!debouncedQuery.trim()) return items;
    const q = debouncedQuery.toLowerCase();
    // Filter items by matching title or searchText from underlying threads
    return items.filter((item) => {
      if (item.title.toLowerCase().includes(q)) return true;
      // Check searchText on the underlying thread records
      const thread = (allThreads ?? []).find(
        (t) => t.id === item.id || t.pairId === item.id,
      );
      return thread?.searchText?.toLowerCase().includes(q) ?? false;
    });
  }, [items, debouncedQuery, allThreads]);

  // Group items: pinned, then by folder, then unfiled
  const pinnedItems = useMemo(
    () => filteredItems.filter((i) => i.pinned),
    [filteredItems],
  );
  const folderedItems = useMemo(() => {
    const map = new Map<string, SidebarItem[]>();
    for (const item of filteredItems) {
      if (item.pinned || !item.folderId) continue;
      const list = map.get(item.folderId) ?? [];
      list.push(item);
      map.set(item.folderId, list);
    }
    return map;
  }, [filteredItems]);
  const unfiledItems = useMemo(
    () => filteredItems.filter((i) => !i.pinned && !i.folderId),
    [filteredItems],
  );

  const folders = allFolders ?? [];

  function viewForItem(item: SidebarItem): ChatView {
    return item.type === "single"
      ? { mode: "single", threadId: item.id }
      : { mode: "compare", pairId: item.id };
  }

  async function handleDelete(item: SidebarItem) {
    if (item.type === "single") {
      await db.messages.where("threadId").equals(item.id).delete();
      await db.threads.delete(item.id);
    } else {
      const paired = await db.threads.where("pairId").equals(item.id).toArray();
      for (const t of paired) {
        await db.messages.where("threadId").equals(t.id).delete();
        await db.threads.delete(t.id);
      }
    }
    if (activeId === item.id) {
      onSelect({ mode: "single" });
    }
  }

  async function handlePin(item: SidebarItem) {
    const next = !item.pinned;
    if (item.type === "single") {
      await db.threads.update(item.id, { pinned: next });
    } else {
      const paired = await db.threads.where("pairId").equals(item.id).toArray();
      for (const t of paired) {
        await db.threads.update(t.id, { pinned: next });
      }
    }
  }

  async function handleMoveToFolder(item: SidebarItem, folderId: string | undefined) {
    if (item.type === "single") {
      await db.threads.update(item.id, { folderId: folderId ?? "" });
    } else {
      const paired = await db.threads.where("pairId").equals(item.id).toArray();
      for (const t of paired) {
        await db.threads.update(t.id, { folderId: folderId ?? "" });
      }
    }
  }

  async function handleExport(
    item: SidebarItem,
    format: "md" | "json" | "jsonl",
  ) {
    const ids = await getExportThreadIds(item.id, item.type);
    for (const id of ids) {
      if (format === "md") await exportAsMarkdown(id);
      else if (format === "json") await exportAsJSON(id);
      else await exportAsJSONL(id);
    }
  }

  const handleNewFolder = useCallback(async () => {
    const name = prompt("Folder name:");
    if (!name?.trim()) return;
    await db.folders.add({
      id: crypto.randomUUID(),
      name: name.trim(),
      createdAt: Date.now(),
    });
  }, []);

  const handleDeleteFolder = useCallback(async (folderId: string) => {
    // Unfile threads in the folder, then delete the folder
    const threads = await db.threads.where("folderId").equals(folderId).toArray();
    for (const t of threads) {
      await db.threads.update(t.id, { folderId: "" });
    }
    await db.folders.delete(folderId);
  }, []);

  const toggleSearch = useCallback(() => {
    setSearchOpen((o) => {
      if (!o) {
        setTimeout(() => searchInputRef.current?.focus(), 50);
      } else {
        setSearchQuery("");
      }
      return !o;
    });
  }, []);

  const renderItem = (item: SidebarItem) => (
    <SidebarMenuItem key={item.id}>
      <SidebarMenuButton
        isActive={activeId === item.id}
        onClick={() => onSelect(viewForItem(item))}
      >
        <span className="truncate">{item.title}</span>
      </SidebarMenuButton>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <SidebarMenuAction
            showOnHover={true}
            title="More actions"
          >
            <MoreHorizontalIcon className="size-4" />
          </SidebarMenuAction>
        </DropdownMenuTrigger>
        <DropdownMenuContent side="right" align="start" className="w-48">
          <DropdownMenuItem onClick={() => handlePin(item)}>
            {item.pinned ? (
              <><PinOffIcon className="mr-2 size-4" />Unpin</>
            ) : (
              <><PinIcon className="mr-2 size-4" />Pin to top</>
            )}
          </DropdownMenuItem>
          {folders.length > 0 && (
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <FolderIcon className="mr-2 size-4" />Move to folder
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                <DropdownMenuItem onClick={() => handleMoveToFolder(item, undefined)}>
                  None (unfiled)
                </DropdownMenuItem>
                {folders.map((f) => (
                  <DropdownMenuItem key={f.id} onClick={() => handleMoveToFolder(item, f.id)}>
                    {f.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          )}
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <DownloadIcon className="mr-2 size-4" />Export
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuItem onClick={() => handleExport(item, "md")}>
                Markdown (.md)
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleExport(item, "json")}>
                JSON (.json)
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleExport(item, "jsonl")}>
                JSONL (.jsonl)
              </DropdownMenuItem>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="text-destructive focus:text-destructive"
            onClick={() => handleDelete(item)}
          >
            <HugeiconsIcon icon={Delete02Icon} className="mr-2 size-4" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </SidebarMenuItem>
  );

  return (
    <>
      <SidebarHeader className="px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="text-base font-semibold tracking-tight">Playground</span>
          <button
            type="button"
            onClick={toggleSearch}
            className="flex size-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            title="Search conversations"
          >
            {searchOpen ? <XIcon className="size-4" /> : <SearchIcon className="size-4" />}
          </button>
        </div>
        {searchOpen && (
          <div className="mt-2">
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chats..."
              className="w-full rounded-md border bg-background px-3 py-1.5 text-sm outline-none placeholder:text-muted-foreground focus:ring-1 focus:ring-ring"
            />
          </div>
        )}
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
              <SidebarMenuItem>
                <SidebarMenuButton onClick={handleNewFolder}>
                  <FolderPlusIcon className="size-4" />
                  <span>New Folder</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Pinned threads */}
        {pinnedItems.length > 0 && (
          <SidebarGroup className="px-4">
            <SidebarGroupLabel className="text-xs font-medium text-muted-foreground/80">
              Pinned
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>{pinnedItems.map(renderItem)}</SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}

        {/* Folders */}
        {folders.map((folder) => {
          const folderItems = folderedItems.get(folder.id) ?? [];
          if (folderItems.length === 0 && debouncedQuery.trim()) return null;
          return (
            <SidebarGroup key={folder.id} className="px-4">
              <Collapsible defaultOpen={true}>
                <div className="flex items-center justify-between">
                  <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-muted-foreground/80 hover:text-foreground">
                    <ChevronRightIcon className="size-3 transition-transform [[data-state=open]>&]:rotate-90" />
                    <FolderIcon className="size-3" />
                    {folder.name}
                    <span className="text-muted-foreground/50">({folderItems.length})</span>
                  </CollapsibleTrigger>
                  <button
                    type="button"
                    onClick={() => handleDeleteFolder(folder.id)}
                    className="size-5 flex items-center justify-center rounded text-muted-foreground/50 hover:text-destructive"
                    title="Delete folder"
                  >
                    <XIcon className="size-3" />
                  </button>
                </div>
                <CollapsibleContent>
                  <SidebarGroupContent>
                    <SidebarMenu>{folderItems.map(renderItem)}</SidebarMenu>
                    {folderItems.length === 0 && (
                      <p className="px-2 py-2 text-center text-xs text-muted-foreground/50">
                        Empty
                      </p>
                    )}
                  </SidebarGroupContent>
                </CollapsibleContent>
              </Collapsible>
            </SidebarGroup>
          );
        })}

        {/* Unfiled threads */}
        <SidebarGroup className="flex-1 px-4">
          <SidebarGroupLabel className="text-xs font-medium text-muted-foreground/80">
            {folders.length > 0 ? "Unfiled" : "Your Chats"}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {unfiledItems.map(renderItem)}
            </SidebarMenu>
            {filteredItems.length === 0 && (
              <p className="px-2 py-6 text-center text-xs text-muted-foreground">
                {debouncedQuery.trim() ? "No matching threads" : "No threads yet"}
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
          href="https://unsloth.ai/blog"
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
