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
import { db, useLiveQuery } from "./db";
import type { ChatView, ThreadRecord } from "./types";

interface SidebarItem {
  type: "single" | "compare";
  id: string;
  title: string;
  createdAt: number;
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
      });
    } else if (!t.pairId) {
      items.push({
        type: "single",
        id: t.id,
        title: t.title,
        createdAt: t.createdAt,
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
  const allThreads = useLiveQuery(
    () => db.threads.orderBy("createdAt").reverse().toArray(),
    [],
  );
  const items = groupThreads(allThreads ?? []);
  const activeId = view.mode === "single" ? view.threadId : view.pairId;

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
                <SidebarMenuItem key={item.id}>
                  <SidebarMenuButton
                    isActive={activeId === item.id}
                    onClick={() => onSelect(viewForItem(item))}
                  >
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                  <SidebarMenuAction
                    showOnHover={true}
                    onClick={() => handleDelete(item)}
                    title="Delete"
                  >
                    <HugeiconsIcon icon={Delete02Icon} />
                  </SidebarMenuAction>
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
