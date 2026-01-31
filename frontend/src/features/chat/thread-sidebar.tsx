import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  ChatAdd01Icon,
  Delete02Icon,
  Message02Icon,
  MessageMultiple01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { db, useLiveQuery } from "./db";
import type { ChatView, ThreadRecord } from "./types";

interface SidebarItem {
  type: "single" | "compare";
  id: string; // threadId for single, pairId for compare
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
    } else if (t.modelType === "base") {
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
}: {
  view: ChatView;
  onSelect: (view: ChatView) => void;
  onNewThread: () => void;
  onNewCompare: () => void;
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
      <div className="flex items-center px-3 pt-2.5 pb-1.5">
        <span className="flex-1 text-xs font-medium text-foreground">
          Chats
        </span>
        <DropdownMenu>
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              className="flex items-center justify-center rounded-md p-1 text-muted-foreground hover:bg-accent transition-colors"
              title="New chat"
            >
              <HugeiconsIcon icon={ChatAdd01Icon} className="size-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={onNewThread}>
              <HugeiconsIcon icon={Message02Icon} className="size-4" />
              New Chat
            </DropdownMenuItem>
            <DropdownMenuItem onClick={onNewCompare}>
              <HugeiconsIcon icon={MessageMultiple01Icon} className="size-4" />
              Compare Mode
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="mx-2 border-t border-border/40" />

      <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
        {items.map((item) => (
          <div
            key={item.id}
            role="button"
            tabIndex={0}
            onClick={() => onSelect(viewForItem(item))}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(viewForItem(item));
              }
            }}
            className={`group flex w-full cursor-pointer items-center gap-2 rounded-lg corner-squircle px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-accent ${
              activeId === item.id ? "bg-accent" : ""
            }`}
          >
            <span className="flex-1 truncate">{item.title}</span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleDelete(item);
              }}
              className="rounded p-0.5 opacity-0 group-hover:opacity-100 hover:bg-muted transition-opacity"
              title="Delete"
            >
              <HugeiconsIcon
                icon={Delete02Icon}
                className="size-3 text-muted-foreground"
              />
            </button>
          </div>
        ))}
        {items.length === 0 && (
          <p className="px-2 py-4 text-center text-xs text-muted-foreground">
            No threads yet
          </p>
        )}
      </div>
    </>
  );
}
