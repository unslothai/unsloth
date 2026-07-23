// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandList,
} from "@/components/ui/command";
import { Cancel01Icon, Message01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Command as CommandPrimitive } from "cmdk";
import { useEffect, useMemo, useState } from "react";
import { useChatSearchIndex } from "../hooks/use-chat-search-index";
import { useChatSearchStore } from "../stores/chat-search-store";

// Lowercased whitespace tokens of the query (haystacks are lowercased in the index).
function queryTokens(search: string): string[] {
  return search.trim().toLowerCase().split(/\s+/).filter(Boolean);
}

function haystackMatches(haystack: string, tokens: string[]): boolean {
  return tokens.every((token) => haystack.includes(token));
}

// We filter rows here (cmdk runs with shouldFilter=false) so we control the
// two-tier behavior and avoid cmdk's fuzzy scorer keeping non-matches visible
// (issue #5572): every whitespace token must be a substring. User messages are
// searched first; expand to the full conversation only when user text alone
// matches nothing anywhere (user messages are short, assistant replies can be huge).
export function selectVisibleChats<
  T extends { userSearchText: string; searchText: string },
>(items: T[], search: string): T[] {
  const tokens = queryTokens(search);
  if (tokens.length === 0) return items;
  const userHits = items.filter((it) => haystackMatches(it.userSearchText, tokens));
  if (userHits.length > 0) return userHits;
  return items.filter((it) => haystackMatches(it.searchText, tokens));
}

function formatRelative(createdAt: number): string {
  const diff = Date.now() - createdAt;
  const day = 86_400_000;
  if (diff < day) return "Today";
  if (diff < 7 * day) return "Past week";
  if (diff < 30 * day) return "Past month";
  return "Older";
}

export function ChatSearchDialog() {
  const isOpen = useChatSearchStore((s) => s.isOpen);
  const setOpen = useChatSearchStore((s) => s.setOpen);
  const close = useChatSearchStore((s) => s.close);
  const navigate = useNavigate();
  const { items, loading } = useChatSearchIndex(isOpen);
  const [query, setQuery] = useState("");

  const visibleItems = useMemo(
    () => selectVisibleChats(items, query),
    [items, query],
  );

  useEffect(() => {
    if (!isOpen) setQuery("");
  }, [isOpen]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "k") return;
      const el = document.activeElement as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || el?.isContentEditable) return;
      e.preventDefault();
      useChatSearchStore.getState().open();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <CommandDialog
      open={isOpen}
      onOpenChange={setOpen}
      className="chat-search-surface rounded-3xl! top-1/2 -translate-y-1/2 w-[635px] max-w-[calc(100%-32px)] gap-0 p-0 ring-0 sm:max-w-[635px]"
      overlayClassName="bg-transparent supports-backdrop-filter:backdrop-blur-none"
    >
      <Command className="rounded-3xl p-0" shouldFilter={false}>
        <div className="flex items-center gap-3 border-b border-border/40 px-4 py-3">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
          <CommandPrimitive.Input
            placeholder="Search chats..."
            onValueChange={setQuery}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          <button
            type="button"
            onClick={close}
            className="flex size-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Close"
          >
            <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} className="size-4" />
          </button>
        </div>
        <CommandList className="cmd-native-scrollbar hover-scrollbar max-h-[420px] p-1">
          <CommandEmpty className="py-6 text-center text-xs text-muted-foreground">
            {loading
              ? "Loading…"
              : items.length === 0
                ? "No chats yet."
                : "No chats match."}
          </CommandEmpty>
          <CommandGroup className="p-0">
            {visibleItems.map((item) => (
              <CommandPrimitive.Item
                key={item.id}
                value={item.id}
                onSelect={() => {
                  navigate({
                    to: "/chat",
                    search:
                      item.type === "single"
                        ? {
                            thread: item.id,
                            ...(item.projectId ? { project: item.projectId } : {}),
                          }
                        : {
                            compare: item.id,
                            ...(item.projectId ? { project: item.projectId } : {}),
                          },
                  });
                  close();
                }}
                className="relative flex cursor-pointer select-none items-center gap-3 rounded-full px-3 py-2.5 text-sm outline-hidden data-selected:bg-muted data-selected:text-foreground"
              >
                <HugeiconsIcon
                  icon={Message01Icon}
                  strokeWidth={2}
                  className="size-4 shrink-0 text-muted-foreground"
                />
                <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium">
                  {item.title || "Untitled chat"}
                </span>
                <span className="shrink-0 text-[0.6875rem] text-muted-foreground">
                  {formatRelative(item.createdAt)}
                </span>
              </CommandPrimitive.Item>
            ))}
          </CommandGroup>
        </CommandList>
      </Command>
    </CommandDialog>
  );
}
