// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandList,
} from "@/components/ui/command";
import { useShortcut } from "@/features/settings/hooks/use-shortcut";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Message01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Command as CommandPrimitive } from "cmdk";
import { useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  chatSearchIndexHasRows,
  useChatSearchIndex,
} from "../hooks/use-chat-search-index";
import { useChatSearchStore } from "../stores/chat-search-store";
import { isCompactChatSearchList } from "../utils/chat-search-list-height";

// Rows mounted while the dialog animates open; the rest follow once it settles, so a long
// history never lays out hundreds of rows mid-transition.
const INITIAL_ROW_COUNT = 24;
const FULL_ROW_REVEAL_MS = 220;

// Lowercased whitespace tokens of the query (haystacks are lowercased in the index).
function queryTokens(search: string): string[] {
  return search.trim().toLowerCase().split(/\s+/).filter(Boolean);
}

function haystackMatches(haystack: string, tokens: string[]): boolean {
  return tokens.every((token) => haystack.includes(token));
}

// We filter rows here (cmdk runs with shouldFilter=false) to control the two-tier behavior and
// avoid cmdk's fuzzy scorer keeping non-matches visible (#5572): every whitespace token must
// be a substring. User messages are searched first; expand to the full conversation only when
// user text alone matches nothing anywhere.
export function selectVisibleChats<
  T extends { userSearchText: string; searchText: string },
>(items: T[], search: string): T[] {
  const tokens = queryTokens(search);
  if (tokens.length === 0) return items;
  const userHits = items.filter((it) =>
    haystackMatches(it.userSearchText, tokens),
  );
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
  // Filtering scans every conversation's text, so keep it off the keystroke path.
  const deferredQuery = useDeferredValue(query);
  // An empty query needs no scan, and the deferred value must not hold a previous filter over a reopened dialog.
  const activeQuery = query === "" ? "" : deferredQuery;
  const [rowLimit, setRowLimit] = useState(INITIAL_ROW_COUNT);
  // Centered, so the height is decided per open and only ever relaxed compact -> fixed.
  const [compactList, setCompactList] = useState(() =>
    isCompactChatSearchList(true, chatSearchIndexHasRows()),
  );

  // Reset in the opening render, not an effect: Radix mounts the portal as this render commits, so
  // an effect would trim rows only after the previous set was in the DOM. Resetting on close
  // instead would tear rows down inside the exit animation.
  const [wasOpen, setWasOpen] = useState(isOpen);
  if (isOpen !== wasOpen) {
    setWasOpen(isOpen);
    if (isOpen) {
      setQuery("");
      setRowLimit(INITIAL_ROW_COUNT);
      setCompactList(isCompactChatSearchList(true, chatSearchIndexHasRows()));
    }
  } else if (
    compactList !== isCompactChatSearchList(compactList, items.length > 0)
  ) {
    // Backstop for an open with no hint at all: the fixed height is taken when the first build lands.
    setCompactList(false);
  }

  const visibleItems = useMemo(
    () => selectVisibleChats(items, activeQuery),
    [items, activeQuery],
  );

  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(
      () => setRowLimit(Number.POSITIVE_INFINITY),
      FULL_ROW_REVEAL_MS,
    );
    return () => clearTimeout(timer);
  }, [isOpen]);

  // skipInTextFields keeps the composer's own Cmd-K and any browser find intact while the user is
  // typing, as the hand-rolled handler did.
  useShortcut("searchChats", () => useChatSearchStore.getState().open(), {
    skipInTextFields: true,
  });

  return (
    <CommandDialog
      open={isOpen}
      onOpenChange={setOpen}
      className="chat-search-surface rounded-3xl! max-sm:rounded-none! top-1/2 -translate-y-1/2 w-[635px] max-w-[calc(100%-2rem)] gap-0 p-0 ring-0 duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] sm:max-w-[635px]"
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
            // Controlled: reopening inside the exit animation reuses the mounted tree, so cmdk would keep
            // the previous text while the filter state is clear.
            value={query}
            onValueChange={setQuery}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          <button
            type="button"
            onClick={close}
            className="flex size-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Close"
          >
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={2}
              className="size-4"
            />
          </button>
        </div>
        <CommandList
          className={cn(
            "cmd-native-scrollbar hover-scrollbar p-1",
            compactList ? "max-h-[420px]" : "h-[420px] max-h-[60dvh]",
          )}
        >
          <CommandEmpty className="py-6 text-center text-xs text-muted-foreground">
            {loading
              ? "Loading…"
              : items.length === 0
                ? "No chats yet."
                : "No chats match."}
          </CommandEmpty>
          <CommandGroup className="p-0">
            {visibleItems.slice(0, rowLimit).map((item) => (
              <CommandPrimitive.Item
                key={item.id}
                value={item.id}
                onSelect={() => {
                  // The list can trail the input by a render, so a row is activatable only if the live query still
                  // matches it. Scanned on activation, not per key.
                  if (
                    query !== activeQuery &&
                    !selectVisibleChats(items, query).some(
                      (live) => live.id === item.id,
                    )
                  ) {
                    return;
                  }
                  navigate({
                    to: "/chat",
                    search:
                      item.type === "single"
                        ? {
                            thread: item.id,
                            ...(item.projectId
                              ? { project: item.projectId }
                              : {}),
                          }
                        : {
                            compare: item.id,
                            ...(item.projectId
                              ? { project: item.projectId }
                              : {}),
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
                <span className="min-w-0 flex-1 truncate text-ui-13 font-medium">
                  {item.title || "Untitled chat"}
                </span>
                <span className="shrink-0 text-ui-11 text-muted-foreground">
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
