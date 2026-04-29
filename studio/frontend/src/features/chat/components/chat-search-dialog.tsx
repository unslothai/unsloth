// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandList,
} from "@/components/ui/command";
import { useTrainingRuntimeStore } from "@/features/training";
import { Cancel01Icon, Message01Icon, SearchIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Command as CommandPrimitive } from "cmdk";
import { useEffect } from "react";
import { useChatSearchIndex } from "../hooks/use-chat-search-index";
import { useChatSearchStore } from "../stores/chat-search-store";

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

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "k") return;
      if (useTrainingRuntimeStore.getState().isTrainingRunning) return;
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
      className="shadow-border corner-squircle w-[635px] max-w-[calc(100%-2rem)] gap-0 p-0 sm:max-w-[635px]"
      overlayClassName="bg-transparent"
    >
      <Command className="rounded-none p-0">
        <div className="flex items-center gap-3 border-b border-border/40 px-4 py-3">
          <HugeiconsIcon
            icon={SearchIcon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
          <CommandPrimitive.Input
            placeholder="Search chats..."
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
        <CommandList className="max-h-[420px] p-1">
          <CommandEmpty className="py-6 text-center text-xs text-muted-foreground">
            {loading
              ? "Loading…"
              : items.length === 0
                ? "No chats yet."
                : "No chats match."}
          </CommandEmpty>
          <CommandGroup className="p-0">
            {items.map((item) => (
              <CommandPrimitive.Item
                key={item.id}
                value={`${item.title} ${item.preview}`}
                onSelect={() => {
                  navigate({
                    to: "/chat",
                    search:
                      item.type === "single"
                        ? { thread: item.id }
                        : { compare: item.id },
                  });
                  close();
                }}
                className="relative flex cursor-default select-none items-center gap-3 rounded-lg px-3 py-2.5 text-sm outline-hidden data-selected:bg-muted data-selected:text-foreground"
              >
                <HugeiconsIcon
                  icon={Message01Icon}
                  strokeWidth={2}
                  className="size-4 shrink-0 text-muted-foreground"
                />
                <span className="min-w-0 flex-1 truncate text-[13px] font-medium">
                  {item.title || "Untitled chat"}
                </span>
                <span className="shrink-0 text-[11px] text-muted-foreground">
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
