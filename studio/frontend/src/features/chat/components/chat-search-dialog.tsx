// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { useTrainingRuntimeStore } from "@/features/training";
import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { useChatSearchIndex } from "../hooks/use-chat-search-index";
import { useChatSearchStore } from "../stores/chat-search-store";

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
    <CommandDialog open={isOpen} onOpenChange={setOpen}>
      <CommandInput placeholder="Search chats…" />
      <CommandList>
        {loading ? (
          <div className="py-6 text-center text-xs text-muted-foreground">Loading…</div>
        ) : items.length === 0 ? (
          <CommandEmpty>No chats yet.</CommandEmpty>
        ) : (
          <>
            <CommandEmpty>No chats match.</CommandEmpty>
            <CommandGroup heading="Recent">
              {items.map((item) => (
                <CommandItem
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
                  className="flex flex-col items-start gap-0.5 py-2"
                >
                  <span className="text-[13px] font-medium">
                    {item.title || "Untitled chat"}
                  </span>
                  {item.preview && (
                    <span className="line-clamp-1 text-[11px] text-muted-foreground">
                      {item.preview}
                    </span>
                  )}
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </CommandDialog>
  );
}
