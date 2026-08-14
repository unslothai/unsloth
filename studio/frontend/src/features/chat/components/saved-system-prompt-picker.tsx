// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { Bookmark02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { type ReactNode, useCallback, useState } from "react";
import { type PromptEntry, listPromptEntries } from "../api/prompts-api";

export function SavedSystemPromptPicker({
  currentSystemPrompt,
  onSelect,
}: {
  currentSystemPrompt: string;
  onSelect: (text: string) => void;
}) {
  const [entries, setEntries] = useState<PromptEntry[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  const loadEntries = useCallback(async (open: boolean) => {
    if (!open) {
      return;
    }
    try {
      setEntries(await listPromptEntries());
    } catch {
      toast.error("Could not load saved prompts");
    } finally {
      setIsLoaded(true);
    }
  }, []);

  let menuItems: ReactNode;
  if (!isLoaded) {
    menuItems = (
      <DropdownMenuItem disabled={true} className="text-ui-13">
        Loading…
      </DropdownMenuItem>
    );
  } else if (entries.length === 0) {
    menuItems = (
      <DropdownMenuItem disabled={true} className="text-ui-13">
        No saved prompts yet
      </DropdownMenuItem>
    );
  } else {
    menuItems = entries.map((entry) => (
      <DropdownMenuItem
        key={entry.id}
        onSelect={() => onSelect(entry.text)}
        className="flex min-h-9 items-center gap-2 px-3 py-0 text-ui-13 font-medium leading-[1.4] tracking-nav"
      >
        <span className="min-w-0 flex-1 truncate">{entry.name}</span>
        {entry.text === currentSystemPrompt ? (
          <HugeiconsIcon
            icon={Tick02Icon}
            strokeWidth={2}
            className="size-3.5 shrink-0"
          />
        ) : null}
      </DropdownMenuItem>
    ));
  }

  return (
    <DropdownMenu
      onOpenChange={(open) => {
        loadEntries(open).catch(() => {
          toast.error("Could not load saved prompts");
        });
      }}
    >
      <Tooltip>
        <TooltipPrimitive.Trigger asChild={true}>
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              className="nav-icon-btn text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
              aria-label="Load saved prompt as system prompt"
            >
              <HugeiconsIcon
                icon={Bookmark02Icon}
                strokeWidth={1.75}
                className="size-3"
              />
            </button>
          </DropdownMenuTrigger>
        </TooltipPrimitive.Trigger>
        <TooltipContent side="top" sideOffset={6} className="tooltip-compact">
          Saved prompts
        </TooltipContent>
      </Tooltip>
      <DropdownMenuContent
        align="end"
        sideOffset={6}
        className="menu-soft-surface max-h-64 w-56 overflow-y-auto rounded-lg border-0 p-1.5 ring-0"
      >
        {menuItems}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
