// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useLocale, useT, type TranslationKey } from "@/i18n";
import { cn } from "@/lib/utils";
import { BubbleChatIcon, ImageCompositionIcon, LayoutLeftIcon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { Portrait } from "../components/portrait";
import { chatViewOf, setChatView } from "../chat-view";
import type { ChatView } from "../types";
import { useThreadScene } from "./use-thread-scene";

/** The chat's characters beside its title: portraits only, since the title already names them. Toggles their panel. */
export function CastChip({
  threadId,
  expanded,
  onToggle,
}: {
  threadId: string | null | undefined;
  expanded: boolean;
  onToggle: () => void;
}) {
  const t = useT();
  const locale = useLocale();
  const thread = useThreadScene(threadId);
  if (!thread) return null;
  const shown = thread.cast.slice(0, 3);
  const names = new Intl.ListFormat(locale, { style: "long", type: "conjunction" }).format(
    thread.cast.map((c) => c.name),
  );
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={expanded}
          data-testid="cast-chip"
          className="flex h-[var(--studio-chat-control-height,34px)] shrink-0 cursor-pointer items-center rounded-full px-1 transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {/* The portraits are decorative, so the names name the button. */}
          <span className="sr-only">{names}</span>
          <span className="flex -space-x-2" aria-hidden={true}>
            {shown.map((character) => (
              <Portrait
                key={character.id}
                subject={character}
                className="size-6.5 rounded-full text-[0.62rem] ring-2 ring-background"
              />
            ))}
          </span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact flex max-w-72 flex-col gap-0.5">
        <span className="font-medium">{names}</span>
        <span className="opacity-80">{t("characters.chat.openCast")}</span>
      </TooltipContent>
    </Tooltip>
  );
}

const VIEWS: { value: ChatView; label: TranslationKey; hint: TranslationKey | null; icon: IconSvgElement }[] = [
  { value: "chat", label: "characters.chat.viewChat", hint: null, icon: BubbleChatIcon },
  { value: "stage", label: "characters.chat.viewStage", hint: "characters.chat.viewStageHint", icon: LayoutLeftIcon },
  { value: "novel", label: "characters.chat.viewNovel", hint: "characters.chat.viewNovelHint", icon: ImageCompositionIcon },
];

/** Picks how a character chat is laid out, remembered per chat. */
export function ChatViewMenu({ threadId }: { threadId: string | null | undefined }) {
  const t = useT();
  const thread = useThreadScene(threadId);
  if (!thread || !threadId) return null;
  const view = chatViewOf(thread.scene.settings);
  const current = VIEWS.find((option) => option.value === view) ?? VIEWS[0];
  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              aria-label={t("characters.chat.view")}
              data-testid="chat-view-menu"
              className={cn(
                "flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:text-white",
                view !== "chat" && "bg-nav-surface-hover text-foreground",
              )}
            >
              <HugeiconsIcon icon={current.icon} strokeWidth={1.75} className="size-icon" />
            </button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
          {t("characters.chat.view")}
        </TooltipContent>
      </Tooltip>
      {/* Focus would return to the button and reopen its tooltip after every choice. */}
      <DropdownMenuContent align="end" className="w-72" onCloseAutoFocus={(event) => event.preventDefault()}>
        <DropdownMenuRadioGroup value={view} onValueChange={(next) => setChatView(threadId, next as ChatView)}>
          {VIEWS.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value} className="items-start gap-2.5">
              <HugeiconsIcon icon={option.icon} strokeWidth={1.75} className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
              <span className="flex flex-col gap-0.5">
                <span>{t(option.label)}</span>
                {option.hint ? <span className="text-ui-12 text-muted-foreground">{t(option.hint)}</span> : null}
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
