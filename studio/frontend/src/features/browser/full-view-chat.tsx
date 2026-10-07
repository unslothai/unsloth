// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  ArrowExpand01Icon,
  ArrowLeft02Icon,
  ArrowRight02Icon,
  BubbleChatIcon,
  MinusSignIcon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { useBrowserStore } from "./store";

const WASH =
  "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]";

export function FullViewChatBar({ title }: { title: string | undefined }) {
  const t = useT();
  const dock = useBrowserStore((state) => state.chatDock);
  const { setChatDock, closePanel, splitWithChatOn } =
    useBrowserStore.getState();
  const [menuOpen, setMenuOpen] = useState(false);
  const barRef = useRef<HTMLDivElement | null>(null);
  const expanded = dock === "expanded";

  useEffect(() => {
    if (!expanded) return;
    const viewport = barRef.current
      ?.closest(".chat-full-view-dock")
      ?.querySelector<HTMLElement>(".aui-thread-viewport");
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [expanded]);

  const toggleLabel = t(
    expanded
      ? "browser.fullView.hideConversation"
      : "browser.fullView.showConversation",
  );
  return (
    <div
      ref={barRef}
      className={cn(
        "grid shrink-0 transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none",
        expanded || menuOpen
          ? "grid-rows-[1fr]"
          : "grid-rows-[0fr] group-hover/dock:grid-rows-[1fr] has-[:focus-visible]:grid-rows-[1fr]",
      )}
    >
      <div className="min-h-0 overflow-hidden">
        <div className="flex h-[calc(52px*var(--ui-space-scale,1))] items-center gap-1 border-b border-border/60 px-2.5">
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                aria-label={t("browser.fullView.minimize")}
                onClick={() => setChatDock("minimized")}
                className={cn(
                  "flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  WASH,
                )}
              >
                <HugeiconsIcon
                  icon={MinusSignIcon}
                  strokeWidth={1.75}
                  className="size-4.5"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent side="top" className="tooltip-compact">
              {t("browser.fullView.minimize")}
            </TooltipContent>
          </Tooltip>
          <button
            type="button"
            aria-label={toggleLabel}
            aria-expanded={expanded}
            title={toggleLabel}
            onClick={() => setChatDock(expanded ? "composer" : "expanded")}
            className="flex h-8 min-w-0 cursor-pointer items-center gap-1.5 rounded-full px-2.5 text-ui-14 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <span className="min-w-0 truncate">
              {title || t("browser.fullView.newChat")}
            </span>
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className={cn(
                "size-4 shrink-0 transition-transform duration-200",
                expanded && "rotate-180",
              )}
            />
          </button>
          <span className="min-w-0 flex-1" />
          <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
            <DropdownMenuTrigger asChild={true}>
              <button
                type="button"
                aria-label={t("browser.fullView.options")}
                className={cn(
                  "flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  WASH,
                )}
              >
                <HugeiconsIcon
                  icon={MoreHorizontalIcon}
                  strokeWidth={1.75}
                  className="size-4.5"
                />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              side="top"
              align="end"
              sideOffset={8}
              className="browser-menu min-w-60 rounded-[20px] p-1.5"
            >
              <DropdownMenuItem onSelect={closePanel}>
                <HugeiconsIcon
                  icon={ArrowExpand01Icon}
                  strokeWidth={1.75}
                  className="size-4.5"
                />
                {t("browser.fullView.openChat")}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => splitWithChatOn("left")}>
                <HugeiconsIcon
                  icon={ArrowLeft02Icon}
                  strokeWidth={1.75}
                  className="size-4.5"
                />
                {t("browser.fullView.moveLeft")}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => splitWithChatOn("right")}>
                <HugeiconsIcon
                  icon={ArrowRight02Icon}
                  strokeWidth={1.75}
                  className="size-4.5"
                />
                {t("browser.fullView.moveRight")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  );
}

export function FullViewChatButton() {
  const t = useT();
  const label = t("browser.fullView.showChat");
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={() => useBrowserStore.getState().setChatDock("composer")}
          className="chat-full-view-dock-button flex size-12 cursor-pointer items-center justify-center rounded-full text-foreground transition-transform hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none"
        >
          <HugeiconsIcon
            icon={BubbleChatIcon}
            strokeWidth={1.75}
            className="size-5.5"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="left" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
