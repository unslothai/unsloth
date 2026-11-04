// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { TheaterIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Portrait } from "../components/portrait";
import { useCharactersStore } from "../store";
import { useThreadScene } from "./use-thread-scene";

/** The chat's characters at the top of a character chat; opens their panel. */
export function CastChip({ threadId, onOpen }: { threadId: string | null | undefined; onOpen: () => void }) {
  const t = useT();
  const thread = useThreadScene(threadId);
  if (!thread) return null;
  const shown = thread.cast.slice(0, 3);
  const label =
    thread.cast.length <= 2
      ? thread.cast.map((c) => c.name).join(" & ")
      : `${thread.cast[0].name}, ${thread.cast[1].name} +${thread.cast.length - 2}`;
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={t("characters.chat.openCast")}
      title={t("characters.chat.castLabel")}
      data-testid="cast-chip"
      className="ml-1 flex h-[var(--studio-chat-control-height,34px)] min-w-0 max-w-[42vw] cursor-pointer items-center gap-2 rounded-full py-1 pl-1 pr-3 text-ui-14 font-medium text-foreground transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <span className="flex shrink-0 -space-x-2">
        {shown.map((character) => (
          <Portrait
            key={character.id}
            id={character.id}
            name={character.name}
            hasImage={character.hasAvatar}
            rev={character.avatarRev}
            className="size-6.5 rounded-full text-[0.62rem] ring-2 ring-background"
          />
        ))}
      </span>
      <span className="truncate">{label}</span>
    </button>
  );
}

/** Toggles the chat's stage view, remembered per chat. */
export function StageToggle({ threadId }: { threadId: string | null | undefined }) {
  const t = useT();
  const thread = useThreadScene(threadId);
  if (!thread || !threadId) return null;
  const on = Boolean(thread.scene.settings.stage);
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-pressed={on}
          aria-label={on ? t("characters.chat.stageOff") : t("characters.chat.stageOn")}
          data-testid="stage-toggle"
          onClick={() => {
            void useCharactersStore
              .getState()
              .patchScene(threadId, { settings: { ...thread.scene.settings, stage: !on } })
              .catch((error) => toast.error(error instanceof Error ? error.message : String(error)));
          }}
          className={cn(
            "flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:text-white",
            on && "bg-nav-surface-hover text-foreground",
          )}
        >
          <HugeiconsIcon icon={TheaterIcon} strokeWidth={1.75} className="size-icon" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {t("characters.chat.stage")}
      </TooltipContent>
    </Tooltip>
  );
}
