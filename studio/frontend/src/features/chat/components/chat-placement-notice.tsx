// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CpuIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { describePlacement } from "../lib/partial-offload";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/** Keeps what the load toast warned about on screen while the model stays loaded: layers, or the whole
 *  model, on the CPU although the load did not ask for that. Nothing for a full or a chosen placement. */
export function ChatPlacementNotice({ loaded }: { loaded: boolean }) {
  const t = useT();
  const placement = useChatRuntimeStore((s) => s.loadedPlacement);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  if (!loaded || modelLoading || !placement?.shortfall) return null;
  const label = t(
    placement.shortfall === "partial"
      ? "generationStats.placement.partlyOnCpu"
      : "generationStats.placement.onCpu",
  );
  const { layers } = describePlacement(placement, t);
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          data-chat-model-notice=""
          data-chat-placement-notice={placement.shortfall}
          aria-label={layers ? `${label}: ${layers}` : label}
          className="flex h-[var(--studio-chat-control-height,34px)] min-w-[calc(30px*var(--ui-space-scale,1))] shrink-0 cursor-default items-center rounded-[10px] px-2 text-ui-13 text-amber-700 transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:text-amber-400 max-sm:min-w-[calc(22px*var(--ui-space-scale,1))] max-sm:px-1"
        >
          <HugeiconsIcon
            icon={CpuIcon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0"
            aria-hidden={true}
          />
          <span className="whitespace-nowrap before:inline-block before:w-1.5 before:content-[''] max-sm:hidden">
            {label}
          </span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="max-w-72">
        {layers ? <span className="block">{layers}</span> : null}
        <span className="mt-1 block text-ui-10">
          {t("generationStats.placement.shortfallHint")}
        </span>
      </TooltipContent>
    </Tooltip>
  );
}
