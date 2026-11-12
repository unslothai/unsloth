// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import type { ReactElement } from "react";
import { openSubagentPanel, useSubagentsPolling } from "../hooks/use-subagents";
import { useSubagentsStore } from "../stores/subagents-store";
import { anySubagentRunning } from "../utils/subagents";

/**
 * Header slot of a single chat: keeps that thread's subagent list polled and, once it has any,
 * shows the button that toggles the side panel. Renders nothing for a thread without subagents.
 */
export function SubagentsHeaderControls({
  threadId,
  active,
}: {
  threadId: string | null | undefined;
  /** False while another view (compare, a browser overlay) owns the screen. */
  active: boolean;
}): ReactElement | null {
  const t = useT();
  useSubagentsPolling(threadId, active);
  const records = useSubagentsStore((state) =>
    threadId ? state.byThread[threadId] : undefined,
  );
  const open = useSubagentsStore(
    (state) => threadId != null && state.openThreadId === threadId,
  );
  if (!threadId || !records || records.length === 0) return null;
  const running = anySubagentRunning(records);
  const label = t("subagents.toolbarTooltip", { count: records.length });
  return (
    <Tooltip>
      <TooltipPrimitive.Trigger asChild={true}>
        <button
          type="button"
          onClick={() => {
            const store = useSubagentsStore.getState();
            if (open) {
              store.closePanel();
              return;
            }
            openSubagentPanel(threadId);
          }}
          className="relative flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:hover:text-white"
          aria-label={label}
          aria-pressed={open}
        >
          <HugeiconsIcon
            icon={Robot01Icon}
            className="size-icon"
            strokeWidth={1.75}
          />
          <span
            className={cn(
              "absolute -right-0.5 -top-0.5 flex min-w-[calc(14px*var(--ui-space-scale,1))] items-center justify-center rounded-full bg-primary px-1 text-ui-10 font-medium leading-[calc(14px*var(--ui-space-scale,1))] tabular-nums text-primary-foreground ring-2 ring-background",
              running && "animate-pulse motion-reduce:animate-none",
            )}
          >
            {records.length > 99 ? "99+" : records.length}
          </span>
        </button>
      </TooltipPrimitive.Trigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
