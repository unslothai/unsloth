// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ActionBarMorePrimitive, useAuiState } from "@assistant-ui/react";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useLocale, useT } from "@/i18n";
import { formatMessageDate } from "@/lib/format-message-date";
import { messageTimestamp } from "@/lib/message-timestamp";
import type { FC } from "react";

/** When the message was written, heading the More menu, with the details button beside it.
 *  The menu renders only while open, so no timer. */
export const MessageMenuTime: FC<{ onShowDetails: () => void }> = ({
  onShowDetails,
}) => {
  const t = useT();
  const locale = useLocale();
  const createdAt = useAuiState(({ message }) => messageTimestamp(message));
  const date =
    createdAt !== undefined && Number.isFinite(createdAt)
      ? new Date(createdAt)
      : null;

  return (
    // One item, so the time opens the details as the button beside it does.
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <ActionBarMorePrimitive.Item
          onSelect={onShowDetails}
          aria-label="See response details"
          className="group/menu-time flex w-fit max-w-full cursor-pointer items-center rounded-[11px] text-muted-foreground outline-none"
        >
          {date && (
            <time
              dateTime={date.toISOString()}
              className="block select-none py-2 pl-3 pr-1 text-sm tabular-nums transition-colors group-hover/menu-time:text-foreground group-focus/menu-time:text-foreground"
            >
              {formatMessageDate(date.getTime(), Date.now(), locale, {
                today: (time) => t("common.todayAt", { time }),
                yesterday: (time) => t("common.yesterdayAt", { time }),
              })}
            </time>
          )}
          {/* Right after the time. Shown while this row is hovered, or when reached by keyboard. */}
          <span className="flex size-7 shrink-0 items-center justify-center rounded-full opacity-0 transition-opacity group-hover/menu-time:bg-accent group-hover/menu-time:text-accent-foreground group-hover/menu-time:opacity-100 group-focus/menu-time:bg-accent group-focus/menu-time:text-accent-foreground group-focus/menu-time:opacity-100">
            <HugeiconsIcon icon={InformationCircleIcon} strokeWidth={1.75} className="size-icon" />
          </span>
        </ActionBarMorePrimitive.Item>
      </TooltipTrigger>
      {/* Above, so it never covers the time. The full date leads, where the time's title gave it. */}
      <TooltipContent side="top" className="tooltip-compact">
        {date
          ? `${date.toLocaleString(locale, { dateStyle: "full", timeStyle: "short" })} · See response details`
          : "See response details"}
      </TooltipContent>
    </Tooltip>
  );
};
