// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ActionBarMorePrimitive, useAuiState } from "@assistant-ui/react";
import { HelpCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useLocale, useT } from "@/i18n";
import { formatMessageDate } from "@/lib/format-message-date";
import type { FC } from "react";

/** When the message was written, heading the More menu, with the details button beside it.
 *  The menu renders only while open, so no timer. */
export const MessageMenuTime: FC<{ onShowDetails: () => void }> = ({
  onShowDetails,
}) => {
  const t = useT();
  const locale = useLocale();
  const createdAt = useAuiState(({ message }) => message.createdAt?.getTime());
  const date =
    createdAt !== undefined && Number.isFinite(createdAt)
      ? new Date(createdAt)
      : null;

  return (
    <div className="flex items-center justify-between gap-3">
      {date ? (
        <time
          dateTime={date.toISOString()}
          title={date.toLocaleString(locale, { dateStyle: "full", timeStyle: "short" })}
          className="block select-none px-3 py-2 text-sm text-muted-foreground tabular-nums"
        >
          {formatMessageDate(date.getTime(), Date.now(), locale, {
            today: (time) => t("common.todayAt", { time }),
            yesterday: (time) => t("common.yesterdayAt", { time }),
          })}
        </time>
      ) : (
        <span />
      )}
      {/* Shown while the menu is hovered, or when reached by keyboard. */}
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <ActionBarMorePrimitive.Item
            onSelect={onShowDetails}
            aria-label="See response details"
            className="flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 outline-none transition-opacity hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:opacity-100 group-hover/more-menu:opacity-100"
          >
            <HugeiconsIcon icon={HelpCircleIcon} strokeWidth={1.75} className="size-icon" />
          </ActionBarMorePrimitive.Item>
        </TooltipTrigger>
        {/* Above, so it never covers the time beside it. */}
        <TooltipContent side="top" className="tooltip-compact">
          See response details
        </TooltipContent>
      </Tooltip>
    </div>
  );
};
