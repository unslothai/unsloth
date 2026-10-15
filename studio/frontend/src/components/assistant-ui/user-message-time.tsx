// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLocale, useT } from "@/i18n";
import { formatMessageDate } from "@/lib/format-message-date";
import { messageTimestamp } from "@/lib/message-timestamp";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuiState } from "@assistant-ui/react";
import type { FC } from "react";

export const UserMessageTime: FC = () => {
  const t = useT();
  const locale = useLocale();
  const createdAt = useAuiState(({ message }) => messageTimestamp(message));
  if (createdAt === undefined) {
    return null;
  }
  const date = new Date(createdAt);
  const fullDate = date.toLocaleString(locale, {
    dateStyle: "full",
    timeStyle: "short",
  });
  // A button only so the keyboard reaches the tooltip; the cursor stays an arrow.
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={fullDate}
          className="aui-user-message-time-trigger mr-2 h-8 min-w-8 flex-1 cursor-default! self-center rounded-sm text-right text-ui-11p5 text-muted-foreground tabular-nums"
        >
          <time
            dateTime={date.toISOString()}
            className="aui-user-message-time block truncate select-none"
          >
            {formatMessageDate(createdAt, Date.now(), locale, {
              today: (time) => time,
              yesterday: (time) => t("common.yesterdayAt", { time }),
            })}
          </time>
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" align="end">
        {fullDate}
      </TooltipContent>
    </Tooltip>
  );
};
