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
  // A button only so the keyboard reaches the tooltip; the spacer keeps it over the time alone.
  return (
    <div className="mr-2 flex min-w-8 flex-1 justify-end self-center">
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label={fullDate}
            className="aui-user-message-time-trigger h-8 min-w-0 max-w-full cursor-default! rounded-full text-right text-ui-11p5 text-muted-foreground tabular-nums"
          >
            <time
              dateTime={date.toISOString()}
              className="aui-user-message-time block truncate select-none"
            >
              {formatMessageDate(createdAt, Date.now(), locale, {
                // Today reads as just the time.
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
    </div>
  );
};
