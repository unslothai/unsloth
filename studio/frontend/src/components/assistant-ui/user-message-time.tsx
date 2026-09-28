// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLocale, useT } from "@/i18n";
import { formatMessageDate } from "@/lib/format-message-date";
import { useAuiState } from "@assistant-ui/react";
import type { FC } from "react";

/** When a prompt was sent, left of its action bar. That bar mounts only while hovered, so no timer. */
export const UserMessageTime: FC = () => {
  const t = useT();
  const locale = useLocale();
  const createdAt = useAuiState(({ message }) => message.createdAt?.getTime());
  if (createdAt === undefined || !Number.isFinite(createdAt)) {
    return null;
  }
  const date = new Date(createdAt);
  return (
    <time
      dateTime={date.toISOString()}
      title={date.toLocaleString(locale, {
        dateStyle: "full",
        timeStyle: "short",
      })}
      className="aui-user-message-time mr-1 self-center whitespace-nowrap select-none text-ui-13 text-muted-foreground tabular-nums"
    >
      {formatMessageDate(createdAt, Date.now(), locale, {
        // Today reads as just the time.
        today: (time) => time,
        yesterday: (time) => t("common.yesterdayAt", { time }),
      })}
    </time>
  );
};
