// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAuiState } from "@assistant-ui/react";
import { useLocale, useT } from "@/i18n";
import { formatMessageDate } from "@/lib/format-message-date";
import type { FC } from "react";

/** When the message was written. Heads the More menu, which renders only while open. */
export const MessageMenuTime: FC = () => {
  const t = useT();
  const locale = useLocale();
  const createdAt = useAuiState(({ message }) => message.createdAt?.getTime());

  if (createdAt === undefined || !Number.isFinite(createdAt)) return null;
  const date = new Date(createdAt);

  return (
    <time
      dateTime={date.toISOString()}
      title={date.toLocaleString(locale, { dateStyle: "full", timeStyle: "short" })}
      className="block select-none px-3 pt-1.5 pb-2 text-sm text-muted-foreground tabular-nums"
    >
      {formatMessageDate(createdAt, Date.now(), locale, {
        today: (time) => t("common.todayAt", { time }),
        yesterday: (time) => t("common.yesterdayAt", { time }),
      })}
    </time>
  );
};
