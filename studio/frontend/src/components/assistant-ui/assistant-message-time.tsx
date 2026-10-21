// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatPreferencesStore } from "@/features/chat";
import { useLocale, useT } from "@/i18n";
import { formatMessageDate } from "@/lib/format-message-date";
import { replyFinishedTimestamp } from "@/lib/message-timestamp";
import { useAuiState } from "@assistant-ui/react";
import { type FC, useState } from "react";

/** Opt-in reply time, always shown. A native title, not a Tooltip: one per reply on a long
 *  thread would each subscribe to the modal-layer store. No clock subscription. */
export const AssistantMessageTime: FC = () => {
  const t = useT();
  const locale = useLocale();
  const showReplyTimes = useChatPreferencesStore((state) => state.showReplyTimes);
  const finishedAt = useAuiState(({ message }) => replyFinishedTimestamp(message));
  const [now] = useState(Date.now);
  if (!showReplyTimes || finishedAt === undefined) {
    return null;
  }
  const date = new Date(finishedAt);
  return (
    <time
      dateTime={date.toISOString()}
      title={date.toLocaleString(locale, { dateStyle: "full", timeStyle: "medium" })}
      className="aui-assistant-message-time ml-[var(--icon-btn-inset)] mr-2 shrink-0 select-none text-ui-11p5 text-muted-foreground tabular-nums"
    >
      {formatMessageDate(finishedAt, now, locale, {
        today: (time) => time,
        yesterday: (time) => t("common.yesterdayAt", { time }),
      })}
    </time>
  );
};
