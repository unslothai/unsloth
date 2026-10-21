// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatPreferencesStore } from "@/features/chat";
import { useLocale } from "@/i18n";
import { replyFinishedTimestamp } from "@/lib/message-timestamp";
import { useAuiState } from "@assistant-ui/react";
import { type FC, useState } from "react";

/** Opt-in reply time, always shown. Absolute, not "Today": a thread left open overnight would
 *  keep calling yesterday's replies today, as nothing re-renders them. A native title, not a
 *  Tooltip: one per reply on a long thread would each subscribe to the modal-layer store. */
export const AssistantMessageTime: FC = () => {
  const locale = useLocale();
  const showReplyTimes = useChatPreferencesStore((state) => state.showReplyTimes);
  const finishedAt = useAuiState(({ message }) => replyFinishedTimestamp(message));
  const [currentYear] = useState(() => new Date().getFullYear());
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
      {date.toLocaleString(locale, {
        month: "short",
        day: "numeric",
        ...(date.getFullYear() !== currentYear && { year: "numeric" }),
        hour: "numeric",
        minute: "2-digit",
      })}
    </time>
  );
};
