// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatPreferencesStore } from "@/features/chat";
import { useLocale } from "@/i18n";
import { replyFinishedTimestamp } from "@/lib/message-timestamp";
import { useTimeFormat, withTimeFormat } from "@/lib/time-format";
import { useAuiState } from "@assistant-ui/react";
import { type FC, useState } from "react";

/** Gated so an off reply mounts no assistant-ui selector; each runs per composer keystroke. */
export const AssistantMessageTime: FC = () => {
  const showReplyTimes = useChatPreferencesStore((state) => state.showReplyTimes);
  return showReplyTimes ? <ReplyTime /> : null;
};

/** Absolute, not Today: no re-render past midnight. Native title, not per-reply Tooltip store. */
const ReplyTime: FC = () => {
  const locale = useLocale();
  useTimeFormat();
  const finishedAt = useAuiState(({ message }) => replyFinishedTimestamp(message));
  const [currentYear] = useState(() => new Date().getFullYear());
  if (finishedAt === undefined) {
    return null;
  }
  const date = new Date(finishedAt);
  return (
    <time
      dateTime={date.toISOString()}
      title={date.toLocaleString(
        locale,
        withTimeFormat({ dateStyle: "full", timeStyle: "medium" }, locale),
      )}
      className="aui-assistant-message-time ml-[var(--icon-btn-inset)] mr-2 shrink-0 select-none text-ui-11p5 text-muted-foreground tabular-nums"
    >
      {date.toLocaleString(
        locale,
        withTimeFormat(
          {
            month: "short",
            day: "numeric",
            ...(date.getFullYear() !== currentYear && { year: "numeric" }),
            hour: "numeric",
            minute: "2-digit",
          },
          locale,
        ),
      )}
    </time>
  );
};
