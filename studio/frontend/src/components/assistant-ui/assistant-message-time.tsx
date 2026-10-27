// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatPreferencesStore } from "@/features/chat";
import { useLocale } from "@/i18n";
import { replyFinishedTimestamp } from "@/lib/message-timestamp";
import { useAuiState } from "@assistant-ui/react";
import { type FC, useState } from "react";

/** Opt-in reply time. Gated here so that, while off, a reply mounts no assistant-ui selector:
 *  every mounted one runs on each composer keystroke (composer-keystroke-subscription-budget). */
export const AssistantMessageTime: FC = () => {
  const showReplyTimes = useChatPreferencesStore((state) => state.showReplyTimes);
  return showReplyTimes ? <ReplyTime /> : null;
};

/** Absolute, not "Today": nothing re-renders an old reply past midnight. A native title, not a
 *  Tooltip: one per reply would subscribe to the modal-layer store. */
const ReplyTime: FC = () => {
  const locale = useLocale();
  const finishedAt = useAuiState(({ message }) => replyFinishedTimestamp(message));
  const [currentYear] = useState(() => new Date().getFullYear());
  if (finishedAt === undefined) {
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
