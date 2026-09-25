// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAuiState } from "@assistant-ui/react";
import { useLocale, useT } from "@/i18n";
import { formatMessageTime } from "@/lib/format-message-time";
import { cn } from "@/lib/utils";
import { type FC, useSyncExternalStore } from "react";

const TICK_MS = 30_000;

// One shared clock for every visible timestamp, running only while one is mounted.
const listeners = new Set<() => void>();
let timer: ReturnType<typeof setInterval> | undefined;
function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  timer ??= setInterval(() => listeners.forEach((l) => l()), TICK_MS);
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && timer !== undefined) {
      clearInterval(timer);
      timer = undefined;
    }
  };
}
const tick = () => Math.floor(Date.now() / TICK_MS);

/** When the message was written, shown while the message is hovered or focused. */
export const MessageRelativeTime: FC<{ className?: string }> = ({
  className,
}) => {
  const t = useT();
  const locale = useLocale();
  const createdAt = useAuiState(({ message }) => message.createdAt?.getTime());
  useSyncExternalStore(subscribe, tick);

  if (createdAt === undefined || !Number.isFinite(createdAt)) return null;
  const date = new Date(createdAt);

  return (
    <time
      dateTime={date.toISOString()}
      title={date.toLocaleString(locale, {
        dateStyle: "full",
        timeStyle: "short",
      })}
      className={cn(
        "whitespace-nowrap px-1 text-ui-13 text-chat-icon-fg tabular-nums opacity-0 transition-opacity group-focus-within/assistant-message:opacity-100 group-hover/assistant-message:opacity-100",
        className,
      )}
    >
      {formatMessageTime(createdAt, Date.now(), locale, t("common.justNow"))}
    </time>
  );
};
