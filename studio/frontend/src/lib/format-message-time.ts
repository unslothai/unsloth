// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Locale } from "@/i18n/messages";
import { formatRelativeTime } from "@/i18n/relative-time";

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** "just now", "3 min. ago", "5 hr. ago", "2 days ago", then a date once it is a week old. */
export function formatMessageTime(
  createdAt: number,
  now: number,
  locale: Locale,
  justNow: string,
): string {
  const elapsed = Math.max(0, now - createdAt);
  if (elapsed < MINUTE) return justNow;
  if (elapsed < HOUR)
    return formatRelativeTime(locale, -Math.floor(elapsed / MINUTE), "minute");
  if (elapsed < DAY)
    return formatRelativeTime(locale, -Math.floor(elapsed / HOUR), "hour");
  if (elapsed < 7 * DAY)
    return formatRelativeTime(locale, -Math.floor(elapsed / DAY), "day");
  const date = new Date(createdAt);
  return date.toLocaleString(locale, {
    month: "short",
    day: "numeric",
    ...(date.getFullYear() !== new Date(now).getFullYear() && {
      year: "numeric",
    }),
    hour: "numeric",
    minute: "2-digit",
  });
}
