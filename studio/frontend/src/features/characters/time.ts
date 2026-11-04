// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatRelativeTime, type Locale } from "@/i18n";
import { withTimeFormat } from "@/lib/time-format";

/** "5 minutes ago", "yesterday", then a date for anything older than a week. */
export function relativeTime(locale: Locale, timestamp: number, now = Date.now()): string {
  const seconds = Math.round((timestamp - now) / 1000);
  const abs = Math.abs(seconds);
  if (abs < 60) return formatRelativeTime(locale, seconds, "second");
  if (abs < 3600) return formatRelativeTime(locale, Math.round(seconds / 60), "minute");
  if (abs < 86400) return formatRelativeTime(locale, Math.round(seconds / 3600), "hour");
  if (abs < 7 * 86400) return formatRelativeTime(locale, Math.round(seconds / 86400), "day");
  return new Date(timestamp).toLocaleDateString(locale, { month: "short", day: "numeric", year: "numeric" });
}

const DAY_MS = 86_400_000;

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/** A chat message's time: "2:58 PM" today, "Yesterday 3:48 PM", "Sunday 3:48 PM" this week, then "Sep 30, 4:46 PM". */
export function messageTime(locale: Locale, timestamp: number, now = Date.now()): string {
  const then = new Date(timestamp);
  const today = new Date(now);
  // Rounded: DST days are 23 or 25 hours long.
  const days = Math.round((startOfDay(today) - startOfDay(then)) / DAY_MS);
  const clock = { hour: "numeric", minute: "2-digit" } as const;
  if (days === 0) return then.toLocaleTimeString(locale, withTimeFormat(clock, locale));
  if (days === 1) {
    const word = new Intl.RelativeTimeFormat(locale, { numeric: "auto" }).format(-1, "day");
    const time = then.toLocaleTimeString(locale, withTimeFormat(clock, locale));
    return `${word.charAt(0).toLocaleUpperCase(locale)}${word.slice(1)} ${time}`;
  }
  if (days > 1 && days < 7) return then.toLocaleString(locale, withTimeFormat({ weekday: "long", ...clock }, locale));
  return then.toLocaleString(
    locale,
    withTimeFormat(
      { month: "short", day: "numeric", year: then.getFullYear() === today.getFullYear() ? undefined : "numeric", ...clock },
      locale,
    ),
  );
}
