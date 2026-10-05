// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Locale } from "@/i18n/messages";

/** "Today, 4:45 AM", "Yesterday, 9:12 PM", then the date, with the year only when it differs. */
export function formatMessageDate(
  createdAt: number,
  now: number,
  locale: Locale,
  labels: {
    today: (time: string) => string;
    yesterday: (time: string) => string;
  },
): string {
  const date = new Date(createdAt);
  const today = new Date(now);
  const time = date.toLocaleTimeString(locale, {
    hour: "numeric",
    minute: "2-digit",
  });
  const sameDay = (a: Date, b: Date) => a.toDateString() === b.toDateString();
  if (sameDay(date, today)) return labels.today(time);
  const yesterday = new Date(today.getFullYear(), today.getMonth(), today.getDate() - 1);
  if (sameDay(date, yesterday)) return labels.yesterday(time);
  return date.toLocaleString(locale, {
    month: "short",
    day: "numeric",
    ...(date.getFullYear() !== today.getFullYear() && { year: "numeric" }),
    hour: "numeric",
    minute: "2-digit",
  });
}
