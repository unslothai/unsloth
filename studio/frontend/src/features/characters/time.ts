// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatRelativeTime, type Locale } from "@/i18n";

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
