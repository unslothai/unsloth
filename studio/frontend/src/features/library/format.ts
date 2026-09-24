// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const DAY_MS = 86_400_000;

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/**
 * A clock time today, then "Yesterday", then the weekday for the past week, then a short date, in
 * `locale` (the app's language; the browser's when left out). A time ahead of `now`, from another
 * machine's clock, shows its date.
 */
export function formatCardTime(ts: number, locale?: string, now: number = Date.now()): string {
  if (!Number.isFinite(ts)) return "";
  const then = new Date(ts);
  const today = new Date(now);
  // Rounded: a day with a daylight-saving change is 23 or 25 hours long.
  const days = Math.round((startOfDay(today) - startOfDay(then)) / DAY_MS);
  if (days === 0) {
    return then.toLocaleTimeString(locale, { hour: "numeric", minute: "2-digit" });
  }
  if (days === 1) {
    const yesterday = new Intl.RelativeTimeFormat(locale, { numeric: "auto" }).format(-1, "day");
    return yesterday.charAt(0).toLocaleUpperCase(locale) + yesterday.slice(1);
  }
  if (days > 1 && days < 7) return then.toLocaleDateString(locale, { weekday: "long" });
  return then.toLocaleDateString(locale, {
    month: "short",
    day: "numeric",
    year: then.getFullYear() === today.getFullYear() ? undefined : "numeric",
  });
}

/** How long ago, down to the minute, for Suggested's Last activity. */
export function formatRelativeTime(ts: number): string {
  const minutes = Math.floor((Date.now() - ts) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return formatCardTime(ts);
}

export function formatSize(bytes: number | null): string | null {
  if (bytes === null) return null;
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (const next of units.slice(1)) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ${unit}`;
}

export function pluralize(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}
