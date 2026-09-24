// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InterpolationValues, Locale, TranslationKey } from "@/i18n";
import { formatRelativeTime } from "@/i18n/relative-time";

type Translate = (key: TranslationKey, values?: InterpolationValues) => string;

const yesterdayFormatters = new Map<Locale, Intl.RelativeTimeFormat>();

/** "Yesterday" as the locale words it. */
function yesterday(locale: Locale): string {
  let formatter = yesterdayFormatters.get(locale);
  if (!formatter) {
    formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
    yesterdayFormatters.set(locale, formatter);
  }
  return formatter.format(-1, "day");
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/** Grid cards: a clock time today, a short date before that. */
export function formatCardTime(ts: number, locale: Locale): string {
  const then = new Date(ts);
  const now = new Date();
  if (then.toDateString() === now.toDateString()) {
    return then.toLocaleTimeString(locale, {
      hour: "numeric",
      minute: "2-digit",
    });
  }
  return then.toLocaleDateString(locale, {
    month: "short",
    day: "numeric",
    year: then.getFullYear() === now.getFullYear() ? undefined : "numeric",
  });
}

/** List rows: how long ago, down to the minute. A timestamp ahead of this clock shows its date. */
export function formatActivityTime(ts: number, locale: Locale, t: Translate): string {
  const elapsed = Date.now() - ts;
  if (elapsed < -60_000) return formatCardTime(ts, locale);
  const minutes = Math.floor(elapsed / 60_000);
  if (minutes < 1) return t("library.list.justNow");
  if (minutes < 60) return formatRelativeTime(locale, -minutes, "minute");
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return formatRelativeTime(locale, -hours, "hour");
  // Calendar days from here, so the day before today reads as "yesterday".
  const days = Math.round((startOfDay(new Date()) - startOfDay(new Date(ts))) / 86_400_000);
  if (days <= 1) return yesterday(locale);
  if (days < 7) return formatRelativeTime(locale, -days, "day");
  return formatCardTime(ts, locale);
}

const SIZE_UNITS = [
  "library.size.kilobytes",
  "library.size.megabytes",
  "library.size.gigabytes",
] as const satisfies readonly TranslationKey[];

export function formatSize(bytes: number | null, locale: Locale, t: Translate): string | null {
  if (bytes === null) return null;
  if (bytes < 1024) {
    return t("library.size.bytes", { value: bytes.toLocaleString(locale, { useGrouping: false }) });
  }
  let value = bytes / 1024;
  let unit: (typeof SIZE_UNITS)[number] = SIZE_UNITS[0];
  for (const next of SIZE_UNITS.slice(1)) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  const digits = value >= 10 ? 0 : 1;
  const formatted = value.toLocaleString(locale, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return t(unit, { value: formatted });
}

/** "{count} items", in the singular for one. */
export function formatItemCount(count: number, t: Translate): string {
  return t(count === 1 ? "library.itemCountOne" : "library.itemCount", { count });
}

/** A scale as the locale writes a percentage: 1.25 is "125%". */
export function formatPercent(scale: number, locale: Locale): string {
  return new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 0 }).format(
    scale,
  );
}
