// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Wall-clock helpers shared by the scheduled-task dialog and scheduled training starts. */

import { withTimeFormat } from "./time-format.ts";

const CLOCK = /^([01]\d|2[0-3]):[0-5]\d$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;

const formatters = new Map<
  string,
  Intl.DateTimeFormat | Intl.ListFormat | Intl.RelativeTimeFormat
>();

function cached<T extends Intl.DateTimeFormat | Intl.ListFormat | Intl.RelativeTimeFormat>(
  key: string,
  make: () => T,
): T {
  let format = formatters.get(key) as T | undefined;
  if (!format) {
    format = make();
    formatters.set(key, format);
  }
  return format;
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

export function isClockTime(value: string): boolean {
  return CLOCK.test(value);
}

export function isDateKey(value: string): boolean {
  return DATE.test(value);
}

export function localDateKey(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function localClock(date: Date): string {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** The wall clock of `date` as "YYYY-MM-DDTHH:MM"; these keys sort in time order. */
export function localMinuteKey(date: Date): string {
  return `${localDateKey(date)}T${localClock(date)}`;
}

/** The next quarter hour, so a fresh start time is never already in the past. */
export function nextQuarterHour(now: Date): { date: string; time: string } {
  const next = new Date(now);
  next.setSeconds(0, 0);
  next.setMinutes(Math.floor(next.getMinutes() / 15) * 15 + 15);
  return { date: localDateKey(next), time: localClock(next) };
}

export function dateFormat(
  locale: string,
  options: Intl.DateTimeFormatOptions,
): Intl.DateTimeFormat {
  return cached(
    `date|${locale}|${JSON.stringify(options)}`,
    () => new Intl.DateTimeFormat(locale, options),
  );
}

export function listFormat(
  locale: string,
  options: Intl.ListFormatOptions,
): Intl.ListFormat {
  return cached(
    `list|${locale}|${JSON.stringify(options)}`,
    () => new Intl.ListFormat(locale, options),
  );
}

export function relativeFormat(locale: string): Intl.RelativeTimeFormat {
  return cached(
    `relative|${locale}`,
    () => new Intl.RelativeTimeFormat(locale, { numeric: "auto" }),
  );
}

export function formatClock(time: string, locale: string): string {
  const [hour, minute] = time.split(":").map(Number);
  const options = withTimeFormat(
    { hour: "numeric", minute: "2-digit", timeZone: "UTC" },
    locale,
  );
  return dateFormat(locale, options).format(Date.UTC(2000, 0, 1, hour, minute));
}

/** A stored wall clock ("YYYY-MM-DDTHH:MM") in the locale's date and time. */
export function formatLocalMinute(localMinute: string, locale: string): string {
  const [year, month, day, hour, minute] = localMinute.split(/[-T:]/).map(Number);
  const options = withTimeFormat(
    { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" },
    locale,
  );
  return dateFormat(locale, options).format(
    Date.UTC(year, month - 1, day, hour, minute),
  );
}

/** Quarter-hour times plus `current` when it falls between them. */
export function timeChoices(current: string): string[] {
  const times = Array.from(
    { length: 96 },
    (_, index) => `${pad(Math.floor(index / 4))}:${pad((index % 4) * 15)}`,
  );
  return !CLOCK.test(current) || times.includes(current)
    ? times
    : [...times, current].sort();
}

/** A time zone as briefly as the locale names it ("PT", "MEZ"), else its offset at `at` ("GMT+3"). */
export function zoneLabel(timeZone: string, locale: string, at: number): string {
  const name = (style: "short" | "shortGeneric") => {
    try {
      return (
        dateFormat(locale, { timeZone, timeZoneName: style })
          .formatToParts(at)
          .find((part) => part.type === "timeZoneName")?.value ?? ""
      );
    } catch {
      return "";
    }
  };
  // The generic name holds across daylight saving, but past an abbreviation it is a phrase.
  const generic = name("shortGeneric");
  const specific = name("short");
  if (generic && !/\s/.test(generic) && (!specific || generic.length <= specific.length)) {
    return generic;
  }
  return specific || timeZone;
}

export function browserTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}
