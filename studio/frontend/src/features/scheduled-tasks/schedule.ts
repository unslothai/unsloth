// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


/** Mirrors core/scheduled_tasks/schedule.py: hourly intervals use elapsed time; calendar schedules use the task's zone. */
export type TaskSchedule =
  | { kind: "once"; at: string }
  | { kind: "hourly"; hours: number }
  | { kind: "daily"; time: string }
  | { kind: "weekly"; time: string; days: number[] }
  | { kind: "monthly"; time: string; day: number };

export type RepeatKind = "once" | "hourly" | "daily" | "weekdays" | "weekly" | "monthly";

export const REPEAT_KINDS: readonly RepeatKind[] = [
  "once",
  "hourly",
  "daily",
  "weekdays",
  "weekly",
  "monthly",
];

export const ISO_WEEKDAYS = [1, 2, 3, 4, 5, 6, 7] as const;
const WORKDAYS = [1, 2, 3, 4, 5];

export interface ScheduleForm {
  repeat: RepeatKind;
  hours: number;
  time: string;
  date: string;
  days: number[];
  dayOfMonth: number;
}

const CLOCK = /^([01]\d|2[0-3]):[0-5]\d$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

export function localDateKey(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function isoWeekday(date: Date): number {
  return ((date.getDay() + 6) % 7) + 1;
}

function isWorkdays(days: readonly number[]): boolean {
  return days.length === WORKDAYS.length && WORKDAYS.every((day) => days.includes(day));
}

/** A new task: daily 09:00, other kinds pre-filled from today; a one-time task starts tomorrow. */
export function defaultScheduleForm(now: Date): ScheduleForm {
  const tomorrow = new Date(now);
  tomorrow.setDate(now.getDate() + 1);
  return {
    repeat: "daily",
    hours: 1,
    time: "09:00",
    date: localDateKey(tomorrow),
    days: [isoWeekday(now)],
    dayOfMonth: now.getDate(),
  };
}

export function formFromSchedule(schedule: TaskSchedule, now: Date): ScheduleForm {
  const form = defaultScheduleForm(now);
  switch (schedule.kind) {
    case "once": {
      const [date, time] = schedule.at.split("T");
      return { ...form, repeat: "once", date, time };
    }
    case "hourly":
      return { ...form, repeat: "hourly", hours: schedule.hours };
    case "daily":
      return { ...form, repeat: "daily", time: schedule.time };
    case "weekly":
      return isWorkdays(schedule.days)
        ? { ...form, repeat: "weekdays", time: schedule.time }
        : { ...form, repeat: "weekly", time: schedule.time, days: [...schedule.days] };
    case "monthly":
      return { ...form, repeat: "monthly", time: schedule.time, dayOfMonth: schedule.day };
  }
}

export function scheduleFromForm(form: ScheduleForm): TaskSchedule {
  switch (form.repeat) {
    case "once":
      return { kind: "once", at: `${form.date}T${form.time}` };
    case "hourly":
      return { kind: "hourly", hours: form.hours };
    case "daily":
      return { kind: "daily", time: form.time };
    case "weekdays":
      return { kind: "weekly", time: form.time, days: [...WORKDAYS] };
    case "weekly":
      return {
        kind: "weekly",
        time: form.time,
        days: [...new Set(form.days)].sort((a, b) => a - b),
      };
    case "monthly":
      return { kind: "monthly", time: form.time, day: form.dayOfMonth };
  }
}

export function scheduleFormComplete(form: ScheduleForm): boolean {
  if (form.repeat === "hourly") {
    return Number.isInteger(form.hours) && form.hours >= 1 && form.hours <= 24;
  }
  if (!CLOCK.test(form.time)) return false;
  switch (form.repeat) {
    case "once":
      return DATE.test(form.date);
    case "weekly":
      return form.days.length > 0;
    case "monthly":
      return Number.isInteger(form.dayOfMonth) && form.dayOfMonth >= 1 && form.dayOfMonth <= 31;
    default:
      return true;
  }
}

const formatters = new Map<string, Intl.DateTimeFormat | Intl.ListFormat | Intl.RelativeTimeFormat>();

function dateFormat(locale: string, options: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  const key = `date|${locale}|${JSON.stringify(options)}`;
  let format = formatters.get(key) as Intl.DateTimeFormat | undefined;
  if (!format) {
    format = new Intl.DateTimeFormat(locale, options);
    formatters.set(key, format);
  }
  return format;
}

function listFormat(locale: string, options: Intl.ListFormatOptions): Intl.ListFormat {
  const key = `list|${locale}|${JSON.stringify(options)}`;
  let format = formatters.get(key) as Intl.ListFormat | undefined;
  if (!format) {
    format = new Intl.ListFormat(locale, options);
    formatters.set(key, format);
  }
  return format;
}

function relativeFormat(locale: string): Intl.RelativeTimeFormat {
  const key = `relative|${locale}`;
  let format = formatters.get(key) as Intl.RelativeTimeFormat | undefined;
  if (!format) {
    format = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
    formatters.set(key, format);
  }
  return format;
}

export function formatClock(time: string, locale: string): string {
  const [hour, minute] = time.split(":").map(Number);
  return dateFormat(locale, { hour: "numeric", minute: "2-digit", timeZone: "UTC" }).format(
    Date.UTC(2000, 0, 1, hour, minute),
  );
}

/** Quarter-hour times plus `current` when it falls between them. */
export function timeChoices(current: string): string[] {
  const times = Array.from(
    { length: 96 },
    (_, index) => `${pad(Math.floor(index / 4))}:${pad((index % 4) * 15)}`,
  );
  return !CLOCK.test(current) || times.includes(current) ? times : [...times, current].sort();
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

export function weekdayName(day: number, locale: string, width: "short" | "long"): string {
  // 2024-01-01 was a Monday, so day d of that week is ISO weekday d.
  return dateFormat(locale, { weekday: width, timeZone: "UTC" }).format(Date.UTC(2024, 0, day));
}

/** Short weekday names via the long list style (the short style writes English as "Mon, Wed, & Fri"). */
export function formatWeekdays(days: readonly number[], locale: string): string {
  return listFormat(locale, { style: "long", type: "conjunction" }).format(
    days.map((day) => weekdayName(day, locale, "short")),
  );
}

export function sameSchedule(a: TaskSchedule, b: TaskSchedule): boolean {
  switch (a.kind) {
    case "once":
      return b.kind === "once" && a.at === b.at;
    case "hourly":
      return b.kind === "hourly" && a.hours === b.hours;
    case "daily":
      return b.kind === "daily" && a.time === b.time;
    case "weekly":
      return (
        b.kind === "weekly" &&
        a.time === b.time &&
        a.days.length === b.days.length &&
        a.days.every((day) => b.days.includes(day))
      );
    case "monthly":
      return b.kind === "monthly" && a.time === b.time && a.day === b.day;
  }
}

export type ScheduleSummary =
  | { key: "hourly"; values: Record<string, never> }
  | { key: "everyHours"; values: { hours: number } }
  | { key: "once"; values: { date: string; time: string } }
  | { key: "daily" | "weekdays"; values: { time: string } }
  | { key: "weekly"; values: { days: string; time: string } }
  | { key: "monthly"; values: { day: number; time: string } };

export function summarizeSchedule(schedule: TaskSchedule, locale: string): ScheduleSummary {
  if (schedule.kind === "hourly") {
    return schedule.hours === 1
      ? { key: "hourly", values: {} }
      : { key: "everyHours", values: { hours: schedule.hours } };
  }
  const clock = schedule.kind === "once" ? schedule.at.split("T")[1] : schedule.time;
  const time = formatClock(clock, locale);
  switch (schedule.kind) {
    case "once": {
      const [year, month, day] = schedule.at.split("T")[0].split("-").map(Number);
      const date = dateFormat(locale, { dateStyle: "medium", timeZone: "UTC" }).format(
        Date.UTC(year, month - 1, day),
      );
      return { key: "once", values: { date, time } };
    }
    case "daily":
      return { key: "daily", values: { time } };
    case "weekly":
      if (schedule.days.length === ISO_WEEKDAYS.length) return { key: "daily", values: { time } };
      if (isWorkdays(schedule.days)) return { key: "weekdays", values: { time } };
      return { key: "weekly", values: { days: formatWeekdays(schedule.days, locale), time } };
    case "monthly":
      return { key: "monthly", values: { day: schedule.day, time } };
  }
}

/** The Repeats column: the cadence alone; weekly is the days themselves. */
export type RepeatsSummary =
  | { key: "once" | "hourly" | "daily" | "weekdays" }
  | { key: "everyHours"; hours: number }
  | { key: "weekly"; days: string }
  | { key: "monthly"; day: number };

export function summarizeRepeats(schedule: TaskSchedule, locale: string): RepeatsSummary {
  switch (schedule.kind) {
    case "hourly":
      return schedule.hours === 1
        ? { key: "hourly" }
        : { key: "everyHours", hours: schedule.hours };
    case "once":
    case "daily":
      return { key: schedule.kind };
    case "weekly":
      if (schedule.days.length === ISO_WEEKDAYS.length) return { key: "daily" };
      if (isWorkdays(schedule.days)) return { key: "weekdays" };
      return {
        key: "weekly",
        days: listFormat(locale, { style: "short", type: "unit" }).format(
          schedule.days.map((day) => weekdayName(day, locale, "short")),
        ),
      };
    case "monthly":
      return { key: "monthly", day: schedule.day };
  }
}

/** The day in `timeZone` (default: this browser's), so it agrees with a time shown in that zone. */
export function formatRunDay(
  epochMs: number,
  now: number,
  locale: string,
  timeZone?: string,
): string {
  const dayStart = (at: number) => {
    const parts = dateFormat("en-US", { timeZone, year: "numeric", month: "numeric", day: "numeric" })
      .formatToParts(at);
    const part = (type: string) => Number(parts.find((entry) => entry.type === type)?.value);
    return Date.UTC(part("year"), part("month") - 1, part("day"));
  };
  // Rounded: a DST change makes one day 23 or 25 hours long.
  const days = Math.round((dayStart(epochMs) - dayStart(now)) / 86_400_000);
  if (days === 0 || days === 1) {
    const word = relativeFormat(locale).format(days, "day");
    return word.charAt(0).toLocaleUpperCase(locale) + word.slice(1);
  }
  return dateFormat(locale, { weekday: "short", month: "short", day: "numeric", timeZone }).format(epochMs);
}

export function relativeTimeParts(
  target: number,
  now: number,
): [number, Intl.RelativeTimeFormatUnit] {
  const seconds = Math.round((target - now) / 1000);
  if (Math.abs(seconds) < 60) return [seconds, "second"];
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return [minutes, "minute"];
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return [hours, "hour"];
  const days = Math.round(hours / 24);
  if (Math.abs(days) < 30) return [days, "day"];
  const months = Math.round(days / 30);
  if (Math.abs(months) < 12) return [months, "month"];
  return [Math.round(days / 365), "year"];
}

export function browserTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

const AUTO_TITLE_LENGTH = 60;

/** A blank name falls back to the instructions' first line, cut at a word near 60 characters. */
export function titleFromPrompt(prompt: string): string {
  const line = prompt.trim().split("\n")[0]?.replace(/\s+/g, " ").trim() ?? "";
  // By code point: cutting by code unit can split an emoji into a lone surrogate, which the server refuses.
  const chars = Array.from(line);
  if (chars.length <= AUTO_TITLE_LENGTH) return line;
  const cut = chars.slice(0, AUTO_TITLE_LENGTH).join("");
  const space = cut.lastIndexOf(" ");
  return `${(space > AUTO_TITLE_LENGTH / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
}
