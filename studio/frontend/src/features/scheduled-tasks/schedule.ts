// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Scheduled task schedules: the form that edits one and the words that describe one. No imports,
 *  so the tests load it directly. */

/** Mirrors core/scheduled_tasks/schedule.py. Times are wall-clock in the task's time zone; `days`
 *  are ISO weekdays, 1 Monday to 7 Sunday; a monthly `day` a month lacks runs on its last day. */
export type TaskSchedule =
  | { kind: "once"; at: string }
  | { kind: "daily"; time: string }
  | { kind: "weekly"; time: string; days: number[] }
  | { kind: "monthly"; time: string; day: number };

/** What the form offers. Weekdays is a weekly schedule the form names for itself. */
export type RepeatKind = "once" | "daily" | "weekdays" | "weekly" | "monthly";

export const REPEAT_KINDS: readonly RepeatKind[] = [
  "once",
  "daily",
  "weekdays",
  "weekly",
  "monthly",
];

export const ISO_WEEKDAYS = [1, 2, 3, 4, 5, 6, 7] as const;
const WORKDAYS = [1, 2, 3, 4, 5];

export interface ScheduleForm {
  repeat: RepeatKind;
  /** HH:MM, 24-hour. */
  time: string;
  /** YYYY-MM-DD, for a one-time task. */
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

/** A new task: every day at 09:00, with the other kinds pre-filled from today so switching kind
 *  never lands on an empty field. A one-time task starts tomorrow, so 09:00 is still ahead. */
export function defaultScheduleForm(now: Date): ScheduleForm {
  const tomorrow = new Date(now);
  tomorrow.setDate(now.getDate() + 1);
  return {
    repeat: "daily",
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

/** Whether the form describes a schedule at all. Whether it is still ahead is the server's call. */
export function scheduleFormComplete(form: ScheduleForm): boolean {
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

/** "9:00 AM" or "09:00", as the locale writes the time of day. */
export function formatClock(time: string, locale: string): string {
  const [hour, minute] = time.split(":").map(Number);
  return new Intl.DateTimeFormat(locale, {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(Date.UTC(2000, 0, 1, hour, minute));
}

/** An ISO weekday's name as the locale writes it. */
export function weekdayName(day: number, locale: string, width: "short" | "long"): string {
  // 2024-01-01 was a Monday, so day d of that week is ISO weekday d.
  return new Intl.DateTimeFormat(locale, { weekday: width, timeZone: "UTC" }).format(
    Date.UTC(2024, 0, day),
  );
}

/** Short weekday names joined as the locale lists things: "Mon, Wed, and Fri". The long list
 *  style, since the short one writes English as "Mon, Wed, & Fri". */
export function formatWeekdays(days: readonly number[], locale: string): string {
  return new Intl.ListFormat(locale, { style: "long", type: "conjunction" }).format(
    days.map((day) => weekdayName(day, locale, "short")),
  );
}

export function sameSchedule(a: TaskSchedule, b: TaskSchedule): boolean {
  switch (a.kind) {
    case "once":
      return b.kind === "once" && a.at === b.at;
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

/** Which sentence describes a schedule, and what fills it in: the caller owns the wording. */
export type ScheduleSummary =
  | { key: "once"; values: { date: string; time: string } }
  | { key: "daily" | "weekdays"; values: { time: string } }
  | { key: "weekly"; values: { days: string; time: string } }
  | { key: "monthly"; values: { day: number; time: string } };

export function summarizeSchedule(schedule: TaskSchedule, locale: string): ScheduleSummary {
  const clock = schedule.kind === "once" ? schedule.at.split("T")[1] : schedule.time;
  const time = formatClock(clock, locale);
  switch (schedule.kind) {
    case "once": {
      const [year, month, day] = schedule.at.split("T")[0].split("-").map(Number);
      const dateFormat = new Intl.DateTimeFormat(locale, { dateStyle: "medium", timeZone: "UTC" });
      return { key: "once", values: { date: dateFormat.format(Date.UTC(year, month - 1, day)), time } };
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

/** A moment in the user's own zone, short enough for a row: "Thu, Oct 15, 9:00 AM". */
export function formatMoment(epochMs: number, locale: string): string {
  return new Intl.DateTimeFormat(locale, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(epochMs);
}

/** The largest whole unit for a distance in time, for Intl.RelativeTimeFormat. */
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

/** The browser's IANA zone, saved with a task so it keeps its wall-clock time across DST. */
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
  if (line.length <= AUTO_TITLE_LENGTH) return line;
  const cut = line.slice(0, AUTO_TITLE_LENGTH);
  const space = cut.lastIndexOf(" ");
  return `${(space > AUTO_TITLE_LENGTH / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
}
