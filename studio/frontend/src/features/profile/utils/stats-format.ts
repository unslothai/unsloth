// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Locale } from "@/i18n";

const COMPACT_FORMATTERS = new Map<string, Intl.NumberFormat>();
const FULL_FORMATTERS = new Map<Locale, Intl.NumberFormat>();

function compactFormatter(
  locale: Locale,
  maximumFractionDigits: 0 | 1,
  numberingSystem?: "latn",
): Intl.NumberFormat {
  const key = `${locale}:${maximumFractionDigits}:${numberingSystem ?? ""}`;
  const cached = COMPACT_FORMATTERS.get(key);
  if (cached) {
    return cached;
  }
  const formatter = new Intl.NumberFormat(locale, {
    notation: "compact",
    maximumFractionDigits,
    ...(numberingSystem ? { numberingSystem } : {}),
  });
  COMPACT_FORMATTERS.set(key, formatter);
  return formatter;
}

/** Probed through a latn formatter: ar-EG/ar-SA default to arab digits, which Number() reads as NaN. */
function compactInteger(locale: Locale, value: number): number {
  for (const part of compactFormatter(locale, 1, "latn").formatToParts(value)) {
    if (part.type === "integer") {
      return Math.abs(Number(part.value));
    }
  }
  return 0;
}

function hasCompactUnit(locale: Locale, value: number): boolean {
  return compactFormatter(locale, 1, "latn")
    .formatToParts(value)
    .some((part) => part.type === "compact");
}

/** One decimal below 100 of a unit; zero digits above it also avoids "1000K". */
export function formatCompactNumber(value: number, locale: Locale): string {
  if (!Number.isFinite(value)) return "0";
  // Below a locale's first compact unit values stay whole (12.5 tokens reads as false precision;
  // averageTokensPerChat is the one fractional caller); compaction is asked of Intl, not hardcoded
  // at 1000, since en and hi compact at 1K while ja and de wait for 万 and Mio.
  if (!hasCompactUnit(locale, value))
    return compactFormatter(locale, 0).format(value);
  return compactInteger(locale, value) < 100
    ? compactFormatter(locale, 1).format(value)
    : compactFormatter(locale, 0).format(value);
}

export function formatFullNumber(value: number, locale: Locale): string {
  if (!Number.isFinite(value)) return "0";
  const cached = FULL_FORMATTERS.get(locale);
  const formatter = cached ?? new Intl.NumberFormat(locale);
  if (!cached) FULL_FORMATTERS.set(locale, formatter);
  return formatter.format(Math.round(value));
}

const DAY_FORMATTERS = new Map<Locale, Intl.NumberFormat>();
const WEEK_FORMATTERS = new Map<Locale, Intl.NumberFormat>();
const COUNT_FORMATTERS = new Map<Locale, Intl.NumberFormat>();
const PLURAL_RULES = new Map<Locale, Intl.PluralRules>();

function cachedFormatter(
  cache: Map<Locale, Intl.NumberFormat>,
  locale: Locale,
  options?: Intl.NumberFormatOptions,
): Intl.NumberFormat {
  const cached = cache.get(locale);
  if (cached) {
    return cached;
  }
  const formatter = new Intl.NumberFormat(locale, options);
  cache.set(locale, formatter);
  return formatter;
}

function cachedPluralRules(locale: Locale): Intl.PluralRules {
  const cached = PLURAL_RULES.get(locale);
  if (cached) {
    return cached;
  }
  const rules = new Intl.PluralRules(locale);
  PLURAL_RULES.set(locale, rules);
  return rules;
}

export function formatDayCount(value: number, locale: Locale): string {
  return cachedFormatter(DAY_FORMATTERS, locale, {
    style: "unit",
    unit: "day",
    unitDisplay: "long",
  }).format(value);
}

export type ProfileCountUnit = "week" | "token" | "message" | "step";

type LexicalProfileCountUnit = Exclude<ProfileCountUnit, "week">;
type PluralCategory = "zero" | "one" | "two" | "few" | "many" | "other";
type CountTemplate = { other: string } & Partial<
  Record<PluralCategory, string>
>;

// Intl has no plural forms for app nouns like tokens or messages, so they live here.
const PROFILE_COUNT_TEMPLATES = {
  en: {
    token: { one: "{value} token", other: "{value} tokens" },
    message: { one: "{value} message", other: "{value} messages" },
    step: { one: "{value} step", other: "{value} steps" },
  },
  "zh-CN": {
    token: { other: "{value} 个 token" },
    message: { other: "{value} 条消息" },
    step: { other: "{value} 步" },
  },
  ja: {
    token: { other: "{value} トークン" },
    message: { other: "{value} 件のメッセージ" },
    step: { other: "{value} ステップ" },
  },
  ko: {
    token: { other: "{value} 토큰" },
    message: { other: "메시지 {value}개" },
    step: { other: "{value} 스텝" },
  },
  es: {
    token: { one: "{value} token", other: "{value} tokens" },
    message: { one: "{value} mensaje", other: "{value} mensajes" },
    step: { one: "{value} paso", other: "{value} pasos" },
  },
  "pt-BR": {
    token: { one: "{value} token", other: "{value} tokens" },
    message: { one: "{value} mensagem", other: "{value} mensagens" },
    step: { one: "{value} passo", other: "{value} passos" },
  },
  fr: {
    token: { one: "{value} token", other: "{value} tokens" },
    message: { one: "{value} message", other: "{value} messages" },
    step: { one: "{value} étape", other: "{value} étapes" },
  },
  de: {
    token: { one: "{value} Token", other: "{value} Tokens" },
    message: { one: "{value} Nachricht", other: "{value} Nachrichten" },
    step: { one: "{value} Schritt", other: "{value} Schritte" },
  },
  it: {
    token: { one: "{value} token", other: "{value} token" },
    message: { one: "{value} messaggio", other: "{value} messaggi" },
    step: { one: "{value} step", other: "{value} step" },
  },
  ru: {
    token: {
      one: "{value} токен",
      few: "{value} токена",
      many: "{value} токенов",
      other: "{value} токена",
    },
    message: {
      one: "{value} сообщение",
      few: "{value} сообщения",
      many: "{value} сообщений",
      other: "{value} сообщения",
    },
    step: {
      one: "{value} шаг",
      few: "{value} шага",
      many: "{value} шагов",
      other: "{value} шага",
    },
  },
  sv: {
    token: { one: "{value} token", other: "{value} tokens" },
    message: { one: "{value} meddelande", other: "{value} meddelanden" },
    step: { one: "{value} steg", other: "{value} steg" },
  },
  hi: {
    token: { one: "{value} टोकन", other: "{value} टोकन" },
    message: { one: "{value} संदेश", other: "{value} संदेश" },
    step: { one: "{value} स्टेप", other: "{value} स्टेप" },
  },
  ar: {
    token: {
      zero: "{value} توكن",
      one: "توكن واحد",
      two: "توكنان",
      few: "{value} توكنات",
      many: "{value} توكنًا",
      other: "{value} توكن",
    },
    message: {
      zero: "{value} رسالة",
      one: "رسالة واحدة",
      two: "رسالتان",
      few: "{value} رسائل",
      many: "{value} رسالة",
      other: "{value} رسالة",
    },
    step: {
      zero: "{value} خطوة",
      one: "خطوة واحدة",
      two: "خطوتان",
      few: "{value} خطوات",
      many: "{value} خطوة",
      other: "{value} خطوة",
    },
  },
  he: {
    token: { one: "טוקן אחד", two: "{value} טוקנים", other: "{value} טוקנים" },
    message: {
      one: "הודעה אחת",
      two: "{value} הודעות",
      other: "{value} הודעות",
    },
    step: { one: "צעד אחד", two: "{value} צעדים", other: "{value} צעדים" },
  },
  "zh-TW": {
    token: { other: "{value} 個 token" },
    message: { other: "{value} 則訊息" },
    step: { other: "{value} 步" },
  },
  "es-419": {
    token: { one: "{value} token", other: "{value} tokens" },
    message: { one: "{value} mensaje", other: "{value} mensajes" },
    step: { one: "{value} paso", other: "{value} pasos" },
  },
  id: {
    token: { other: "{value} token" },
    message: { other: "{value} pesan" },
    step: { other: "{value} langkah" },
  },
  vi: {
    token: { other: "{value} token" },
    message: { other: "{value} tin nhắn" },
    step: { other: "{value} bước" },
  },
  tr: {
    token: { other: "{value} token" },
    message: { other: "{value} mesaj" },
    step: { other: "{value} adım" },
  },
  pl: {
    token: {
      one: "{value} token",
      few: "{value} tokeny",
      many: "{value} tokenów",
      other: "{value} tokena",
    },
    message: {
      one: "{value} wiadomość",
      few: "{value} wiadomości",
      many: "{value} wiadomości",
      other: "{value} wiadomości",
    },
    step: {
      one: "{value} krok",
      few: "{value} kroki",
      many: "{value} kroków",
      other: "{value} kroku",
    },
  },
  uk: {
    token: {
      one: "{value} токен",
      few: "{value} токени",
      many: "{value} токенів",
      other: "{value} токена",
    },
    message: {
      one: "{value} повідомлення",
      few: "{value} повідомлення",
      many: "{value} повідомлень",
      other: "{value} повідомлення",
    },
    step: {
      one: "{value} крок",
      few: "{value} кроки",
      many: "{value} кроків",
      other: "{value} кроку",
    },
  },
  th: {
    token: { other: "{value} โทเค็น" },
    message: { other: "{value} ข้อความ" },
    step: { other: "{value} สเต็ป" },
  },
  fa: {
    token: { other: "{value} توکن" },
    message: { other: "{value} پیام" },
    step: { other: "{value} گام" },
  },
  cs: {
    token: {
      one: "{value} token",
      few: "{value} tokeny",
      many: "{value} tokenu",
      other: "{value} tokenů",
    },
    message: {
      one: "{value} zpráva",
      few: "{value} zprávy",
      many: "{value} zprávy",
      other: "{value} zpráv",
    },
    step: {
      one: "{value} krok",
      few: "{value} kroky",
      many: "{value} kroku",
      other: "{value} kroků",
    },
  },
  nl: {
    token: { one: "{value} token", other: "{value} tokens" },
    message: { one: "{value} bericht", other: "{value} berichten" },
    step: { one: "{value} stap", other: "{value} stappen" },
  },
} satisfies Record<Locale, Record<LexicalProfileCountUnit, CountTemplate>>;

type ProfileCountLocale = Locale;

export function formatProfileCount(
  value: number,
  unit: ProfileCountUnit,
  locale: ProfileCountLocale,
  displayValue?: string,
): string {
  const finiteValue = Number.isFinite(value) ? value : 0;
  if (unit === "week") {
    return cachedFormatter(WEEK_FORMATTERS, locale, {
      style: "unit",
      unit: "week",
      unitDisplay: "long",
    }).format(finiteValue);
  }

  const category = cachedPluralRules(locale).select(
    finiteValue,
  ) as PluralCategory;
  const templates: CountTemplate = PROFILE_COUNT_TEMPLATES[locale][unit];
  const template = templates[category] ?? templates.other;
  const formattedValue =
    displayValue ??
    cachedFormatter(COUNT_FORMATTERS, locale).format(finiteValue);
  return template.replace("{value}", () => formattedValue);
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0m";
  const total = Math.round(seconds);
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;

  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return secs > 0 ? `${minutes}m ${secs}s` : `${minutes}m`;
  return `${secs}s`;
}

export function formatMilliseconds(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export type HeatLevel = 0 | 1 | 2 | 3 | 4;

/** Quartiles of the active days, then the busiest. Not ratios to the peak: one outlier washed out every other day. */
export function heatThresholds(values: readonly number[]): [number, number, number, number] {
  const active = values.filter((value) => value > 0).sort((a, b) => a - b);
  if (active.length === 0) return [0, 0, 0, 0];
  const at = (fraction: number) =>
    active[Math.min(active.length - 1, Math.floor(fraction * active.length))] ?? 0;
  return [at(0.25), at(0.5), at(0.75), active.at(-1) ?? 0];
}

/** A day's shade, 0 for none. The busiest always takes the strongest: with 2 to 4 active days it is the top quartile itself. */
export function heatLevel(
  tokens: number,
  thresholds: readonly [number, number, number, number],
): HeatLevel {
  if (tokens <= 0) return 0;
  const [q1, q2, q3, peak] = thresholds;
  if (tokens > q3 || tokens >= peak) return 4;
  if (tokens > q2) return 3;
  if (tokens > q1) return 2;
  return 1;
}

export type ActivityCell<Day> = { key: string; day: Day | null; value: number };

/** The latest `columns` Monday-started weeks; days after today stay `null`. */
export function buildActivityColumns<Day extends { date: string }>(
  daily: readonly Day[],
  values: readonly number[],
  columns: number,
  mode: ActivityMode,
): ActivityCell<Day>[][] {
  const lastDay = daily.at(-1);
  if (!lastDay || columns <= 0) return [];
  const trailing = DAYS_PER_WEEK - 1 - mondayIndex(lastDay.date);
  const capacity = columns * DAYS_PER_WEEK - trailing;
  const start = Math.max(0, daily.length - capacity);
  const visible = daily.slice(start);
  const firstVisible = visible[0];
  if (!firstVisible) return [];
  const baseline = windowBaseline([...values], start, mode);

  const cells: ActivityCell<Day>[] = [];
  for (let index = 0; index < mondayIndex(firstVisible.date); index += 1) {
    cells.push({ key: `pad-${index}`, day: null, value: 0 });
  }
  for (const [index, day] of visible.entries()) {
    cells.push({ key: day.date, day, value: (values[start + index] ?? 0) - baseline });
  }
  for (let index = 0; index < trailing; index += 1) {
    cells.push({ key: `after-${index}`, day: null, value: 0 });
  }
  const grid: ActivityCell<Day>[][] = [];
  for (let index = 0; index < cells.length; index += DAYS_PER_WEEK) {
    grid.push(cells.slice(index, index + DAYS_PER_WEEK));
  }
  return grid;
}

const DAYS_PER_WEEK = 7;

/** 0 for Monday through 6 for Sunday. */
function mondayIndex(dayKey: string): number {
  return (parseDayKey(dayKey).getDay() + 6) % DAYS_PER_WEEK;
}

/** One label per shown 1st; a leading partial month yields to it, never the reverse (lost Jan). */
export function monthLabelColumns(
  grid: ReadonlyArray<ReadonlyArray<{ day: { date: string } | null }>>,
): Array<{ column: number; date: string }> {
  const labels: Array<{ column: number; date: string }> = [];
  for (const [column, week] of grid.entries()) {
    const first = week.find((cell) => cell.day?.date.endsWith("-01"))?.day;
    if (first) labels.push({ column, date: first.date });
  }
  const leading = grid[0]?.find((cell) => cell.day !== null)?.day;
  if (leading && labels[0]?.column !== 0 && (labels[0]?.column ?? Number.POSITIVE_INFINITY) >= 3) {
    labels.unshift({ column: 0, date: leading.date });
  }
  return labels;
}

/** Local YYYY-MM-DD, matching the backend's day keys (which use local time). */
export function toLocalDayKey(date: Date): string {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/** Parse as a local date; UTC would shift a day. */
export function parseDayKey(key: string): Date {
  const [year, month, day] = key.split("-").map(Number);
  return new Date(year, (month ?? 1) - 1, day ?? 1);
}

export type ActivityMode = "daily" | "weekly" | "cumulative";

/**
 * Subtracted from a cumulative series once older days drop, or the first visible bar opens at the
 * hidden total.
 */
export function windowBaseline(
  values: number[],
  start: number,
  mode: ActivityMode,
): number {
  if (mode !== "cumulative" || start <= 0) return 0;
  return values[start - 1] ?? 0;
}

export type ActivityGridCell = {
  day: { tokens: number } | null;
};

export type ActivityGridColumn = ActivityGridCell[];

export function activitySummaryForMode(
  grid: ActivityGridColumn[],
  mode: ActivityMode,
): number {
  if (grid.length === 0) return 0;
  if (mode === "weekly") {
    return grid.reduce((peak, column) => {
      let weekTokens = 0;
      for (const cell of column) {
        if (cell.day) weekTokens += cell.day.tokens;
      }
      return Math.max(peak, weekTokens);
    }, 0);
  }
  return grid.reduce(
    (sum, column) =>
      column.reduce((total, cell) => total + (cell.day?.tokens ?? 0), sum),
    0,
  );
}

export type WeekChartPoint = { week: string; tokens: number };

/**
 * One point per visible week, keyed by its first shown day; the week's largest value is its total
 * for weekly series or its end-of-week running total for cumulative ones.
 */
export function weekChartPoints(
  grid: Array<Array<{ day: { date: string } | null; value: number }>>,
): WeekChartPoint[] {
  const points: WeekChartPoint[] = [];
  for (const column of grid) {
    let week: string | null = null;
    let tokens = 0;
    for (const cell of column) {
      if (!cell.day) {
        continue;
      }
      week ??= cell.day.date;
      tokens = Math.max(tokens, cell.value);
    }
    if (week !== null) {
      points.push({ week, tokens });
    }
  }
  return points;
}

export function seriesForMode(
  daily: Array<{ date: string; tokens: number }>,
  mode: ActivityMode,
): number[] {
  if (mode === "daily") return daily.map((day) => day.tokens);

  if (mode === "cumulative") {
    let running = 0;
    return daily.map((day) => {
      running += day.tokens;
      return running;
    });
  }

  // Weekly: every day carries its Monday-started week's total.
  const weekTotals: number[] = [];
  const weekOfDay: number[] = [];
  let week = -1;
  for (const [index, day] of daily.entries()) {
    const isMonday = parseDayKey(day.date).getDay() === 1;
    if (index === 0 || isMonday) {
      week += 1;
      weekTotals[week] = 0;
    }
    weekTotals[week] = (weekTotals[week] ?? 0) + day.tokens;
    weekOfDay[index] = week;
  }
  return weekOfDay.map((weekIndex) => weekTotals[weekIndex] ?? 0);
}
