// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Formatting helpers for the profile stats panel.
 *
 * Kept free of React so the numbers can be unit-tested directly.
 */

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

/** Magnitude of the compact form, so "past 100 of a unit" is asked of Intl
 * rather than derived from a hardcoded 1e3/1e6/1e9 ladder that only matches
 * locales grouping in thousands. ja and zh group in 万, hi in लाख.
 *
 * Probed through a latn formatter, never the display one: the default numbering
 * system is per-locale AND per-ICU-build, and ar-EG / ar-SA resolve to `arab`,
 * where the integer part is "١" and Number() gives NaN. NaN < 100 is false, so
 * every Arabic value silently lost its decimal ("٢ مليون" for 1.9M). The unit
 * grouping is identical across numbering systems, so this only changes the
 * digits we parse, not which unit Intl picked. */
function compactInteger(locale: Locale, value: number): number {
  for (const part of compactFormatter(locale, 1, "latn").formatToParts(value)) {
    if (part.type === "integer") {
      return Math.abs(Number(part.value));
    }
  }
  return 0;
}

/** Whether the compact form actually applied a unit (K / 万 / लाख), probed in latn for the
 * same reason as compactInteger. */
function hasCompactUnit(locale: Locale, value: number): boolean {
  return compactFormatter(locale, 1, "latn")
    .formatToParts(value)
    .some((part) => part.type === "compact");
}

/**
 * Compact form used on every stat tile: 12.3K, 4.5M, 19.8B in English, and
 * each locale's own units elsewhere (1.2万 in ja, 1,9 Mrd. in de, 1.2 लाख in
 * hi). One decimal below 100 of a unit keeps "1.9B" readable; above it the
 * decimal is noise, and asking for zero digits also avoids "1000K", which is
 * not compact.
 */
export function formatCompactNumber(value: number, locale: Locale): string {
  if (!Number.isFinite(value)) return "0";
  // Below the locale's first compact unit there is no unit to be a fraction OF, and these
  // are whole things: averageTokensPerChat is the one fractional caller, and "12.5 tokens"
  // reads as false precision where the pre-localization code said 13. Asked of Intl (is there
  // a `compact` part?) rather than a hardcoded 1000, because the first unit is per-locale:
  // en and hi compact at 1K, ja and de not until 万 and Mio.
  if (!hasCompactUnit(locale, value)) return compactFormatter(locale, 0).format(value);
  return compactInteger(locale, value) < 100
    ? compactFormatter(locale, 1).format(value)
    : compactFormatter(locale, 0).format(value);
}

/** The exact count behind a tile, grouped the way the chosen locale groups. */
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

/** Locale-aware day unit, including languages with multiple plural forms. */
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

// Intl formats calendar units such as weeks, but not app-specific nouns such
// as tokens, messages, or training steps. Keep those forms together here so
// every call site selects them with the same CLDR plural category.
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
} satisfies Record<Locale, Record<LexicalProfileCountUnit, CountTemplate>>;

type ProfileCountLocale = Locale;

/** A localized count phrase for the dynamic nouns used by profile stats. */
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

/** Compact duration for chat and training time: 4h 8m, 12m 30s, 45s. */
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

/**
 * Bucket a day's tokens into one of five heatmap intensities (0 = empty).
 * Thresholds are relative to the busiest day so any usage scale looks alive.
 */
export function heatLevel(tokens: number, peak: number): 0 | 1 | 2 | 3 | 4 {
  if (tokens <= 0) return 0;
  if (peak <= 0) return 1;
  const ratio = tokens / peak;
  if (ratio > 0.6) return 4;
  if (ratio > 0.3) return 3;
  if (ratio > 0.1) return 2;
  return 1;
}

/** Local YYYY-MM-DD, matching the backend's day keys (which use local time). */
export function toLocalDayKey(date: Date): string {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/** Parse a backend day key as a local date (not UTC, which would shift a day). */
export function parseDayKey(key: string): Date {
  const [year, month, day] = key.split("-").map(Number);
  return new Date(year, (month ?? 1) - 1, day ?? 1);
}

export type ActivityMode = "daily" | "weekly" | "cumulative";

/**
 * Recast the dense daily series for the selected mode. Weekly sums each
 * calendar week onto its days so the grid shows week-level intensity;
 * cumulative is the running total across the displayed window, not lifetime,
 * since the backend caps the series and seeding it with everything older would
 * flatten every bar against a baseline the grid cannot show.
 */
/**
 * What to subtract from a cumulative series once the grid drops older days.
 * Without it the first visible bar opens at the hidden total and the whole
 * window flattens against a baseline the user cannot see.
 */
export function windowBaseline(
  values: number[],
  start: number,
  mode: ActivityMode,
): number {
  if (mode !== "cumulative" || start <= 0) return 0;
  return values[start - 1] ?? 0;
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

  // Weekly: every day carries the total of the Monday-started week it sits in.
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
