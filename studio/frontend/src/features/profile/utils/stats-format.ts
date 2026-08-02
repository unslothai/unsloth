// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Formatting helpers for the profile stats panel.
 *
 * Kept free of React so the numbers can be unit-tested directly.
 */

const TRAILING_ZERO_DECIMAL = /\.0$/;

/** Compact form used on every stat tile: 12.3K, 4.5M, 19.8B. */
export function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return "0";
  const abs = Math.abs(value);
  if (abs < 1000) return String(Math.round(value));

  const units: Array<{ limit: number; suffix: string }> = [
    { limit: 1e12, suffix: "T" },
    { limit: 1e9, suffix: "B" },
    { limit: 1e6, suffix: "M" },
    { limit: 1e3, suffix: "K" },
  ];
  for (const [index, { limit, suffix }] of units.entries()) {
    if (abs < limit) continue;
    const scaled = value / limit;
    // One decimal below 100 keeps "1.9B" readable; above it the decimal is noise.
    const rounded =
      Math.abs(scaled) >= 100 ? Math.round(scaled) : Number(scaled.toFixed(1));
    // Rounding can push a value over the next boundary, and "1000K" is not
    // compact. Step up a unit rather than print four digits.
    const next = units[index - 1];
    if (next && Math.abs(rounded) >= 1000) {
      return `${(value / next.limit).toFixed(1).replace(TRAILING_ZERO_DECIMAL, "")}${next.suffix}`;
    }
    const text =
      Math.abs(scaled) >= 100 ? rounded.toString() : scaled.toFixed(1);
    return `${text.replace(TRAILING_ZERO_DECIMAL, "")}${suffix}`;
  }
  return String(Math.round(value));
}

export function formatFullNumber(value: number): string {
  if (!Number.isFinite(value)) return "0";
  return Math.round(value).toLocaleString();
}

/** Locale-aware day unit, including languages with multiple plural forms. */
export function formatDayCount(value: number, locale: string): string {
  return new Intl.NumberFormat(locale, {
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
} satisfies Record<string, Record<LexicalProfileCountUnit, CountTemplate>>;

type ProfileCountLocale = keyof typeof PROFILE_COUNT_TEMPLATES;

/** A localized count phrase for the dynamic nouns used by profile stats. */
export function formatProfileCount(
  value: number,
  unit: ProfileCountUnit,
  locale: ProfileCountLocale,
  displayValue?: string,
): string {
  const finiteValue = Number.isFinite(value) ? value : 0;
  if (unit === "week") {
    return new Intl.NumberFormat(locale, {
      style: "unit",
      unit: "week",
      unitDisplay: "long",
    }).format(finiteValue);
  }

  const category = new Intl.PluralRules(locale).select(
    finiteValue,
  ) as PluralCategory;
  const templates: CountTemplate = PROFILE_COUNT_TEMPLATES[locale][unit];
  const template = templates[category] ?? templates.other;
  const formattedValue =
    displayValue ?? new Intl.NumberFormat(locale).format(finiteValue);
  return template.replace("{value}", formattedValue);
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
