


import type { Locale } from "./messages";

const relativeTimeFormatters = new Map<string, Intl.RelativeTimeFormat>();

function getFormatter(
  locale: Locale,
  style: "short" | "long",
): Intl.RelativeTimeFormat {
  const key = `${locale}:${style}`;
  let formatter = relativeTimeFormatters.get(key);
  if (!formatter) {
    formatter = new Intl.RelativeTimeFormat(locale, {
      numeric: "always",
      style,
    });
    relativeTimeFormatters.set(key, formatter);
  }
  return formatter;
}

export function formatRelativeTime(
  locale: Locale,
  value: number,
  unit: Intl.RelativeTimeFormatUnit,
): string {
  // format() throws RangeError on non-finite input, and an unparseable
  // timestamp reaches here as NaN. Callers render during a React commit, so a
  // throw would unmount the tree; degrade to empty text instead.
  if (!Number.isFinite(value)) {
    return "";
  }
  const short = getFormatter(locale, "short");
  const formatted = short.format(value, unit);
  // Some CLDR short patterns drop the past/future marker. Arabic months in the
  // "few" plural category (3-10) render "5 months ago" as "خلال 5 أشهر" ("in
  // 5 months"), which is byte-identical to the future form. Where the two
  // directions are indistinguishable the long style still carries the marker,
  // so use it rather than report the wrong tense.
  if (value !== 0 && formatted === short.format(-value, unit)) {
    return getFormatter(locale, "long").format(value, unit);
  }
  return formatted;
}
