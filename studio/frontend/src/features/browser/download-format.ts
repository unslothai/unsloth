// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Decimal units, as file managers show sizes. */
export function formatSize(bytes: number, locale: string): string {
  const units = ["byte", "kilobyte", "megabyte", "gigabyte"] as const;
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit++;
  }
  return new Intl.NumberFormat(locale, {
    style: "unit",
    unit: units[unit],
    unitDisplay: "short",
    maximumFractionDigits: value < 10 && unit > 0 ? 1 : 0,
  }).format(value);
}

/** Each platform's own name for showing a file in its folder. */
export function revealLabelKey() {
  const platform = typeof navigator === "undefined" ? "" : navigator.userAgent;
  if (/Mac/i.test(platform)) return "browser.pages.showInFinder" as const;
  if (/Windows/i.test(platform)) return "browser.pages.showInExplorer" as const;
  return "browser.pages.showInFolder" as const;
}
