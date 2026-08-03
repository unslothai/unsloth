// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Locale } from "./messages";

const relativeTimeFormatters = new Map<Locale, Intl.RelativeTimeFormat>();

export function formatRelativeTime(
  locale: Locale,
  value: number,
  unit: Intl.RelativeTimeFormatUnit,
): string {
  let formatter = relativeTimeFormatters.get(locale);
  if (!formatter) {
    formatter = new Intl.RelativeTimeFormat(locale, {
      numeric: "always",
      style: "short",
    });
    relativeTimeFormatters.set(locale, formatter);
  }
  return formatter.format(value, unit);
}
