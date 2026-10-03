// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatRelativeTime, type useLocale } from "@/i18n";

export function ago(ms: number, locale: ReturnType<typeof useLocale>): string {
  const mins = Math.floor((Date.now() - ms) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return formatRelativeTime(locale, -mins, "minute");
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return formatRelativeTime(locale, -hrs, "hour");
  return formatRelativeTime(locale, -Math.floor(hrs / 24), "day");
}
