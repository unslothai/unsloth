// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLocale, useT } from "@/i18n";
import type { ProfileStats } from "../../api/profile-stats";
import {
  formatCompactNumber,
  formatDayCount,
  formatDuration,
  formatFullNumber,
} from "../../utils/stats-format";
import { StatTile } from "./stat-primitives";

/** The five headline numbers, mirroring the app's top-of-profile summary. */
export function StatsHighlights({ stats }: { stats: ProfileStats }) {
  const t = useT();
  const locale = useLocale();
  const { totals, streak, peakDay, longestChat } = stats;

  return (
    <div className="grid grid-cols-2 gap-y-5 rounded-2xl border border-border bg-background dark:border-transparent dark:bg-white/[0.06] px-4 py-5 sm:grid-cols-3 lg:grid-cols-5">
      <StatTile
        value={formatCompactNumber(totals.totalTokens, locale)}
        label={t("settings.profile.stats.lifetimeTokens")}
        hint={formatFullNumber(totals.totalTokens, locale)}
      />
      <StatTile
        value={peakDay ? formatCompactNumber(peakDay.tokens, locale) : "—"}
        label={t("settings.profile.stats.peakTokens")}
        {...(peakDay ? { hint: peakDay.date } : {})}
      />
      <StatTile
        value={longestChat ? formatDuration(longestChat.seconds) : "—"}
        label={t("settings.profile.stats.longestChat")}
        {...(longestChat?.title ? { hint: longestChat.title } : {})}
      />
      <StatTile
        value={formatDayCount(streak.current, locale)}
        label={t("settings.profile.stats.currentStreak")}
      />
      <StatTile
        value={formatDayCount(streak.longest, locale)}
        label={t("settings.profile.stats.longestStreak")}
      />
    </div>
  );
}
