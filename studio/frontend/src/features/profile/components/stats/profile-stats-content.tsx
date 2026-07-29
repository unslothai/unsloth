// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { useProfileStats } from "../../hooks/use-profile-stats";
import { ActivityInsightsCard, TopModelsCard } from "./insights-card";
import { StatsCard } from "./stat-primitives";
import { StatsHighlights } from "./stats-highlights";
import { StatsSkeleton } from "./stats-skeleton";
import { TokenActivityCard } from "./token-activity-card";
import { TrainingHighlightsCard } from "./training-card";

/**
 * Everything below the personalization form on the Profile tab: headline
 * numbers, activity grid, insights and training.
 *
 * All of it comes from `/api/profile/stats`, which reads local history only.
 *
 * Loaded lazily by `profile-stats-panel.tsx` so none of it reaches the main
 * bundle, since the Profile tab is the only place it renders.
 */
export function ProfileStatsContent() {
  const t = useT();
  const { stats, loading, error, reload } = useProfileStats();

  if (loading && stats === null) {
    return <StatsSkeleton />;
  }

  if (error !== null && stats === null) {
    return (
      <StatsCard title={t("settings.profile.stats.title")}>
        <div className="flex flex-col items-center gap-3 py-4 text-center">
          <p className="text-xs text-muted-foreground">{error}</p>
          <Button type="button" size="sm" variant="outline" onClick={reload}>
            {t("settings.profile.stats.retry")}
          </Button>
        </div>
      </StatsCard>
    );
  }

  if (stats === null) return null;

  const hasChats = stats.totals.messages > 0;
  const hasTraining = stats.training.runs > 0;

  return (
    <div className="flex w-full flex-col gap-4">
      <header className="flex flex-col gap-0.5">
        <h2
          data-settings-label={t("settings.profile.stats.title")}
          className="text-base font-semibold font-heading text-foreground"
        >
          {t("settings.profile.stats.title")}
        </h2>
        <p className="text-xs text-muted-foreground">
          {t("settings.profile.stats.subtitle")}
        </p>
      </header>

      <StatsHighlights stats={stats} />

      {hasChats ? (
        <>
          <TokenActivityCard daily={stats.daily} />
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <ActivityInsightsCard stats={stats} />
            <TopModelsCard stats={stats} />
          </div>
        </>
      ) : (
        <StatsCard>
          <p className="py-6 text-center text-xs text-muted-foreground">
            {t("settings.profile.stats.emptyChats")}
          </p>
        </StatsCard>
      )}

      {hasTraining ? <TrainingHighlightsCard stats={stats} /> : null}

      <p className="px-1 pb-2 text-ui-11 text-muted-foreground">
        {t("settings.profile.stats.privacyNote")}
      </p>
    </div>
  );
}
