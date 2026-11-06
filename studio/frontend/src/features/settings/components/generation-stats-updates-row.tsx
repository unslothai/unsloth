// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedPill } from "@/components/segmented-control";
import { useChatPreferencesStore } from "@/features/chat";
import { GENERATION_STATS_UPDATES } from "@/features/chat/lib/live-generation-stats";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useId } from "react";
import { SettingsRow } from "./settings-row";

/** Speed and context stats, in Settings > Chat > Chat defaults. */
export function GenerationStatsUpdatesRow() {
  const t = useT();
  const pillId = useId();
  const generationStatsUpdates = useChatPreferencesStore(
    (state) => state.generationStatsUpdates,
  );
  const setGenerationStatsUpdates = useChatPreferencesStore(
    (state) => state.setGenerationStatsUpdates,
  );
  return (
    <SettingsRow
      label={t("settings.chat.generationStatsUpdates")}
      description={t("settings.chat.generationStatsUpdatesDescription")}
      hint={t("settings.chat.generationStatsUpdatesHint")}
    >
      <div
        className="hub-tab-toggle inline-flex h-8 items-center rounded-full"
        // biome-ignore lint/a11y/useSemanticElements: matches Scroll while generating; a fieldset's default box would break the pill row.
        role="group"
        aria-label={t("settings.chat.generationStatsUpdates")}
      >
        {GENERATION_STATS_UPDATES.map((mode) => (
          <button
            key={mode}
            type="button"
            aria-pressed={generationStatsUpdates === mode}
            onClick={() => setGenerationStatsUpdates(mode)}
            className={cn(
              "relative inline-flex h-8 cursor-pointer items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              generationStatsUpdates === mode
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {generationStatsUpdates === mode && (
              <SegmentedPill group={`${pillId}-generation-stats`} />
            )}
            {t(
              mode === "realtime"
                ? "settings.chat.generationStatsUpdatesRealtime"
                : "settings.chat.generationStatsUpdatesSettled",
            )}
          </button>
        ))}
      </div>
    </SettingsRow>
  );
}
