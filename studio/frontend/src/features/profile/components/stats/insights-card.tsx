


import { useLocale, useT } from "@/i18n";
import type { ProfileStats } from "../../api/profile-stats";
import {
  formatCompactNumber,
  formatDuration,
  formatFullNumber,
  formatMilliseconds,
  formatProfileCount,
} from "../../utils/stats-format";
import { StatMeter, StatRow, StatsCard } from "./stat-primitives";

/** Left column: the "how you use Unsloth" numbers. */
export function ActivityInsightsCard({ stats }: { stats: ProfileStats }) {
  const t = useT();
  const locale = useLocale();
  const { totals, speed } = stats;
  const averageTokensPerChat =
    totals.threads > 0 ? totals.totalTokens / totals.threads : 0;
  const cacheShare =
    totals.promptTokens > 0 ? totals.cachedTokens / totals.promptTokens : 0;

  return (
    <StatsCard title={t("settings.profile.stats.insightsTitle")}>
      <div className="flex flex-col divide-y divide-border/60">
        <StatRow
          label={t("settings.profile.stats.totalChats")}
          value={formatFullNumber(totals.threads, locale)}
        />
        <StatRow
          label={t("settings.profile.stats.totalMessages")}
          value={formatFullNumber(totals.messages, locale)}
        />
        <StatRow
          label={t("settings.profile.stats.tokensIn")}
          value={formatCompactNumber(totals.promptTokens, locale)}
        />
        <StatRow
          label={t("settings.profile.stats.tokensOut")}
          value={formatCompactNumber(totals.completionTokens, locale)}
        />
        <StatRow
          label={t("settings.profile.stats.cachedTokens")}
          value={
            cacheShare > 0
              ? t("settings.profile.stats.cachedValue", {
                  tokens: formatCompactNumber(totals.cachedTokens, locale),
                  percent: Math.round(cacheShare * 100),
                })
              : formatCompactNumber(totals.cachedTokens, locale)
          }
        />
        <StatRow
          label={t("settings.profile.stats.avgTokensPerChat")}
          value={formatCompactNumber(averageTokensPerChat, locale)}
        />
        <StatRow
          label={t("settings.profile.stats.timeInChat")}
          value={formatDuration(totals.chatSeconds)}
        />
        <StatRow
          label={t("settings.profile.stats.activeDays")}
          value={formatFullNumber(totals.activeDays, locale)}
        />
        <StatRow
          label={t("settings.profile.stats.toolCalls")}
          value={formatFullNumber(totals.toolCalls, locale)}
        />
        <StatRow
          label={t("settings.profile.stats.attachments")}
          value={formatFullNumber(totals.attachments, locale)}
        />
        <StatRow
          label={t("settings.profile.stats.avgSpeed")}
          value={
            speed.averageTokensPerSecond === null
              ? "—"
              : t("settings.profile.stats.tokensPerSecond", {
                  value: speed.averageTokensPerSecond.toFixed(1),
                })
          }
        />
        <StatRow
          label={t("settings.profile.stats.bestSpeed")}
          value={
            speed.bestTokensPerSecond === null
              ? "—"
              : t("settings.profile.stats.tokensPerSecond", {
                  value: speed.bestTokensPerSecond.toFixed(1),
                })
          }
        />
        <StatRow
          label={t("settings.profile.stats.firstToken")}
          value={
            speed.averageFirstTokenMs === null
              ? "—"
              : formatMilliseconds(speed.averageFirstTokenMs)
          }
        />
      </div>
    </StatsCard>
  );
}

/** Right column: model leaderboard, ranked by tokens exchanged. */
export function TopModelsCard({ stats }: { stats: ProfileStats }) {
  const t = useT();
  const locale = useLocale();
  const models = stats.models;
  const peak = models.reduce((max, model) => Math.max(max, model.tokens), 0);

  return (
    <StatsCard
      title={t("settings.profile.stats.topModelsTitle")}
      description={t("settings.profile.stats.topModelsDescription")}
    >
      {models.length === 0 ? (
        <p className="py-6 text-center text-xs text-muted-foreground">
          {t("settings.profile.stats.noModels")}
        </p>
      ) : (
        <ol className="flex flex-col gap-3">
          {models.map((model, index) => (
            <li key={model.id} className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between gap-3">
                <span className="flex min-w-0 items-baseline gap-2">
                  <span className="w-4 shrink-0 text-ui-11 tabular-nums text-muted-foreground">
                    {index + 1}
                  </span>
                  <span
                    className="min-w-0 truncate text-sm text-foreground"
                    title={model.id}
                  >
                    {model.label}
                  </span>
                </span>
                <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                  {t("settings.profile.stats.modelSummary", {
                    tokens: formatProfileCount(
                      model.tokens,
                      "token",
                      locale,
                      formatCompactNumber(model.tokens, locale),
                    ),
                    messages: formatProfileCount(
                      model.messages,
                      "message",
                      locale,
                    ),
                  })}
                </span>
              </div>
              <StatMeter progress={peak > 0 ? model.tokens / peak : 0} />
            </li>
          ))}
        </ol>
      )}
    </StatsCard>
  );
}
