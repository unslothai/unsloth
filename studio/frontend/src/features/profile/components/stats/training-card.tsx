


import { useLocale, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ProfileStats } from "../../api/profile-stats";
import {
  formatCompactNumber,
  formatDuration,
  formatFullNumber,
  formatProfileCount,
} from "../../utils/stats-format";
import { StatTile, StatsCard } from "./stat-primitives";

const STATUS_TONE: Record<string, string> = {
  completed: "text-primary",
  running: "text-foreground",
  error: "text-destructive",
  stopped: "text-muted-foreground",
};

/** Training-side counterpart to the chat stats: runs, steps, GPU time, loss. */
export function TrainingHighlightsCard({ stats }: { stats: ProfileStats }) {
  const t = useT();
  const locale = useLocale();
  const { training } = stats;

  return (
    <StatsCard
      title={t("settings.profile.stats.trainingTitle")}
      description={t("settings.profile.stats.trainingDescription")}
    >
      <div className="grid grid-cols-2 gap-y-4 sm:grid-cols-3 lg:grid-cols-6">
        <StatTile
          value={formatFullNumber(training.runs)}
          label={t("settings.profile.stats.trainingRuns")}
        />
        <StatTile
          value={formatFullNumber(training.completed)}
          label={t("settings.profile.stats.trainingCompleted")}
        />
        <StatTile
          value={formatCompactNumber(training.steps)}
          label={t("settings.profile.stats.trainingSteps")}
        />
        <StatTile
          value={formatCompactNumber(training.tokens)}
          label={t("settings.profile.stats.trainingTokens")}
        />
        <StatTile
          value={formatDuration(training.seconds)}
          label={t("settings.profile.stats.trainingTime")}
        />
        <StatTile
          value={
            training.bestLoss === null ? "—" : training.bestLoss.toFixed(3)
          }
          label={t("settings.profile.stats.bestLoss")}
        />
      </div>

      {training.recent.length > 0 ? (
        <ul className="flex flex-col divide-y divide-border/60 border-t border-border/60 pt-1">
          {training.recent.map((run) => (
            <li
              key={run.id}
              className="flex items-center justify-between gap-3 py-2"
            >
              <div className="flex min-w-0 flex-col gap-0.5">
                {/* A renamed run leads with the name the user chose, so the
                    model moves down beside the dataset to stay visible. */}
                <span
                  className="min-w-0 truncate text-sm text-foreground"
                  title={run.name}
                >
                  {run.name}
                </span>
                <span className="min-w-0 truncate text-ui-11 text-muted-foreground">
                  {run.name === run.modelLabel
                    ? run.datasetLabel
                    : `${run.modelLabel} · ${run.datasetLabel}`}
                </span>
              </div>
              <div className="flex shrink-0 items-center gap-4 text-xs tabular-nums">
                <span className="text-muted-foreground">
                  {t("settings.profile.stats.runSteps", {
                    steps: formatProfileCount(run.steps, "step", locale),
                  })}
                </span>
                <span className="text-muted-foreground">
                  {run.finalLoss === null
                    ? "—"
                    : t("settings.profile.stats.runLoss", {
                        loss: run.finalLoss.toFixed(3),
                      })}
                </span>
                <span
                  className={cn(
                    "w-16 text-right",
                    STATUS_TONE[run.status] ?? "text-muted-foreground",
                  )}
                >
                  {run.status}
                </span>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </StatsCard>
  );
}
