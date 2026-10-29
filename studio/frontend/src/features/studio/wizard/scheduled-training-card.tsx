// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  type TrainingSchedule,
  useScheduledTrainingStore,
} from "@/features/training";
import { useLocale, useT } from "@/i18n";
import { formatLocalMinute, zoneLabel } from "@/lib/schedule-time";
import { cn } from "@/lib/utils";
import { Cancel01Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";

function scheduleMoment(
  schedule: TrainingSchedule,
  locale: string,
): { when: string; zone: string } {
  return {
    when: formatLocalMinute(schedule.startAtLocal, locale),
    zone: zoneLabel(schedule.timezone, locale, schedule.startAt),
  };
}

export function PendingScheduleCard({
  schedule,
}: {
  schedule: TrainingSchedule;
}): ReactElement {
  const t = useT();
  const locale = useLocale();
  const cancelling = useScheduledTrainingStore(
    (state) => state.cancellingId === schedule.id,
  );
  const error = useScheduledTrainingStore((state) => state.error);
  const cancel = useScheduledTrainingStore((state) => state.cancel);
  const { when, zone } = scheduleMoment(schedule, locale);

  return (
    <section className="flex flex-col gap-2 rounded-lg border border-border/60 bg-muted/40 px-3 py-2.5 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]">
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-start gap-2">
          <HugeiconsIcon
            icon={Clock01Icon}
            strokeWidth={1.75}
            className="mt-0.5 size-4 shrink-0 text-muted-foreground"
          />
          <div className="min-w-0">
            <p className="text-ui-11 font-medium text-foreground">
              {t("studio.schedule.pendingTitle")}
            </p>
            <p className="text-ui-10p5 leading-relaxed text-foreground/80">
              {when} · {zone}
            </p>
            <p className="truncate text-ui-10p5 text-muted-foreground">
              {schedule.modelName}
            </p>
            <p className="text-ui-10p5 leading-relaxed text-muted-foreground/80">
              {t("studio.schedule.pendingNote")}
            </p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="xs"
          className="shrink-0 text-muted-foreground"
          disabled={cancelling}
          onClick={() => void cancel(schedule.id)}
        >
          {cancelling ? t("studio.schedule.cancelling") : t("studio.schedule.cancel")}
        </Button>
      </div>
      {error && (
        <p role="alert" className="break-words text-ui-10p5 text-destructive">
          {error}
        </p>
      )}
    </section>
  );
}

/** A scheduled start that ended without training: the row is the only record, so say what happened. */
export function ScheduledTrainingProblemCard({
  schedule,
}: {
  schedule: TrainingSchedule;
}): ReactElement {
  const t = useT();
  const locale = useLocale();
  const dismiss = useScheduledTrainingStore((state) => state.dismiss);
  const missed = schedule.status === "missed";
  const { when, zone } = scheduleMoment(schedule, locale);

  return (
    <section
      className={cn(
        "flex flex-col gap-1 rounded-lg border px-3 py-2.5",
        "border-status-warning/25 bg-status-warning/[0.06]",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-ui-11 font-medium text-foreground">
          {t(missed ? "studio.schedule.missedTitle" : "studio.schedule.failedTitle")}
        </p>
        <Button
          variant="ghost"
          size="icon-xs"
          className="shrink-0 text-muted-foreground"
          aria-label={t("studio.schedule.dismiss")}
          onClick={() => dismiss(schedule.id)}
        >
          <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} />
        </Button>
      </div>
      <p className="text-ui-10p5 leading-relaxed text-foreground/80">
        {missed
          ? t("studio.schedule.missedDescription")
          : (schedule.error ?? t("studio.schedule.failedDescription"))}
      </p>
      <p className="text-ui-10p5 text-muted-foreground">
        {when} · {zone}
      </p>
    </section>
  );
}
