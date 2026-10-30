// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  trainingScheduleDraftValid,
  useScheduledTrainingStore,
} from "@/features/training";
import { useLocale, useT } from "@/i18n";
import {
  formatClock,
  localDateKey,
  timeChoices,
} from "@/lib/schedule-time";
import type { ReactElement } from "react";
import { useState } from "react";
import { useShallow } from "zustand/react/shallow";

export function ScheduleStartControl({
  disabled,
}: {
  disabled: boolean;
}): ReactElement {
  const t = useT();
  const locale = useLocale();
  const { enabled, date, time, setEnabled, setDate, setTime } =
    useScheduledTrainingStore(
      useShallow((state) => ({
        enabled: state.draftEnabled,
        date: state.draftDate,
        time: state.draftTime,
        setEnabled: state.setDraftEnabled,
        setDate: state.setDraftDate,
        setTime: state.setDraftTime,
      })),
    );

  const [now] = useState(() => new Date());
  const valid = trainingScheduleDraftValid(date, time, now);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-3">
        <label
          htmlFor="training-start-at"
          className="text-ui-12 font-medium text-foreground"
        >
          {t("studio.schedule.startAt")}
        </label>
        <Switch
          id="training-start-at"
          checked={enabled}
          onCheckedChange={setEnabled}
          disabled={disabled}
        />
      </div>
      {enabled && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <Input
              type="date"
              value={date}
              min={localDateKey(now)}
              onChange={(event) => setDate(event.target.value)}
              aria-label={t("studio.schedule.date")}
              className="h-8 min-w-0 flex-1 text-ui-12"
              disabled={disabled}
            />
            <Select value={time} onValueChange={setTime} disabled={disabled}>
              <SelectTrigger
                size="sm"
                aria-label={t("studio.schedule.time")}
                className="shrink-0 text-ui-12"
              >
                <SelectValue>{formatClock(time, locale)}</SelectValue>
              </SelectTrigger>
              <SelectContent
                align="end"
                className="max-h-[min(--spacing(72),var(--radix-select-content-available-height))]"
              >
                {timeChoices(time).map((choice) => (
                  <SelectItem key={choice} value={choice}>
                    {formatClock(choice, locale)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {!valid && (
            <p role="status" className="text-ui-10p5 text-status-warning">
              {t("studio.schedule.pickDateTime")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
