// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isClockTime, isDateKey, localDateKey, pad } from "../../../lib/schedule-time.ts";
import type { TrainingSchedule } from "../types/api.ts";

export function isTrainingScheduleActive(schedule: TrainingSchedule): boolean {
  return schedule.status === "pending" || schedule.status === "starting";
}

/** The one start that is still waiting, if any; the list puts it first. */
export function activeTrainingSchedule(
  schedules: readonly TrainingSchedule[],
): TrainingSchedule | null {
  return schedules.find(isTrainingScheduleActive) ?? null;
}

/** The newest start that ended without training, unless the user dismissed it. */
export function latestTrainingScheduleProblem(
  schedules: readonly TrainingSchedule[],
  dismissedId: string | null,
): TrainingSchedule | null {
  return (
    schedules.find(
      (schedule) =>
        (schedule.status === "failed" || schedule.status === "missed") &&
        schedule.id !== dismissedId,
    ) ?? null
  );
}

/** A date and time the server will accept: a real wall clock that is still ahead. */
export function trainingScheduleDraftValid(
  date: string,
  time: string,
  now: Date = new Date(),
): boolean {
  if (!(isDateKey(date) && isClockTime(time))) {
    return false;
  }
  const nowMinute = `${localDateKey(now)}T${pad(now.getHours())}:${pad(now.getMinutes())}`;
  return `${date}T${time}` > nowMinute;
}
