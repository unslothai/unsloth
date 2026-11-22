// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  getAuthSessionEpoch,
} from "@/features/auth";
import { localMinuteKey, nextQuarterHour } from "@/lib/schedule-time.ts";
import { create } from "zustand";
import {
  cancelTrainingSchedule,
  createTrainingSchedule,
  listTrainingSchedules,
} from "../api/train-api";
import type {
  TrainingSchedule,
  TrainingScheduleInput,
} from "../types/api";

interface ScheduledTrainingState {
  schedules: TrainingSchedule[];
  error: string | null;
  creating: boolean;
  cancellingId: string | null;
  dismissedId: string | null;
  draftEnabled: boolean;
  draftDate: string;
  draftTime: string;
  refresh: () => Promise<void>;
  create: (input: TrainingScheduleInput) => Promise<TrainingSchedule>;
  cancel: (id: string) => Promise<void>;
  dismiss: (id: string) => void;
  setDraftEnabled: (enabled: boolean) => void;
  setDraftDate: (date: string) => void;
  setDraftTime: (time: string) => void;
}

let refreshGeneration = 0;

const SESSION_RESET = {
  schedules: [],
  error: null,
  creating: false,
  cancellingId: null,
  dismissedId: null,
  draftEnabled: false,
} satisfies Partial<ScheduledTrainingState>;

const initialDraft = nextQuarterHour(new Date());

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export const useScheduledTrainingStore = create<ScheduledTrainingState>()(
  (set) => ({
    ...SESSION_RESET,
    draftDate: initialDraft.date,
    draftTime: initialDraft.time,
    refresh: async () => {
      const generation = ++refreshGeneration;
      try {
        const { schedules } = await listTrainingSchedules();
        if (generation === refreshGeneration) set({ schedules, error: null });
      } catch (error) {
        if (generation === refreshGeneration) set({ error: errorText(error) });
      }
    },
    create: async (input) => {
      const epoch = getAuthSessionEpoch();
      set({ creating: true, error: null });
      try {
        const schedule = await createTrainingSchedule(input);
        if (getAuthSessionEpoch() === epoch) {
          // A list read that started before this insert must not overwrite it.
          refreshGeneration += 1;
          set((state) => ({
            schedules: [
              schedule,
              ...state.schedules.filter((entry) => entry.id !== schedule.id),
            ],
            error: null,
            draftEnabled: false,
          }));
        }
        return schedule;
      } finally {
        set({ creating: false });
      }
    },
    cancel: async (id) => {
      const epoch = getAuthSessionEpoch();
      set({ cancellingId: id, error: null });
      try {
        await cancelTrainingSchedule(id);
        if (getAuthSessionEpoch() !== epoch) return;
        // A list read that started before the cancel must not show the row as waiting again.
        refreshGeneration += 1;
        set((state) => ({
          schedules: state.schedules.map((schedule) =>
            schedule.id === id
              ? { ...schedule, status: "cancelled", updatedAt: Date.now() }
              : schedule,
          ),
        }));
      } catch (error) {
        if (getAuthSessionEpoch() === epoch) set({ error: errorText(error) });
      } finally {
        set({ cancellingId: null });
      }
    },
    dismiss: (id) => set({ dismissedId: id }),
    setDraftEnabled: (enabled) =>
      set((state) => {
        if (!enabled) return { draftEnabled: false };
        const now = new Date();
        // A draft left over from an earlier visit may already be in the past.
        const stale = `${state.draftDate}T${state.draftTime}` < localMinuteKey(now);
        const next = stale ? nextQuarterHour(now) : null;
        return {
          draftEnabled: true,
          ...(next && { draftDate: next.date, draftTime: next.time }),
        };
      }),
    setDraftDate: (date) => set({ draftDate: date }),
    setDraftTime: (time) => set({ draftTime: time }),
  }),
);

if (typeof window !== "undefined") {
  // Module state: a sign-out must drop it or the next account sees these starts.
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    refreshGeneration += 1;
    useScheduledTrainingStore.setState(SESSION_RESET);
  });
}
