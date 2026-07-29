// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** One day of the activity series. Dense: every day in range is present. */
export type ProfileStatsDay = {
  date: string;
  tokens: number;
  messages: number;
  chats: number;
};

export type ProfileStatsModel = {
  id: string;
  label: string;
  messages: number;
  tokens: number;
};

export type ProfileStatsRun = {
  id: string;
  name: string;
  modelLabel: string;
  datasetLabel: string;
  status: string;
  finalLoss: number | null;
  steps: number;
  seconds: number;
  startedAt: string | null;
};

export type ProfileStats = {
  generatedAt: number;
  days: number;
  totals: {
    threads: number;
    messages: number;
    userMessages: number;
    assistantMessages: number;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    cachedTokens: number;
    toolCalls: number;
    attachments: number;
    activeDays: number;
    chatSeconds: number;
  };
  streak: {
    current: number;
    longest: number;
    lastActiveDay: string | null;
  };
  peakDay: { date: string; tokens: number } | null;
  longestChat: {
    threadId: string | null;
    title: string | null;
    seconds: number;
    messages: number;
  } | null;
  daily: ProfileStatsDay[];
  /** Messages per hour of day, index 0..23. */
  hourly: number[];
  /** Messages per weekday, index 0 = Monday. */
  weekday: number[];
  models: ProfileStatsModel[];
  speed: {
    averageTokensPerSecond: number | null;
    bestTokensPerSecond: number | null;
    bestTokensPerSecondModel: string | null;
    averageResponseMs: number | null;
    averageFirstTokenMs: number | null;
    samples: number;
  };
  training: {
    runs: number;
    completed: number;
    steps: number;
    tokens: number;
    seconds: number;
    models: number;
    datasets: number;
    bestLoss: number | null;
    recent: ProfileStatsRun[];
  };
};

export async function loadProfileStats(
  signal?: AbortSignal,
): Promise<ProfileStats> {
  const res = await authFetch("/api/profile/stats", { signal });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load your stats"));
  }
  return (await res.json()) as ProfileStats;
}
