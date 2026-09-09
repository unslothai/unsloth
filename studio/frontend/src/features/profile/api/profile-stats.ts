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
    chatPromptTokens: number;
    chatCompletionTokens: number;
    chatTokens: number;
    apiPromptTokens: number;
    apiCompletionTokens: number;
    apiTokens: number;
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
  // Bucket days and hours in this browser's timezone, which is not the
  // server's when Unsloth is reached over the network. The IANA name is what
  // gives each historical date its own daylight-saving offset; the current
  // offset only covers callers whose host cannot resolve the name.
  const query = new URLSearchParams({
    tz_offset_minutes: String(new Date().getTimezoneOffset()),
    tz: Intl.DateTimeFormat().resolvedOptions().timeZone ?? "",
  });
  const res = await authFetch(`/api/profile/stats?${query}`, { signal });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load your stats"));
  }
  return (await res.json()) as ProfileStats;
}
