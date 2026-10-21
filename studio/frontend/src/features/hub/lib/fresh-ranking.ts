// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The Latest feed's order: trending, decayed by creation age, as Hacker News ranks stories. */

export interface FreshRankable {
  createdAt?: string | Date;
  updatedAt?: string | Date;
  likes?: number;
  trendingScore?: number;
}

const DAY_MS = 86_400_000;
/** Higher sinks older models faster; 1.6 still keeps a week-old hit above a day-old model nobody likes. */
const GRAVITY = 1.6;
const AGE_OFFSET_DAYS = 2;
const TREND_OFFSET = 2;

/** Days since the repo was created. Never last modified: a re-upload of a week-old model is not new. */
export function createdAgeDays(model: FreshRankable, now: number): number {
  if (model.createdAt === undefined) {
    return Number.POSITIVE_INFINITY;
  }
  const time = new Date(model.createdAt).getTime();
  return Number.isFinite(time)
    ? Math.max(0, (now - time) / DAY_MS)
    : Number.POSITIVE_INFINITY;
}

export function freshScore(model: FreshRankable, now: number): number {
  return (
    ((model.trendingScore ?? 0) + TREND_OFFSET) /
    (createdAgeDays(model, now) + AGE_OFFSET_DAYS) ** GRAVITY
  );
}

export function rankFresh<T extends FreshRankable>(
  models: readonly T[],
  now: number,
): T[] {
  return models
    .map((model) => ({ model, score: freshScore(model, now) }))
    .sort((a, b) => b.score - a.score)
    .map(({ model }) => model);
}
