// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The Latest feed's order: trending, decayed by age, as Hacker News ranks stories. */

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
/** At least this many likes, nearly all of them recent, before likes may vouch for the date. */
const RECENT_LIKES_MIN = 5;
const RECENT_LIKES_SHARE = 0.8;

function ageDays(value: string | Date | undefined, now: number): number | null {
  if (value === undefined) {
    return null;
  }
  const time = new Date(value).getTime();
  return Number.isFinite(time) ? Math.max(0, (now - time) / DAY_MS) : null;
}

/** The Hub has no publish date and createdAt predates a private-until-launch repo going public; nearly all-recent likes mean it just went public, so its last change dates it. */
export function releaseAgeDays(model: FreshRankable, now: number): number {
  const created = ageDays(model.createdAt, now);
  const updated = ageDays(model.updatedAt, now);
  const likes = model.likes ?? 0;
  const trending = model.trendingScore ?? 0;
  const recentlyPublic =
    likes >= RECENT_LIKES_MIN && trending >= RECENT_LIKES_SHARE * likes;
  if (recentlyPublic && updated !== null) {
    return created === null ? updated : Math.min(created, updated);
  }
  return created ?? updated ?? Number.POSITIVE_INFINITY;
}

export function freshScore(model: FreshRankable, now: number): number {
  const age = releaseAgeDays(model, now);
  return (
    ((model.trendingScore ?? 0) + TREND_OFFSET) /
    (age + AGE_OFFSET_DAYS) ** GRAVITY
  );
}

export function rankFresh<T extends FreshRankable>(models: readonly T[], now: number): T[] {
  return models
    .map((model) => ({ model, score: freshScore(model, now) }))
    .sort((a, b) => b.score - a.score)
    .map(({ model }) => model);
}
