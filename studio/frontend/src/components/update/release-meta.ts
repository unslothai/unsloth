// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Locale, formatRelativeTime } from "@/i18n";
import { releases } from "./releases";
import { displayVersion } from "./update-offer-store";

export type Release = (typeof releases)[number];

export const releaseUrl = (version: string) =>
  `https://github.com/unslothai/unsloth/releases/tag/v${displayVersion(version)}-beta`;

export const sameVersion = (a: string, b: string) =>
  displayVersion(a) === displayVersion(b);

const TAG = /^(\d+)\.(\d+)\.(\d+)$/;

function parts(version: string): number[] | null {
  const match = TAG.exec(displayVersion(version));
  return match ? match.slice(1).map(Number) : null;
}

/** Null when either side is not an x.y.z tag, as PyPI date versions are not. */
export function compareVersions(a: string, b: string): number | null {
  const left = parts(a);
  const right = parts(b);
  if (!(left && right)) {
    return null;
  }
  for (let i = 0; i < 3; i++) {
    if (left[i] !== right[i]) {
      return left[i] - right[i];
    }
  }
  return 0;
}

/** Bundled releases after `current` up to and including `latest`, newest first. */
export function releasesBetween(current: string, latest: string): Release[] {
  return releases.filter((release) => {
    const afterCurrent = compareVersions(release.version, current);
    const upToLatest = compareVersions(release.version, latest);
    return (
      afterCurrent !== null &&
      afterCurrent > 0 &&
      upToLatest !== null &&
      upToLatest <= 0
    );
  });
}

const DAY_MS = 86_400_000;

/** "today", "3 days ago", "2 wk. ago": release dates are whole UTC days. */
export function releaseAge(
  date: string,
  locale: Locale,
  now = Date.now(),
): string {
  const start = Date.parse(`${date}T00:00:00Z`);
  const today = Math.floor(now / DAY_MS) * DAY_MS;
  const days = Math.round((today - start) / DAY_MS);
  if (days < 14) {
    return formatRelativeTime(locale, -Math.max(0, days), "day");
  }
  if (days < 60) {
    return formatRelativeTime(locale, -Math.round(days / 7), "week");
  }
  return formatRelativeTime(locale, -Math.round(days / 30), "month");
}
