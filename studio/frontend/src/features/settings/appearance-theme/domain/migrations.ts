// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CURRENT_THEME_SCHEMA_VERSION } from "./schema.ts";

type RawTheme = Record<string, unknown>;

/**
 * Upgrades a raw file from version N to N+1, keyed by N. Empty while v1 is the only schema; a
 * later bump adds a step here rather than changing the reader.
 */
export const THEME_MIGRATIONS: Record<number, (raw: RawTheme) => RawTheme> = {};

export function migrateThemeRaw(raw: RawTheme, fromVersion: number): RawTheme {
  let next = raw;
  for (
    let version = fromVersion;
    version < CURRENT_THEME_SCHEMA_VERSION;
    version += 1
  ) {
    const step = THEME_MIGRATIONS[version];
    if (step) next = step(next);
  }
  return next;
}
