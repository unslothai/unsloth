// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Values at a dotted path of a route's answer; `name[]` maps over a list (`tasks[].id`). */
export function pickOptions(data: unknown, path: string | undefined): string[] {
  let values: unknown[] = [data];
  for (const raw of (path ?? "").split(".").filter(Boolean)) {
    const each = raw.endsWith("[]");
    const key = each ? raw.slice(0, -2) : raw;
    values = values.flatMap((v) => {
      const next =
        key && v && typeof v === "object"
          ? (v as Record<string, unknown>)[key]
          : v;
      return each ? (Array.isArray(next) ? next : []) : [next];
    });
  }
  const flat =
    values.length === 1 && Array.isArray(values[0]) ? values[0] : values;
  return flat
    .filter((v) => typeof v === "string" || typeof v === "number")
    .map(String)
    .slice(0, 2000);
}
