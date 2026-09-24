// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A clock time today, a short date before that. */
export function formatCardTime(ts: number): string {
  const then = new Date(ts);
  const now = new Date();
  if (then.toDateString() === now.toDateString()) {
    return then.toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
    });
  }
  return then.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: then.getFullYear() === now.getFullYear() ? undefined : "numeric",
  });
}

export function formatSize(bytes: number | null): string | null {
  if (bytes === null) return null;
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (const next of units.slice(1)) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ${unit}`;
}

export function pluralize(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}
