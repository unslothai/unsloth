// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Capability } from "./capabilities";

/**
 * Per-model accent slug. Picked from the model's first matching capability so
 * the inspector "feels like" the model — vision models get an indigo wash,
 * code models cyan, etc. Falls back to a neutral slate when nothing matches.
 *
 * Class strings are kept in static records so Tailwind's class extractor can
 * see every one of them. Don't compose accent classes dynamically from the
 * slug — always read through the records.
 */
export type AccentSlug =
  | "indigo"
  | "cyan"
  | "violet"
  | "rose"
  | "amber"
  | "emerald"
  | "sky"
  | "slate";

const CAPABILITY_PRIORITY: Array<[Capability["key"], AccentSlug]> = [
  ["vision", "indigo"],
  ["reasoning", "violet"],
  ["code", "cyan"],
  ["audio", "rose"],
  ["tools", "amber"],
  ["embedding", "emerald"],
  ["multilingual", "sky"],
];

export function pickAccent(capabilities: Capability[] | undefined): AccentSlug {
  if (!capabilities || capabilities.length === 0) return "slate";
  const set = new Set(capabilities.map((c) => c.key));
  for (const [key, slug] of CAPABILITY_PRIORITY) {
    if (set.has(key)) return slug;
  }
  return "slate";
}

/** Top-of-inspector hairline rule. */
export const ACCENT_RULE: Record<AccentSlug, string> = {
  indigo: "bg-indigo-500/70",
  cyan: "bg-cyan-500/70",
  violet: "bg-violet-500/70",
  rose: "bg-rose-500/70",
  amber: "bg-amber-500/70",
  emerald: "bg-emerald-500/70",
  sky: "bg-sky-500/70",
  slate: "bg-slate-500/40",
};

/** Soft gradient wash behind the title — fades top→bottom, never opaque. */
export const ACCENT_WASH: Record<AccentSlug, string> = {
  indigo:
    "bg-[radial-gradient(120%_80%_at_0%_0%,theme(colors.indigo.500/0.18),transparent_60%)]",
  cyan: "bg-[radial-gradient(120%_80%_at_0%_0%,theme(colors.cyan.500/0.16),transparent_60%)]",
  violet:
    "bg-[radial-gradient(120%_80%_at_0%_0%,theme(colors.violet.500/0.18),transparent_60%)]",
  rose: "bg-[radial-gradient(120%_80%_at_0%_0%,theme(colors.rose.500/0.16),transparent_60%)]",
  amber:
    "bg-[radial-gradient(120%_80%_at_0%_0%,theme(colors.amber.500/0.16),transparent_60%)]",
  emerald:
    "bg-[radial-gradient(120%_80%_at_0%_0%,theme(colors.emerald.500/0.16),transparent_60%)]",
  sky: "bg-[radial-gradient(120%_80%_at_0%_0%,theme(colors.sky.500/0.16),transparent_60%)]",
  slate:
    "bg-[radial-gradient(120%_80%_at_0%_0%,theme(colors.slate.500/0.10),transparent_60%)]",
};

/** Foreground accent (text + ring). Used for the Loaded chip / focus ring. */
export const ACCENT_TEXT: Record<AccentSlug, string> = {
  indigo: "text-indigo-600 dark:text-indigo-400",
  cyan: "text-cyan-700 dark:text-cyan-300",
  violet: "text-violet-600 dark:text-violet-400",
  rose: "text-rose-600 dark:text-rose-400",
  amber: "text-amber-700 dark:text-amber-400",
  emerald: "text-emerald-700 dark:text-emerald-400",
  sky: "text-sky-700 dark:text-sky-400",
  slate: "text-slate-600 dark:text-slate-300",
};

export const ACCENT_RING: Record<AccentSlug, string> = {
  indigo: "ring-indigo-500/30",
  cyan: "ring-cyan-500/30",
  violet: "ring-violet-500/30",
  rose: "ring-rose-500/30",
  amber: "ring-amber-500/30",
  emerald: "ring-emerald-500/30",
  sky: "ring-sky-500/30",
  slate: "ring-slate-500/20",
};
