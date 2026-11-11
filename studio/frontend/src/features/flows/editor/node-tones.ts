// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CSSProperties } from "react";
import { portInk, studioInk } from "./theme-colors";

// Recipe Studio's icon tile tones (utils/ui-tones.ts), one per Flows category.
export const CATEGORY_TONES: Record<string, string> = {
  input: "bg-slate-50 text-slate-700 border-slate-200 dark:bg-slate-900/40 dark:text-slate-300 dark:border-slate-800",
  text: "bg-violet-50 text-violet-700 border-violet-100 dark:bg-violet-950/30 dark:text-violet-300 dark:border-violet-900/60",
  logic: "bg-amber-50 text-amber-700 border-amber-100 dark:bg-amber-950/30 dark:text-amber-300 dark:border-amber-900/60",
  transform: "bg-orange-50 text-orange-700 border-orange-100 dark:bg-orange-950/30 dark:text-orange-300 dark:border-orange-900/60",
  knowledge: "bg-teal-50 text-teal-700 border-teal-100 dark:bg-teal-950/30 dark:text-teal-300 dark:border-teal-900/60",
  image: "bg-pink-50 text-pink-700 border-pink-100 dark:bg-pink-950/30 dark:text-pink-300 dark:border-pink-900/60",
  video: "bg-red-50 text-red-700 border-red-100 dark:bg-red-950/30 dark:text-red-300 dark:border-red-900/60",
  audio: "bg-emerald-50 text-emerald-700 border-emerald-100 dark:bg-emerald-950/30 dark:text-emerald-300 dark:border-emerald-900/60",
  data: "bg-blue-50 text-blue-700 border-blue-100 dark:bg-blue-950/30 dark:text-blue-300 dark:border-blue-900/60",
  table: "bg-cyan-50 text-cyan-700 border-cyan-100 dark:bg-cyan-950/30 dark:text-cyan-300 dark:border-cyan-900/60",
  automation: "bg-teal-50 text-teal-700 border-teal-100 dark:bg-teal-950/30 dark:text-teal-300 dark:border-teal-900/60",
  output: "bg-sky-50 text-sky-700 border-sky-100 dark:bg-sky-950/30 dark:text-sky-300 dark:border-sky-900/60",
  story: "bg-purple-50 text-purple-700 border-purple-100 dark:bg-purple-950/30 dark:text-purple-300 dark:border-purple-900/60",
  world: "bg-green-50 text-green-700 border-green-100 dark:bg-green-950/30 dark:text-green-300 dark:border-green-900/60",
  chat: "bg-teal-50 text-teal-700 border-teal-100 dark:bg-teal-950/30 dark:text-teal-300 dark:border-teal-900/60",
  memory: "bg-teal-50 text-teal-700 border-teal-100 dark:bg-teal-950/30 dark:text-teal-300 dark:border-teal-900/60",
};

/** The same tile tone from a hex, for categories the table above does not know. */
export function accentTone(color: string): CSSProperties {
  const ink = studioInk(color);
  return {
    color: portInk(color),
    background: `color-mix(in oklab, ${ink} 10%, transparent)`,
    borderColor: `color-mix(in oklab, ${ink} 22%, transparent)`,
  };
}
