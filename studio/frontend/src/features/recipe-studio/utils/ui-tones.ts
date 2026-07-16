// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const RECIPE_STUDIO_NODE_TONES = {
  sampler:
    "bg-emerald-50 text-emerald-700 border-emerald-100 dark:bg-emerald-950/30 dark:text-emerald-300 dark:border-emerald-900/60",
  llm:
    "bg-sky-50 text-sky-700 border-sky-100 dark:bg-sky-950/30 dark:text-sky-300 dark:border-sky-900/60",
  validator:
    "bg-rose-50 text-rose-700 border-rose-100 dark:bg-rose-950/30 dark:text-rose-300 dark:border-rose-900/60",
  expression:
    "bg-indigo-50 text-indigo-700 border-indigo-100 dark:bg-indigo-950/30 dark:text-indigo-300 dark:border-indigo-900/60",
  note:
    "bg-violet-50 text-violet-700 border-violet-100 dark:bg-violet-950/30 dark:text-violet-300 dark:border-violet-900/60",
  seed:
    "bg-lime-50 text-lime-700 border-lime-100 dark:bg-lime-950/30 dark:text-lime-300 dark:border-lime-900/60",
  model_provider:
    "bg-amber-50 text-amber-700 border-amber-100 dark:bg-amber-950/30 dark:text-amber-300 dark:border-amber-900/60",
  model_config:
    "bg-orange-50 text-orange-700 border-orange-100 dark:bg-orange-950/30 dark:text-orange-300 dark:border-orange-900/60",
  tool_config:
    "bg-cyan-50 text-cyan-700 border-cyan-100 dark:bg-cyan-950/30 dark:text-cyan-300 dark:border-cyan-900/60",
} as const;

export const RECIPE_STUDIO_USER_NODE_TONE =
  "bg-amber-50 text-amber-700 border-amber-100 dark:bg-amber-950/30 dark:text-amber-300 dark:border-amber-900/60";

export const RECIPE_STUDIO_REFERENCE_BADGE_TONES = {
  user:
    "corner-squircle border-amber-500/25 bg-amber-500/10 font-mono text-[11px] text-amber-700 dark:text-amber-300",
  seed:
    "corner-squircle border-blue-500/25 bg-blue-500/10 font-mono text-[11px] text-blue-700 dark:text-blue-300",
  default: "corner-squircle font-mono text-[11px]",
} as const;

export const RECIPE_STUDIO_WARNING_BADGE_TONE =
  "border-amber-500/40 bg-amber-500/10 text-amber-700 hover:bg-amber-500/20 dark:text-amber-300";

export const RECIPE_STUDIO_WARNING_ICON_TONE =
  "text-amber-600 dark:text-amber-400";

export const RECIPE_STUDIO_ONBOARDING_SURFACE_TONE =
  "border-primary/20 bg-primary/[0.045]";

export const RECIPE_STUDIO_ONBOARDING_ICON_TONE =
  "bg-primary/10 text-primary";
