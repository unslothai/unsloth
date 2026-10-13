// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const RAISED_SURFACE =
  "bg-white shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none";

export const CARD_SHADOW =
  "shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_16px_rgba(0,0,0,0.06)] dark:shadow-none";

// Light mode only: white thumbnails on a near-white page hid the old bg-only hover. Dark mode already
// reads through its accent fill.
export const CARD_HOVER =
  "relative after:pointer-events-none after:absolute after:inset-0 after:rounded-[inherit] after:transition-colors group-hover/library-card:after:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:group-hover/library-card:after:bg-transparent";

export const OVERLAY_CONTROL =
  "bg-white shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-neutral-700 dark:shadow-none";

export const GLASS_SURFACE =
  "bg-white/60 text-neutral-900 shadow-none ring-1 ring-white/50 backdrop-blur-md dark:bg-black/40 dark:text-white dark:ring-white/20";

export const GLASS_CONTROL = `${GLASS_SURFACE} hover:bg-white/80 hover:text-neutral-900 dark:hover:bg-black/55 dark:hover:text-white`;
