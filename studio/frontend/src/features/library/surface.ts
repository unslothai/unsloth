// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The new-chat composer's surface: white with its soft shadow, and the card color in dark mode. */
export const RAISED_SURFACE =
  "bg-white shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none";

/** The round controls that float over a card: the ⋯ button and the select dot. */
export const OVERLAY_CONTROL =
  "bg-white shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-neutral-700 dark:shadow-none";

/** Frosted glass over a picture, which can be any color: white in light mode, black in dark. */
export const GLASS_SURFACE =
  "bg-white/60 text-neutral-900 shadow-none ring-1 ring-white/50 backdrop-blur-md dark:bg-black/40 dark:text-white dark:ring-white/20";

/** The card's round controls (the ⋯ button and the select dot) over a picture. */
export const GLASS_CONTROL = `${GLASS_SURFACE} hover:bg-white/80 hover:text-neutral-900 dark:hover:bg-black/55 dark:hover:text-white`;
