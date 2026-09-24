// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The new-chat composer's surface: white with its soft shadow, and the card color in dark mode. */
export const RAISED_SURFACE =
  "bg-white shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none";

/** The round controls that float over a card: the ⋯ button and the select dot. */
export const OVERLAY_CONTROL =
  "bg-white shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-neutral-700 dark:shadow-none";

/** The same controls over a picture, which can be any color: frosted glass, legible on light and dark
 *  alike. */
export const GLASS_CONTROL =
  "bg-black/40 text-white shadow-none ring-1 ring-white/20 backdrop-blur-md hover:bg-black/55 hover:text-white dark:bg-black/40 dark:text-white dark:hover:bg-black/55";
