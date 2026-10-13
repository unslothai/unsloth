// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const RAISED_SURFACE =
  "bg-white shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none";

export const CARD_SHADOW =
  "shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_16px_rgba(0,0,0,0.06)] dark:shadow-none";

// Hover-revealed card controls. Light mode fills them grey: white chips vanished on white
// thumbnails and on the hovered card's neutral-100.
export const OVERLAY_CONTROL =
  "bg-neutral-200 shadow-none hover:bg-neutral-300 dark:bg-neutral-700";

export const GLASS_SURFACE =
  "bg-neutral-200/85 text-neutral-900 shadow-none ring-1 ring-black/5 backdrop-blur-md dark:bg-black/40 dark:text-white dark:ring-white/20";

export const GLASS_CONTROL = `${GLASS_SURFACE} hover:bg-neutral-300/90 hover:text-neutral-900 dark:hover:bg-black/55 dark:hover:text-white`;
