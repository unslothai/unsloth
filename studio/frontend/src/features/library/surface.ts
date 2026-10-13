// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const RAISED_SURFACE =
  "bg-white shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none";

export const CARD_SHADOW =
  "shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_16px_rgba(0,0,0,0.06)] dark:shadow-none";

// Hover-revealed card controls. Light mode outlines them and deepens the shadow: white chips
// vanished on white thumbnails and on the hovered card's neutral-100.
const CONTROL_EDGE =
  "shadow-[0_2px_10px_-2px_rgba(0,0,0,0.24)] ring-1 ring-black/[0.035] dark:shadow-none";

export const OVERLAY_CONTROL = `bg-white ${CONTROL_EDGE} dark:bg-neutral-700 dark:ring-0`;

// Solid, not frosted: the blur smeared the thumbnail behind each control.
export const GLASS_SURFACE = `bg-white text-neutral-900 ${CONTROL_EDGE} dark:bg-neutral-700 dark:text-white dark:ring-0`;

export const GLASS_CONTROL = `${GLASS_SURFACE} hover:bg-neutral-100 hover:text-neutral-900 dark:hover:bg-neutral-600 dark:hover:text-white`;
