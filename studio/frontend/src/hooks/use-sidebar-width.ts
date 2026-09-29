// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createPanelWidthStore } from "./use-panel-width.ts";

/** The previous fixed 17.5rem, at a 16px root font size. */
export const SIDEBAR_WIDTH_DEFAULT = 280;
/** Narrowest width that fits the header (logo, wordmark, BETA, search) at the
 * default font size, with room for Firefox's ~3px wider heading. */
export const SIDEBAR_WIDTH_MIN = 224;
export const SIDEBAR_WIDTH_MAX = 480;

const store = createPanelWidthStore({
  key: "sidebar_width",
  min: SIDEBAR_WIDTH_MIN,
  max: SIDEBAR_WIDTH_MAX,
  fallback: SIDEBAR_WIDTH_DEFAULT,
});

export const clampSidebarWidth = store.clamp;
export const useSidebarWidth = store.useWidth;
