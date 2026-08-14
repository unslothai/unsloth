// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createPanelWidthStore } from "./use-panel-width.ts";

/** The previous fixed 17.5rem, at a 16px root font size. */
export const SIDEBAR_WIDTH_DEFAULT = 280;
/** Narrowest width that still fits the wordmark. Firefox is the constraint:
 * it renders the heading ~3px wider than Chromium and WebKit. */
export const SIDEBAR_WIDTH_MIN = 260;
export const SIDEBAR_WIDTH_MAX = 480;

const store = createPanelWidthStore({
  key: "sidebar_width",
  min: SIDEBAR_WIDTH_MIN,
  max: SIDEBAR_WIDTH_MAX,
  fallback: SIDEBAR_WIDTH_DEFAULT,
});

export const clampSidebarWidth = store.clamp;
export const useSidebarWidth = store.useWidth;
