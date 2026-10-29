// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "../lib/api-base.ts";
import { createPanelWidthStore } from "./use-panel-width.ts";

/** The previous fixed 17.5rem, at a 16px root font size. */
export const SIDEBAR_WIDTH_DEFAULT = 280;
/** Fits the web header without truncating the wordmark, plus Firefox's ~3px wider heading. */
export const SIDEBAR_WIDTH_MIN_WEB = 260;
/** The desktop collapse button lives in the titlebar. */
export const SIDEBAR_WIDTH_MIN_DESKTOP = 224;
export const SIDEBAR_WIDTH_MIN = isTauri
  ? SIDEBAR_WIDTH_MIN_DESKTOP
  : SIDEBAR_WIDTH_MIN_WEB;
export const SIDEBAR_WIDTH_MAX = 480;

const store = createPanelWidthStore({
  key: "sidebar_width",
  min: SIDEBAR_WIDTH_MIN,
  max: SIDEBAR_WIDTH_MAX,
  fallback: SIDEBAR_WIDTH_DEFAULT,
});

export const clampSidebarWidth = store.clamp;
export const useSidebarWidth = store.useWidth;
