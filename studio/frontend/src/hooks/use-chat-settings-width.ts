// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createPanelWidthStore } from "./use-panel-width.ts";

/** The previous fixed 17rem, at a 16px root font size. */
export const CHAT_SETTINGS_WIDTH_DEFAULT = 272;
/** Below this the sliders and their value pills start colliding. */
export const CHAT_SETTINGS_WIDTH_MIN = 248;
export const CHAT_SETTINGS_WIDTH_MAX = 560;

const store = createPanelWidthStore({
  key: "chat_settings_width",
  min: CHAT_SETTINGS_WIDTH_MIN,
  max: CHAT_SETTINGS_WIDTH_MAX,
  fallback: CHAT_SETTINGS_WIDTH_DEFAULT,
});

export const clampChatSettingsWidth = store.clamp;
export const useChatSettingsWidth = store.useWidth;
