// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BackgroundAdjustments } from "../../../chat-background/model.ts";
import type { AppearanceCustomization } from "../../lib/appearance-customization.ts";
import type { ColorThemeId } from "../../lib/color-themes.ts";
import type { ThemeBackground, ThemeGlass } from "../domain/schema.ts";

/** The appearance the app shows right now, read back so an import can be undone. */
export type AppearanceSnapshot = {
  palette: ColorThemeId;
  customization: AppearanceCustomization;
  glass: ThemeGlass;
  background: { adjustments: BackgroundAdjustments; blob: Blob } | null;
};

export type ThemeChange =
  | "palette"
  | "colors"
  | "typography"
  | "code"
  | "chat"
  | "glass"
  | "layout"
  | "fonts"
  | "background";

export type ThemeApplyPlan = {
  palette: ColorThemeId;
  customization: AppearanceCustomization;
  glass: ThemeGlass;
  background: "keep" | ThemeBackground;
  changes: ThemeChange[];
};
