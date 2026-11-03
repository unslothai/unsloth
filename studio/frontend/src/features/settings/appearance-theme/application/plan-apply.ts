// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type AppearanceCustomization,
  type ImportedFont,
  sanitizeCustomization,
} from "../../lib/appearance-customization.ts";
import { LAYOUT_FIELDS } from "../domain/classification.ts";
import type { AppearanceTheme, ThemeGlass } from "../domain/schema.ts";
import type {
  AppearanceSnapshot,
  ThemeApplyPlan,
  ThemeChange,
} from "./types.ts";

export type ApplyOptions = {
  applyLayout: boolean;
  applyAssets: boolean;
};

type CustomizationField = keyof AppearanceCustomization;

const TYPOGRAPHY_FIELDS: readonly CustomizationField[] = [
  "uiFont",
  "headingFont",
  "chatFont",
  "codeFont",
  "uiFontSize",
  "codeFontSize",
  "fontSmoothing",
];
const CODE_FIELDS: readonly CustomizationField[] = [
  "codeLineNumbers",
  "codeWordWrap",
  "codeLigatures",
];
const CHAT_FIELDS: readonly CustomizationField[] = [
  "chatWidth",
  "composerOutline",
];
const COLOR_FIELDS: readonly CustomizationField[] = ["colors", "contrast"];
const FONT_FIELDS: readonly CustomizationField[] = ["importedFonts"];

/** Order of the changes list shown to the user. */
const CHANGE_ORDER: readonly ThemeChange[] = [
  "palette",
  "colors",
  "typography",
  "code",
  "chat",
  "glass",
  "layout",
  "fonts",
  "background",
];

function groupDiffers(
  current: AppearanceCustomization,
  next: AppearanceCustomization,
  fields: readonly CustomizationField[],
): boolean {
  const pick = (c: AppearanceCustomization) => fields.map((field) => c[field]);
  return JSON.stringify(pick(current)) !== JSON.stringify(pick(next));
}

function glassKey(glass: ThemeGlass): string {
  return JSON.stringify([
    glass.material,
    glass.strength.light,
    glass.strength.dark,
    glass.lens,
    glass.menus,
  ]);
}

/** Theme fonts first, then the current fonts the theme does not replace by name. */
function mergeFonts(
  themeFonts: ImportedFont[],
  currentFonts: ImportedFont[],
): ImportedFont[] {
  const names = new Set(themeFonts.map((font) => font.name));
  return [
    ...themeFonts,
    ...currentFonts.filter((font) => !names.has(font.name)),
  ];
}

/**
 * Computes what applying a parsed theme would do, without touching the app. Personal fields
 * always come from the current customization. A theme without a section resets that section's
 * fields to their defaults, because the parser fills every look field.
 */
export function planThemeApply(
  current: AppearanceSnapshot,
  theme: AppearanceTheme,
  opts: ApplyOptions,
): ThemeApplyPlan {
  const importedFonts =
    opts.applyAssets && theme.fonts
      ? mergeFonts(theme.fonts, current.customization.importedFonts)
      : current.customization.importedFonts;
  const customization = sanitizeCustomization({
    ...current.customization,
    ...theme.look,
    ...(opts.applyLayout && theme.layout ? theme.layout : {}),
    importedFonts,
  });
  const glass: ThemeGlass = {
    material: theme.glass.material,
    strength: {
      light: theme.glass.strength.light,
      dark: theme.glass.strength.dark,
    },
    lens: theme.glass.lens,
    menus: theme.glass.menus,
  };
  const background =
    opts.applyAssets && theme.background ? theme.background : "keep";

  const differs: Record<ThemeChange, boolean> = {
    palette: current.palette !== theme.palette,
    colors: groupDiffers(current.customization, customization, COLOR_FIELDS),
    typography: groupDiffers(
      current.customization,
      customization,
      TYPOGRAPHY_FIELDS,
    ),
    code: groupDiffers(current.customization, customization, CODE_FIELDS),
    chat: groupDiffers(current.customization, customization, CHAT_FIELDS),
    glass: glassKey(current.glass) !== glassKey(glass),
    layout: groupDiffers(
      current.customization,
      customization,
      LAYOUT_FIELDS,
    ),
    fonts: groupDiffers(current.customization, customization, FONT_FIELDS),
    background: background !== "keep",
  };

  return {
    palette: theme.palette,
    customization,
    glass,
    background,
    changes: CHANGE_ORDER.filter((change) => differs[change]),
  };
}
